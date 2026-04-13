from typing import Optional

import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn
from torch.distributions import Categorical

from src.layers import *


class gRNAde(torch.nn.Module):
    """gRNAde: Generative model for RNA inverse design.

    The model accepts flexible multi-modal inputs that can include:
        - Target pseudoknotted secondary structure (2D base pairing patterns)
        - 3D backbone coordinates from experimentally determined or predicted structures
        - Partial sequence constraints (e.g., functional motifs to preserve)
    All three input modalities are optional, allowing the model to design from 
    incomplete structural information.

    Architecture:
        The core architecture consists of an SE(3)-equivariant Graph Neural Network 
        with Geometric Vector Perceptron (GVP) layers. Each nucleotide is represented 
        as a node in a geometric graph with three types of edges:
        - Backbone connectivity: 32 nearest neighbors along the RNA chain (1D)
        - Base pairing: all nucleotides involved in pairs and pseudoknots (2D)
        - Spatial proximity: 32 nearest neighbors in 3D space (if structure provided)
        
        An encoder-decoder architecture processes these graphs:
        - GVP-GNN encoder layers process the structural information into latent 
          representations that capture both scalar and vector (geometric) features
        - Autoregressive GVP-GNN decoder generates sequences from 5' to 3', with 
          optional logit biasing to enforce sequence constraints

    Multi-conformation support:
        The model explicitly handles multiple structural conformations (multi-state 
        design), pooling features across conformations using masked mean pooling to 
        design sequences that fold consistently across different structural states.

    Training:
        Trained on 4,211 diverse RNA structures up to 500 nucleotides from the PDB 
        (resolution ≤4.0Å). A novel 3D dropout strategy during training enables 
        the model to design even without complete 3D information, relying on implicit 
        3D constraints from secondary structure alone.

    Usage:
        The standard forward pass requires sequence information as input and should 
        be used for training or evaluating likelihood. For sequence design and 
        sampling, use the `sample()` method which generates sequences autoregressively.

    Input format:
        Takes in RNA structure graphs of type `torch_geometric.data.Data` or 
        `torch_geometric.data.Batch` and returns a categorical distribution over 
        4 nucleotides at each position in a `torch.Tensor` of shape [n_nodes, 4].
        See data/featurizer.py for details on graph construction and features.

    Args:
        node_in_dim (tuple): Input node feature dimensions as (scalar_dim, vector_dim). 
            Nodes are featurized using 3-bead coarse-grained representation (P, C4', 
            N1/N9 atoms) and local geometric properties. Default: (15, 4).
        node_h_dim (tuple): Hidden node feature dimensions as (scalar_dim, vector_dim) 
            for GVP-GNN layers. Default: (128, 16).
        edge_in_dim (tuple): Input edge feature dimensions as (scalar_dim, vector_dim). 
            Edges encode edge type, relative distances and orientations between nodes. 
            Default: (132, 3).
        edge_h_dim (tuple): Hidden edge feature dimensions as (scalar_dim, vector_dim) 
            for GVP-GNN layers. Default: (64, 4).
        num_layers (int): Number of GVP-GNN layers in both encoder and decoder. 
            Default: 4.
        drop_rate (float): Dropout rate applied in all dropout layers for 
            regularization. Default: 0.5.
        out_dim (int): Output dimension corresponding to number of nucleotide types. 
            Default: 4 (A, C, G, U).
    """

    def __init__(
        self,
        node_in_dim=(15, 4),
        node_h_dim=(128, 16),
        edge_in_dim=(132, 3),
        edge_h_dim=(64, 4),
        num_layers=4,
        drop_rate=0.5,
        out_dim=4,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim
        self.edge_h_dim = edge_h_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        activations = (F.silu, None)

        # Node input embedding
        self.W_v = torch.nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim, activations=(None, None), vector_gate=True),
        )

        # Edge input embedding
        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim, activations=(None, None), vector_gate=True),
        )

        # Encoder layers (supports multiple conformations)
        self.encoder_layers = nn.ModuleList(
            MultiGVPConvLayer(
                self.node_h_dim,
                self.edge_h_dim,
                activations=activations,
                vector_gate=True,
                drop_rate=drop_rate,
                norm_first=True,
            )
            for _ in range(num_layers)
        )

        # Decoder layers
        self.W_s = nn.Embedding(self.out_dim + 1, self.out_dim)
        self.edge_h_dim = (self.edge_h_dim[0] + self.out_dim, self.edge_h_dim[1])
        self.decoder_layers = nn.ModuleList(
            GVPConvLayer(
                self.node_h_dim,
                self.edge_h_dim,
                activations=activations,
                vector_gate=True,
                drop_rate=drop_rate,
                autoregressive=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        )

        # Output
        self.W_out = GVP(self.node_h_dim, (self.out_dim, 0), activations=(None, None))

    def forward(self, batch):
        """Forward pass for sequence likelihood computation and training.

        Processes RNA structure graph(s) through encoder layers, pools multi-conformation
        features, and decodes autoregressively to produce logits over nucleotide
        vocabulary at each position. Requires ground truth sequence information.

        Args:
            batch (torch_geometric.data.Data or torch_geometric.data.Batch): 
                Input graph(s) containing:
                - node_s: node scalar features
                - node_v: node vector features  
                - edge_s: edge scalar features
                - edge_v: edge vector features
                - edge_index: graph connectivity
                - seq: ground truth sequence for autoregressive decoding
                - mask_confs: mask indicating valid conformations

        Returns:
            logits (torch.Tensor): Unnormalized log probabilities over nucleotide
                vocabulary of shape [n_nodes, 4]
        """
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        seq = batch.seq

        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)

        # Pool multi-conformation features:
        # nodes: (n_nodes, d_s), (n_nodes, d_v, 3)
        # edges: (n_edges, d_se), (n_edges, d_ve, 3)
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)

        encoder_embeddings = h_V

        h_S = self.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])

        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x=encoder_embeddings)

        logits = self.W_out(h_V)

        return logits

    @torch.no_grad()
    def sample(
        self,
        batch,
        n_samples,
        temperature: Optional[float] = 0.1,
        logit_bias: Optional[torch.Tensor] = None,
        return_logits: Optional[bool] = False,
    ):
        """Sample sequences autoregressively from the learned distribution.

        Generates multiple sequence designs for a given RNA structure by sampling
        from the categorical distribution at each position. The model encodes the
        structure, pools multi-conformation features, then autoregressively decodes
        nucleotides one position at a time. No gradients are computed during sampling.

        Args:
            batch (torch_geometric.data.Data): Input graph containing a single RNA
                backbone structure to design sequences for, with:
                - node_s: node scalar features
                - node_v: node vector features
                - edge_s: edge scalar features
                - edge_v: edge vector features
                - edge_index: graph connectivity
                - mask_confs: mask indicating valid conformations
            n_samples (int): Number of sequences to sample for the given structure
            temperature (float): Temperature parameter for softmax sampling. Lower
                values (e.g., 0.1) produce more deterministic samples, higher values
                (e.g., 1.0) produce more diverse samples. Default is 0.1.
            logit_bias (torch.Tensor, optional): Additive bias to apply to logits
                during sampling for manual control over specific positions. Shape
                should be [n_nodes, 4] to bias each position's nucleotide logits.
                Can be used to fix certain nucleotides or encourage specific bases.
            return_logits (bool): If True, return both sampled sequences and their
                corresponding logits. If False, return only sequences. Default is False.

        Returns:
            If return_logits is False:
                seq (torch.Tensor): Sampled sequences of shape [n_samples, n_nodes]
                    with integer values corresponding to nucleotide indices based on
                    the model's nucleotide-to-int mapping (typically A=0, C=1, G=2, U=3).
            
            If return_logits is True:
                tuple: A tuple containing:
                    - seq (torch.Tensor): Sampled sequences of shape [n_samples, n_nodes]
                    - logits (torch.Tensor): Unnormalized log probabilities of shape
                        [n_samples, n_nodes, 4] for each position and nucleotide
        """
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index

        device = edge_index.device
        num_nodes = h_V[0].shape[0]

        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)

        # Pool multi-conformation features
        # nodes: (n_nodes, d_s), (n_nodes, d_v, 3)
        # edges: (n_edges, d_se), (n_edges, d_ve, 3)
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)

        # Repeat features for sampling n_samples times
        h_V = (h_V[0].repeat(n_samples, 1), h_V[1].repeat(n_samples, 1, 1))
        h_E = (h_E[0].repeat(n_samples, 1), h_E[1].repeat(n_samples, 1, 1))

        # Expand edge index for autoregressive decoding
        edge_index = edge_index.expand(n_samples, -1, -1)
        offset = num_nodes * torch.arange(n_samples, device=device).view(-1, 1, 1)
        edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
        # This is akin to 'batching' (in PyG style) n_samples copies of the graph

        seq = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.int)
        h_S = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)
        logits = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)

        h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]

        # Decode one token at a time
        for i in range(num_nodes):
            h_S_ = h_S[edge_index[0]]
            h_S_[edge_index[0] >= edge_index[1]] = 0
            h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])

            edge_mask = edge_index[1] % num_nodes == i  # True for all edges where dst is node i
            edge_index_ = edge_index[:, edge_mask]  # subset all incoming edges to node i
            h_E_ = tuple_index(h_E_, edge_mask)
            node_mask = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.bool)
            node_mask[i::num_nodes] = True  # True for all nodes i and its repeats

            for j, layer in enumerate(self.decoder_layers):
                out = layer(
                    h_V_cache[j],
                    edge_index_,
                    h_E_,
                    autoregressive_x=h_V_cache[0],
                    node_mask=node_mask,
                )

                out = tuple_index(out, node_mask)  # subset out to only node i and its repeats

                if j < len(self.decoder_layers) - 1:
                    h_V_cache[j + 1][0][i::num_nodes] = out[0]
                    h_V_cache[j + 1][1][i::num_nodes] = out[1]

            lgts = self.W_out(out)
            # Add logit bias if provided to fix or bias positions
            if logit_bias is not None:
                lgts += logit_bias[i]
            # Sample from logits
            seq[i::num_nodes] = Categorical(logits=lgts / temperature).sample()
            h_S[i::num_nodes] = self.W_s(seq[i::num_nodes])
            logits[i::num_nodes] = lgts

        if return_logits:
            return seq.view(n_samples, num_nodes), logits.view(n_samples, num_nodes, self.out_dim)
        else:
            return seq.view(n_samples, num_nodes)

    def pool_multi_conf(self, h_V, h_E, mask_confs, edge_index):
        """Pool multi-conformation features using masked mean pooling.

        Aggregates features across multiple structural conformations by computing
        a masked average. Handles both scalar and vector features in GVP format.
        If only one conformation is present, returns features directly without pooling.

        Args:
            h_V (tuple): Node features as (scalar_features, vector_features) where:
                - scalar_features: shape (n_nodes, n_conf, d_s)
                - vector_features: shape (n_nodes, n_conf, d_v, 3)
            h_E (tuple): Edge features as (scalar_features, vector_features) where:
                - scalar_features: shape (n_edges, n_conf, d_se)
                - vector_features: shape (n_edges, n_conf, d_ve, 3)
            mask_confs (torch.Tensor): Boolean mask indicating valid conformations
                of shape (n_nodes, n_conf), where True indicates a valid conformation
            edge_index (torch.Tensor): Graph connectivity of shape (2, n_edges)

        Returns:
            tuple: A tuple containing:
                - h_V (tuple): Pooled node features as (scalar, vector) where:
                    - scalar: shape (n_nodes, d_s)
                    - vector: shape (n_nodes, d_v, 3)
                - h_E (tuple): Pooled edge features as (scalar, vector) where:
                    - scalar: shape (n_edges, d_se)
                    - vector: shape (n_edges, d_ve, 3)
        """
        if mask_confs.size(1) == 1:
            # Number of conformations is 1, no need to pool
            return (h_V[0][:, 0], h_V[1][:, 0]), (h_E[0][:, 0], h_E[1][:, 0])

        # True num_conf for masked mean pooling
        n_conf_true = mask_confs.sum(1, keepdim=True)  # (n_nodes, 1)

        # Mask scalar features
        mask = mask_confs.unsqueeze(2)  # (n_nodes, n_conf, 1)
        h_V0 = h_V[0] * mask
        h_E0 = h_E[0] * mask[edge_index[0]]

        # Mask vector features
        mask = mask.unsqueeze(3)  # (n_nodes, n_conf, 1, 1)
        h_V1 = h_V[1] * mask
        h_E1 = h_E[1] * mask[edge_index[0]]

        # Average pooling multi-conformation features
        h_V = (
            h_V0.sum(dim=1) / n_conf_true,  # (n_nodes, d_s)
            h_V1.sum(dim=1) / n_conf_true.unsqueeze(2),
        )  # (n_nodes, d_v, 3)
        h_E = (
            h_E0.sum(dim=1) / n_conf_true[edge_index[0]],  # (n_edges, d_se)
            h_E1.sum(dim=1) / n_conf_true[edge_index[0]].unsqueeze(2),
        )  # (n_edges, d_ve, 3)

        return h_V, h_E
