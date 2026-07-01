"""Dataset and batching utilities for gRNAde RNA inverse folding.

This module provides PyTorch dataset classes and custom batch samplers for training
and evaluating structure-conditioned RNA sequence design models. The core components
handle multi-state RNA structures with 3D backbone coordinates, secondary structure
constraints, and flexible graph featurization.

Key Features:
    - Multi-conformer RNA representation with up to N conformational states per sequence
    - 3-bead coarse-grained backbone (P, C4', N1/N9 atoms)
    - 3D geometric graph construction with scalar and vector features
    - Graph edges encoding backbone connectivity (1D), base pairing (2D), and spatial
      proximity (3D)
    - Training augmentation via coordinate noise and optional 3D dropout
    - Custom batch sampling to optimize GPU memory usage by grouping sequences of
      similar lengths
"""

import os
import random

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

from src.constants import DISTANCE_EPS, RNA_ATOMS, RNA_NUCLEOTIDES
from src.data.data_utils import *
from src.data.featurizer import *


##################################
# Dataset Classes
##################################

class RNADesignDataset(data.Dataset):
    """Multi-state RNA inverse folding dataset with geometric graph representation.

    This dataset builds 3D geometric graphs from RNA structures for training and 
    evaluating structure-conditioned sequence design models. Each RNA is
    represented as a coarse-grained 3-bead backbone model (P, C4', N1/N9 atoms) and
    converted into a geometric graph with:
        - Nodes: RNA nucleotides with scalar (orientations, dihedrals) and vector (positions)
          features
        - Edges: Three types encoding 1D backbone connectivity, 2D base pairing from
          secondary structure, and 3D spatial proximity

    The dataset supports multi-conformer structures, allowing models to learn from multiple
    experimental conformations of the same RNA sequence. During training, coordinate noise
    and 3D information dropout are applied for regularization.

    Graph Attributes:
        Each returned graph is of type `torch_geometric.data.Data` with:
        - seq: Nucleotide sequence as integer tensor, shape [num_nodes]
        - node_s: Node scalar features (orientations, angles), shape [num_nodes, num_conf, num_bb_atoms x 5]
        - node_v: Node vector features (positions, orientations), shape [num_nodes, num_conf, 2 + (num_bb_atoms - 1), 3]
        - edge_s: Edge scalar features (distances, angles, RBF, pos encodings), shape [num_edges, num_conf, features]
        - edge_v: Edge vector features (directions), shape [num_edges, num_conf, num_bb_atoms, 3]
        - edge_index: Edge connectivity, shape [2, num_edges]
        - mask: Node validity mask, False for nodes with missing coordinates
        - mask_seq: Sequence mask for positions to be designed

    Args:
        data_list (list): List of RNA structure dictionaries, each containing:
            - 'sequence' (str): RNA nucleotide sequence
            - 'coords_list' (list): List of coordinate tensors for multiple conformers
            - 'sec_struct_list' (list): List of secondary structures in dot-bracket notation
        split (str): Dataset split name ('train', 'val', 'test'). Coordinate noise is
            applied only during training. Defaults to 'train'.
        radius (float): Radial cutoff distance for drawing edges in Angstroms. Currently
            not used in favor of top_k. Defaults to 4.5.
        top_k (int): Number of k-nearest spatial neighbors to connect for each node.
            Defaults to 32.
        num_rbf (int): Number of radial basis functions for encoding distances. Defaults to 32.
        num_posenc (int): Number of sinusoidal positional encodings per edge. Defaults to 32.
        max_num_conformers (int): Maximum number of conformers to sample per RNA sequence.
            Defaults to 1.
        noise_scale (float): Standard deviation of Gaussian noise added to coordinates
            during training for data augmentation. Defaults to 0.1.
        drop_prob_3d (float): Probability of dropping 3D structure information during
            training, forcing the model to rely on 1D/2D information. Defaults to 0.75.
        random_order (bool): Whether to randomly permute node order for each sample.
            Defaults to True.
        pyrimidine_bb_indices (list): Indices of backbone atoms for pyrimidines (C, U)
            in the atom list. Defaults to [P, C4', N1].
        purine_bb_indices (list): Indices of backbone atoms for purines (A, G) in the
            atom list. Defaults to [P, C4', N9].
        distance_eps (float): Small epsilon value to prevent division by zero in distance
            calculations. Defaults to DISTANCE_EPS from constants.
        device (str): Device for tensor operations ('cpu', 'cuda'). Defaults to 'cpu'.

    Workflow:
        1. Initialize featurizer with geometric graph construction parameters
        2. Pre-process input data:
            a. Extract backbone coordinates for each conformer
            b. Filter out structures with all-missing coordinates
            c. Store valid conformers for each RNA
        3. During __getitem__, apply optional random permutation and featurize on-the-fly

    Note:
        Structures with missing coordinates for ALL residues are filtered out during
        preprocessing, but structures with partial missing data are retained and masked.
    """

    def __init__(
        self,
        data_list=[],
        split="train",
        radius=4.5,
        top_k=32,
        num_rbf=32,
        num_posenc=32,
        max_num_conformers=1,
        noise_scale=0.1,
        drop_prob_3d=0.75,
        random_order=True,
        pyrimidine_bb_indices=[
            RNA_ATOMS.index("P"),
            RNA_ATOMS.index("C4'"),
            RNA_ATOMS.index("N1"),
        ],
        purine_bb_indices=[RNA_ATOMS.index("P"), RNA_ATOMS.index("C4'"), RNA_ATOMS.index("N9")],
        distance_eps=DISTANCE_EPS,
        device="cpu",
    ):
        super().__init__()

        self.pyrimidine_bb_indices = pyrimidine_bb_indices
        self.purine_bb_indices = purine_bb_indices
        self.random_order = random_order
        self.featurizer = RNAGraphFeaturizer(
            split=split,
            radius=radius,
            top_k=top_k,
            num_rbf=num_rbf,
            num_posenc=num_posenc,
            max_num_conformers=max_num_conformers,
            noise_scale=noise_scale,
            drop_prob_3d=drop_prob_3d,
            distance_eps=distance_eps,
            device=device,
        )

        # Pre-process raw data to prepare self.data_list
        print(f"\n{split.upper()} DATASET")
        print(f"    Pre-processing {len(data_list)} samples")
        self.data_list = []
        for rna in tqdm(data_list):
            coords_list = []
            for coords, sec_struct in zip(
                rna["coords_list"],
                rna["sec_struct_list"],
            ):
                # Only keep backbone atom coordinates: num_res x num_atoms x 3
                coords = get_backbone_coords(
                    coords,
                    rna["sequence"],
                    self.pyrimidine_bb_indices,
                    self.purine_bb_indices,
                )
                # Do not add structures with missing coordinates for ALL residues
                if not torch.all((coords == FILL_VALUE).sum(axis=(1, 2)) > 0):
                    coords_list.append(coords)

            if len(coords_list) > 0:
                # Add processed coords_list to self.data_list
                rna["coords_list"] = coords_list
                self.data_list.append(rna)

        print(f"    Finished: {len(self.data_list)} pre-processed samples")

        # Compute number of nodes per sample (used for batching)
        self.node_counts = [len(entry["sequence"]) for entry in self.data_list]

    def __len__(self):
        """Return the number of RNA structures in the dataset.

        Returns:
            int: Number of preprocessed RNA samples available for training/evaluation
        """
        return len(self.data_list)

    def __getitem__(self, i):
        """Retrieve and featurize a single RNA structure as a geometric graph.

        This method fetches the i-th RNA structure from the dataset, optionally applies
        random node permutation for data augmentation, and constructs a geometric graph
        with node and edge features.

        Args:
            i (int): Index of the RNA structure to retrieve

        Returns:
            torch_geometric.data.Data: Geometric graph representing the RNA structure
                with attributes including seq, node_s, node_v, edge_s, edge_v,
                edge_index, mask, and mask_seq

        Note:
            Random permutation of nucleotide order is applied when self.random_order
            is True, which helps the model learn permutation-invariant representations.
        """
        permutation = None
        if self.random_order:
            permutation = torch.randperm(self.node_counts[i])
        return self.featurizer(
            self.data_list[i],
            permutation=permutation,
        )


##################################
# Batch Sampling Utilities
##################################

class BatchSampler(data.Sampler):
    """Custom batch sampler for efficient GPU memory utilization with variable-length RNAs.

    This sampler groups RNA structures into batches such that the total number of nucleotides
    (nodes) in each batch does not exceed a specified maximum. This is critical for graph
    neural networks where memory consumption scales with the total number of nodes rather
    than the number of graphs in a batch.

    The sampler handles structures of varying lengths by:
        - Grouping small/medium structures together up to max_nodes_batch total nodes
        - Processing large structures (> max_nodes_batch) individually if they fit within
          max_nodes_sample
        - Filtering out structures exceeding max_nodes_sample

    This approach maximizes GPU utilization while preventing out-of-memory errors and
    ensures efficient training on datasets with heterogeneous sequence lengths.

    Credit: Adapted from https://github.com/jingraham/neurips19-graph-protein-design

    Args:
        node_counts (array-like): Array of nucleotide counts for each RNA structure in
            the dataset. Used to determine batch composition.
        max_nodes_batch (int): Maximum total number of nucleotides allowed in a batch
            when combining multiple structures. Defaults to 3000.
        max_nodes_sample (int): Maximum number of nucleotides for a single structure.
            Structures with more nucleotides than this are excluded from training.
            Defaults to 5000.
        shuffle (bool): Whether to shuffle the order of batches. Set to True for training
            and False for evaluation. Defaults to True.
    """

    def __init__(self, node_counts, max_nodes_batch=3000, max_nodes_sample=5000, shuffle=True):
        self.node_counts = node_counts
        self.shuffle = shuffle
        self.max_nodes_batch = max_nodes_batch
        self.max_nodes_sample = max_nodes_sample

        # indices of samples with node count <= max_nodes_batch
        max_nodes = min(max_nodes_batch, max_nodes_sample)
        self.idx = [i for i in range(len(node_counts)) if node_counts[i] <= max_nodes]
        # [indices] of samples with max_nodes_sample >= node count > max_nodes_batch
        # appended to list of batches at the end
        self.batches_single = [
            [i]
            for i in range(len(node_counts))
            if max_nodes_sample >= node_counts[i] > max_nodes_batch
        ]

        self._form_batches()
        if len(self.batches_single) > 0:
            # append single samples to batches list
            self.batches += self.batches_single
            if self.shuffle:
                random.shuffle(self.batches)

    def _form_batches(self):
        """Construct batches by greedily grouping samples to maximize node count per batch.

        This method uses a greedy algorithm to pack RNA structures into batches:
            1. Optionally shuffle the dataset indices
            2. Iterate through structures in order
            3. For each structure, add it to the current batch if it fits within max_nodes_batch
            4. When adding the next structure would exceed the limit, start a new batch
            5. Continue until all structures are assigned to batches

        The resulting batches are stored in self.batches as a list of lists, where each
        inner list contains dataset indices for structures in that batch.
        """
        self.batches = []
        if self.shuffle:
            random.shuffle(self.idx)

        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes_batch:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)

    def __len__(self):
        """Return the total number of batches.

        Returns:
            int: Number of batches that will be yielded during iteration
        """
        if not self.batches:
            self._form_batches()
        return len(self.batches)

    def __iter__(self):
        """Iterate over batches, yielding lists of dataset indices.

        Yields:
            list: List of integer dataset indices for each batch. Each yielded list
                contains indices of RNA structures that should be collated together.
        """
        if not self.batches:
            self._form_batches()
        yield from self.batches
