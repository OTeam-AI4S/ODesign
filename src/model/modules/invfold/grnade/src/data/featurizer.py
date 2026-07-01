import math
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, dense_to_sparse, to_undirected

from src.constants import DISTANCE_EPS, RNA_ATOMS, RNA_NUCLEOTIDES
from src.data.data_utils import *
from src.data.sec_struct_utils import dotbracket_to_adjacency


##################################
# Main Featurizer Class
##################################

class RNAGraphFeaturizer:
    """Geometric graph featurizer for RNA backbone structures.

    This class converts RNA 3D structures into geometric graphs suitable for training
    and inference with structure-conditioned sequence design models. The featurization
    builds on a 3-bead coarse-grained backbone representation (P, C4', N1/N9 atoms) and
    constructs graphs with scalar and vector features.

    The featurizer supports multi-conformer inputs, allowing the model to learn from
    multiple experimental structures of the same RNA. Graph edges encode three types of
    structural information:
        1. 1D: Backbone connectivity (sequential neighbors along the RNA chain)
        2. 2D: Base pairing from secondary structure (Watson-Crick and non-canonical)
        3. 3D: Spatial proximity (k-nearest neighbors in 3D space)

    During training, augmentations are applied including Gaussian coordinate noise and
    stochastic 3D information dropout, forcing the model to learn robust representations.
    The featurizer automatically handles missing coordinates by masking affected
    nodes and edges, allowing partial structures to be processed.

    Graph Structure:
        Returned graph is of type `torch_geometric.data.Data` with attributes:
        - seq: Nucleotide sequence as integers, shape [num_nodes]
        - node_s: Scalar node features (internal coords), shape [num_nodes, num_conf, num_bb_atoms x 5]
        - node_v: Vector node features (orientations), shape [num_nodes, num_conf, 2 + (num_bb_atoms - 1), 3]
        - edge_s: Scalar edge features (distances, RBF, pos enc), shape [num_edges, num_conf, features]
        - edge_v: Vector edge features (directions), shape [num_edges, num_conf, num_bb_atoms, 3]
        - edge_index: Edge connectivity [2, num_edges]
        - mask: Node validity mask, False for missing coordinates
        - mask_seq: Sequence design mask indicating positions to design

    Args:
        split (str): Dataset split ('train', 'val', 'test'). Coordinate noise is applied
            only during training. Defaults to 'train'.
        radius (float): Radial cutoff distance for drawing edges in Angstroms. Currently
            not used in favor of top_k. Defaults to 4.5.
        top_k (int): Number of k-nearest spatial neighbors to connect for each node.
            Defaults to 32.
        num_rbf (int): Number of radial basis functions for encoding pairwise distances.
            Defaults to 32.
        num_posenc (int): Number of sinusoidal positional encoding dimensions per edge.
            Defaults to 32.
        max_num_conformers (int): Maximum number of conformers to sample per RNA sequence.
            If more conformers are available, a random subset is selected. Defaults to 1.
        noise_scale (float): Standard deviation of Gaussian noise added to coordinates
            during training for data augmentation. Defaults to 0.1.
        drop_prob_3d (float): Probability of dropping 3D structure information during
            training, forcing the model to rely on 1D/2D information only. Defaults to 0.75.
        distance_eps (float): Small epsilon value added to squared distances before
            taking square root, ensuring smooth gradients. Defaults to DISTANCE_EPS.
        device (str): Device for tensor operations ('cpu', 'cuda'). Defaults to 'cpu'.

    Workflow:
        1. Sample up to max_num_conformers conformations
        2. Extract 3-bead backbone coordinates
        3. Optionally apply coordinate noise (training only)
        4. Compute internal coordinates (bonds, angles, dihedrals)
        5. Construct multi-edge graph (1D, 2D, 3D connectivity)
        6. Compute node and edge features (scalar and vector)
        7. Optionally drop 3D information (training only)
        8. Return PyTorch Geometric Data object
    """

    def __init__(
        self,
        split="train",
        radius=4.5,
        top_k=32,
        num_rbf=32,
        num_posenc=32,
        max_num_conformers=1,
        noise_scale=0.1,
        drop_prob_3d=0.75,
        distance_eps=DISTANCE_EPS,
        device="cpu",
    ):
        super().__init__()

        self.split = split
        self.radius = radius
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_posenc = num_posenc
        self.max_num_conformers = max_num_conformers
        self.noise_scale = noise_scale
        self.drop_prob_3d = drop_prob_3d
        self.distance_eps = distance_eps
        self.device = device

        # nucleotide mapping: {'A': 0, 'G': 1, 'C': 2, 'U': 3, '_': 4}
        self.letter_to_num = dict(zip(RNA_NUCLEOTIDES, list(range(len(RNA_NUCLEOTIDES)))))
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}
        self.letter_to_num["_"] = len(self.letter_to_num)  # unknown nucleotide

    def __call__(self, rna, permutation=None):
        """Convert RNA structure dictionary to geometric graph.

        This method is the main entry point for featurization, taking a raw RNA structure
        dictionary and converting it into a PyTorch Geometric graph with scalar and vector
        node and edge features.

        Args:
            rna (dict): RNA data dictionary containing:
                - 'sequence' (str): RNA nucleotide sequence
                - 'coords_list' (list): List of coordinate tensors [num_conf, num_res, num_atoms, 3]
                - 'sec_struct_list' (list): List of secondary structures in dot-bracket notation
            permutation (torch.Tensor, optional): Permutation indices for randomizing node order.
                If provided, should be a tensor of shape [num_residues] with values 0 to num_residues-1.
                Used for data augmentation during training. Defaults to None.

        Returns:
            torch_geometric.data.Data: Geometric graph with attributes:
                - seq: Integer sequence [num_nodes]
                - node_s, node_v: Scalar and vector node features
                - edge_s, edge_v: Scalar and vector edge features
                - edge_index: Edge connectivity [2, num_edges]
                - mask: Node validity mask
                - mask_seq: Sequence design mask
        """
        with torch.no_grad():
            # Target sequence: num_res x 1
            seq = torch.as_tensor(
                [self.letter_to_num[residue] for residue in rna["sequence"]],
                device=self.device,
                dtype=torch.long,
            )

            # Mask non-standard nucleotides: num_res x 1
            mask_seq = seq != self.letter_to_num["_"]

            # Set of coordinates: num_conf x num_res x num_bb_atoms x 3
            coords_list, sec_struct_list = get_k_random_entries(rna, k=self.max_num_conformers)
            # assert coords_list.shape[2] == 3, "Keep only 3 backbone atoms"

            # Coordinate mask: num_conf x num_res x num_bb_atoms
            mask_atoms = (coords_list == FILL_VALUE).sum(-1) == 0
            mask_residues = mask_atoms.sum(-1) != 0
            mask_confs = mask_residues.sum(-1) != 0

            # Convert to torch tensors
            coords_list = torch.as_tensor(coords_list, device=self.device, dtype=torch.float32)
            mask_atoms = torch.BoolTensor(mask_atoms)
            mask_residues = torch.BoolTensor(mask_residues) & mask_seq
            mask_confs = torch.BoolTensor(mask_confs).repeat(len(seq), 1)

            # Data augmentation during training
            if "train" in self.split:
                # Add noise to prevent overfitting on crystalisation artifacts
                noise = torch.randn_like(coords_list, device=self.device) * self.noise_scale
                coords_list += noise * mask_atoms.unsqueeze(-1).float()

                if random.random() < self.drop_prob_3d:
                    # Randomly drop all 3D coordinates (augmentation)
                    coords_list = torch.ones_like(coords_list) * FILL_VALUE
                    mask_atoms = torch.zeros_like(mask_atoms)
                    mask_residues = torch.zeros_like(mask_residues)

            if "2d" in self.split or self.drop_prob_3d == 1.0:
                # Enforce 2D-only; drop all 3D coordinates
                coords_list = torch.ones_like(coords_list) * FILL_VALUE
                mask_atoms = torch.zeros_like(mask_atoms)
                mask_residues = torch.zeros_like(mask_residues)

            # Node internal coordinates (scalars) and normalised vectors
            dihedrals, angles, lengths = internal_coords(coords_list, mask_residues)
            angle_stack = torch.cat([dihedrals, angles], dim=-1)
            lengths = torch.log(lengths + self.distance_eps)
            internal_coords_feat = (
                torch.cat([torch.cos(angle_stack), torch.sin(angle_stack), lengths], dim=-1)
                * mask_residues.unsqueeze(-1).float()
            )
            internal_vecs_feat = internal_vecs(coords_list, mask_atoms)

            if permutation is not None:
                assert len(permutation) == len(seq)
                # Map permutation index to original index
                # 0th entry in inverse_permutation ==
                # index of the 0th entry in the permuted sequence
                inverse_permutation = torch.argsort(permutation)
                # Apply the given permutation to nodes
                seq = seq[permutation]
                mask_seq = mask_seq[permutation]
                coords_list = coords_list[:, permutation]
                mask_atoms = mask_atoms[:, permutation]
                mask_residues = mask_residues[:, permutation]
                mask_confs = mask_confs[permutation]
                internal_coords_feat = internal_coords_feat[:, permutation]
                internal_vecs_feat = internal_vecs_feat[:, permutation]

            # Construct merged edge index
            edge_index = []
            edge_type = []

            # 3D K-nearest neighbour graph
            for _coords, _mask_atoms, _mask_residues in zip(
                coords_list, mask_atoms, mask_residues
            ):
                # K-nearest neighbour graph using centroids of each neucleotide
                _centroids = _coords.sum(1) / (
                    _mask_atoms.sum(1).unsqueeze(-1) + self.distance_eps
                )
                _edge_index = torch_cluster.knn_graph(_centroids, self.top_k)
                # Remove edges to/from masked residues (missing 3D atomic coordinates)
                _masked = torch.nonzero(~_mask_residues, as_tuple=False).squeeze(1)
                _edge_0_mask = ~_edge_index[0].unsqueeze(1).eq(_masked).any(dim=1)
                _edge_1_mask = ~_edge_index[1].unsqueeze(1).eq(_masked).any(dim=1)
                _edge_index = _edge_index[:, _edge_0_mask & _edge_1_mask]
                edge_index.append(_edge_index)
                edge_type.append(torch.ones(_edge_index.shape[1], dtype=torch.long) * 2)

            # 2D secondary structure graph
            for sec_struct in sec_struct_list:
                _sec_struct = torch.tensor(
                    dotbracket_to_adjacency(sec_struct, keep_pseudoknots=True),
                    dtype=torch.long,
                )
                if permutation is not None:
                    _sec_struct = _sec_struct[permutation][:, permutation]
                _edge_index = dense_to_sparse(_sec_struct)[0]
                edge_index.append(_edge_index)
                edge_type.append(torch.ones(_edge_index.shape[1], dtype=torch.long) * 3)

            # Sequence proximity graph
            _edge_index = []
            _edge_type = []
            for i in range(len(seq)):
                for j in range(i + 1, i + 1 + self.top_k):
                    if permutation is not None:
                        if (
                            j < len(seq)
                            and mask_seq[inverse_permutation[i]]
                            and mask_seq[inverse_permutation[j]]
                        ):
                            _i, _j = inverse_permutation[i], inverse_permutation[j]
                            _edge_index.append([_i, _j])
                            _edge_type.append(1)
                            _edge_index.append([_j, _i])
                            _edge_type.append(1)
                    else:
                        if j < len(seq) and mask_seq[i] and mask_seq[j]:
                            _edge_index.append([i, j])
                            _edge_type.append(1)
                            _edge_index.append([j, i])
                            _edge_type.append(1)
            edge_index.append(torch.tensor(_edge_index, dtype=torch.long).t())
            edge_type.append(torch.tensor(_edge_type, dtype=torch.long))

            # Merge edge index
            edge_index, edge_type = coalesce(
                torch.concat(edge_index, dim=1),
                torch.concat(edge_type, dim=0),
                num_nodes=len(seq),
                reduce="max",  # max edge type if multiple edges present
            )
            edge_index, edge_type = to_undirected(
                edge_index,
                edge_type,
                num_nodes=len(seq),
                reduce="max",  # max edge type if multiple edges present
            )

            # Remove edges to/from masked residues
            masked_idx = torch.nonzero(~mask_seq, as_tuple=False).squeeze(1)
            edge_0_mask = ~edge_index[0].unsqueeze(1).eq(masked_idx).any(dim=1)
            edge_1_mask = ~edge_index[1].unsqueeze(1).eq(masked_idx).any(dim=1)
            edge_index = edge_index[:, edge_0_mask & edge_1_mask]
            edge_type = edge_type[edge_0_mask & edge_1_mask]

            # Reshape: num_res x num_conf x ...
            coords_list = coords_list.permute(1, 0, 2, 3)
            internal_coords_feat = internal_coords_feat.permute(1, 0, 2)
            internal_vecs_feat = internal_vecs_feat.permute(1, 0, 2, 3)

            # Edge displacement vectors: num_edges x num_conf x num_res x 3
            edge_vectors = coords_list[edge_index[0]] - coords_list[edge_index[1]]
            edge_mask = (
                mask_atoms.permute(1, 0, 2)[edge_index[0]]
                & mask_atoms.permute(1, 0, 2)[edge_index[1]]
            )
            edge_vectors = edge_vectors * edge_mask.unsqueeze(-1).repeat(1, 1, 1, 3).float()

            # Edge lengths: num_edges x num_conf x num_res x 1
            edge_lengths = (
                torch.sqrt((edge_vectors**2).sum(dim=-1) + self.distance_eps) * edge_mask.float()
            )

            # Edge RBF features: num_edges x num_conf x num_rbf
            edge_rbf = rbf_expansion(edge_lengths, num_rbf=self.num_rbf)

            # Edge positional encodings: num_edges x num_conf x num_posenc
            if permutation is not None:
                # perm_mapping = torch.as_tensor(permutation, device=self.device)
                idx_diff = permutation[edge_index[0]] - permutation[edge_index[1]]
            else:
                idx_diff = edge_index[0] - edge_index[1]
            edge_posenc = (
                positional_encoding(idx_diff[..., None], self.num_posenc)
                .unsqueeze_(1)
                .repeat(1, self.max_num_conformers, 1)
            )

            # Edge types as one-hot: num_edges x num_conf x num_edge_types
            edge_type = (
                F.one_hot(edge_type, num_classes=4)
                .float()
                .unsqueeze_(1)
                .repeat(1, self.max_num_conformers, 1)
            )

            node_s = internal_coords_feat
            node_v = internal_vecs_feat
            edge_s = torch.cat(
                [
                    edge_type,
                    edge_rbf,
                    edge_posenc,
                ],
                dim=-1,
            )
            edge_v = normed_vec(edge_vectors)
            node_s, node_v, edge_s, edge_v = map(
                torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
            )

            data = Data(
                seq=seq,  # num_res x 1
                node_s=node_s,  # num_res x num_conf x (num_bb_atoms x 5)
                node_v=node_v,  # num_res x num_conf x (2 + (num_bb_atoms - 1)) x 3
                edge_s=edge_s,  # num_edges x num_conf x (num_bb_atoms x num_edge_types + num_rbf + num_posenc)
                edge_v=edge_v,  # num_edges x num_conf x num_bb_atoms x 3
                edge_index=edge_index,  # 2 x num_edges
                mask_seq=mask_seq,  # num_res
                mask_confs=mask_confs,  # num_res x num_conf
            )

        return data

    def featurize(self, rna, permutation=None):
        """Featurize RNA backbone from dictionary of tensors.

        Convenience method that calls __call__ internally. Provided for API consistency
        and explicit naming.

        Args:
            rna (dict): Raw RNA data dictionary with keys:
                - sequence (str): RNA sequence of length num_residues
                - coords_list (list): List of backbone coordinate tensors, each with shape
                  [num_residues, num_bb_atoms, 3]
                - sec_struct_list (list): List of secondary structure strings in dot-bracket
                  notation, one per conformer
            permutation (torch.Tensor, optional): Permutation indices for randomizing node
                order. Defaults to None.

        Returns:
            torch_geometric.data.Data: Geometric graph representing the RNA structure
        """
        return self(rna, permutation)

    def featurize_from_pdb_file(self, pdb_filepath):
        """Featurize RNA backbone structure directly from a PDB file.

        This method provides a convenient interface for featurizing single RNA structures
        stored in PDB format. It automatically extracts the sequence, coordinates, and
        secondary structure from the file.

        Args:
            pdb_filepath (str): Absolute or relative path to PDB file containing RNA structure

        Returns:
            torch_geometric.data.Data: Geometric graph representing the RNA structure

        Note:
            This method assumes the PDB file contains a single RNA structure. For
            multi-conformer structures, use featurize_from_pdb_filelist.
        """
        sequence, coords, sec_struct, _ = pdb_to_tensor(
            pdb_filepath, return_sec_struct=True, return_sasa=False
        )
        coords = get_backbone_coords(coords, sequence)
        rna = {
            "sequence": sequence,
            "coords_list": [coords],
            "sec_struct_list": [sec_struct],
        }
        return self(rna), rna

    def featurize_from_pdb_filelist(self, pdb_filelist):
        """Featurize multi-conformer RNA structure from multiple PDB files.

        This method handles multiple experimental structures (conformers) of the same RNA
        molecule, which is important for capturing conformational diversity and dynamic
        behavior. All PDB files should contain the same RNA sequence in different conformations.

        Args:
            pdb_filelist (list): List of PDB file paths, each containing a different conformer
                of the same RNA sequence. All structures must have the same sequence length
                and nucleotide identity.

        Returns:
            torch_geometric.data.Data: Geometric graph with multi-conformer features where
                node and edge features have an additional conformer dimension

        Raises:
            AssertionError: If sequences differ across PDB files

        Note:
            The number of conformers will be capped at self.max_num_conformers. If more
            PDB files are provided, a random subset will be selected.
        """
        # read first pdb file
        sequence, coords, sec_struct, _ = pdb_to_tensor(
            pdb_filelist[0], return_sec_struct=True, return_sasa=False, keep_pseudoknots=True
        )
        coords = get_backbone_coords(coords, sequence)
        rna = {
            "sequence": sequence,
            "coords_list": [coords],
            "sec_struct_list": [sec_struct],
        }

        # read remaining pdb files
        for pdb_filepath in pdb_filelist[1:]:
            sequence, coords, sec_struct, _ = pdb_to_tensor(
                pdb_filepath, return_sec_struct=True, return_sasa=False, keep_pseudoknots=True
            )
            coords = get_backbone_coords(coords, sequence)
            assert sequence == rna["sequence"], "All PDBs must have the same sequence"
            rna["coords_list"].append(coords)
            rna["sec_struct_list"].append(sec_struct)

        return self(rna), rna


##################################
# Conformer Sampling Utilities
##################################

def get_k_random_entries(rna, k):
    """Sample k random conformers from RNA structure with multiple conformations.

    This function randomly samples a subset of conformers from an RNA structure that has
    multiple experimental conformations. If k exceeds the available number of conformers,
    random conformers are sampled with replacement to reach k samples.

    Args:
        rna (dict): RNA data dictionary containing:
            - 'coords_list': List of coordinate tensors, one per conformer
            - 'sec_struct_list': List of secondary structure strings, one per conformer
        k (int): Number of conformers to sample. Can exceed the available number.

    Returns:
        dict: Updated RNA dictionary with sampled conformers:
            - 'coords_list': List of k sampled coordinate tensors
            - 'sec_struct_list': List of k sampled secondary structure strings

    Note:
        When k > n (available conformers), sampling is done with replacement, allowing
        the same conformer to appear multiple times in the output.
    """
    n = len(rna["coords_list"])
    coords_list = np.array(rna["coords_list"])
    sec_struct_list = np.array(rna["sec_struct_list"])
    if k > n:
        # If k is greater than the length of the list,
        # return all the entries in the list and pad random entries up to k
        rand_idx = np.random.choice(n, size=k - n, replace=True)
        coords_subset = np.concatenate((coords_list, coords_list[rand_idx]), axis=0)
        sec_struct_subset = np.concatenate((sec_struct_list, sec_struct_list[rand_idx]), axis=0)
    else:
        # If k is less than or equal to the length of the list,
        # randomly select k entries
        rand_idx = np.random.choice(n, size=k, replace=False)
        coords_subset = coords_list[rand_idx]
        sec_struct_subset = sec_struct_list[rand_idx]
    return coords_subset, sec_struct_subset


##################################
# Internal Coordinate Computation
##################################

def internal_coords(
    X: torch.Tensor,
    C: Optional[torch.Tensor] = None,
    return_masks: bool = False,
    distance_eps: float = DISTANCE_EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute internal coordinates (dihedrals, angles, lengths) from RNA backbone.

    This function computes internal coordinates from RNA backbone atom positions, providing
    a geometrically meaningful and SE(3)-invariant representation. Internal coordinates
    capture local geometry while being invariant to global translations and rotations.

    The computation uses smooth distance approximations (sqrt(sum_sq + eps)) to ensure
    differentiability everywhere, avoiding gradient discontinuities at zero distances.
    This is critical for stable gradient-based learning.

    Computes internal coordinates for:
        - P-C4' backbone connectivity (bond lengths, angles, dihedrals)
        - C4'-N base attachment (bond lengths, angles, dihedrals)

    Args:
        X (torch.Tensor): Backbone coordinates with shape
            [num_batch, num_residues, num_atom_types, 3] where num_atom_types
            is typically 3 (P, C4', N1/N9)
        C (torch.Tensor, optional): Chain map tensor with shape [num_batch, num_residues]
            indicating chain membership. Used to mask inter-chain bonds. If None, assumes
            single chain. Defaults to None.
        return_masks (bool): If True, also return masks indicating valid internal coordinates.
            Defaults to False.
        distance_eps (float): Small value added to squared distances before taking square root
            to ensure smooth gradients. Defaults to DISTANCE_EPS.

    Returns:
        tuple: If return_masks=False, returns (dihedrals, angles, lengths):
            - dihedrals (torch.Tensor): Backbone dihedral angles [num_batch, num_residues, 4]
            - angles (torch.Tensor): Backbone bond angles [num_batch, num_residues, 4]
            - lengths (torch.Tensor): Backbone bond lengths [num_batch, num_residues, 4]
        
        If return_masks=True, returns (dihedrals, angles, lengths, mask_D, mask_A, mask_L)
            with additional masks indicating valid internal coordinates.

    Note:
        Adapted from Chroma protein design framework. In our case, num_batch corresponds
        to num_conformations, allowing efficient batched processing of multiple conformers.
    """
    mask = (C > 0).float()
    X_chain = X[:, :, :2, :]
    num_batch, num_residues, _, _ = X_chain.shape
    X_chain = X_chain.reshape(num_batch, 2 * num_residues, 3)

    # This function historically returns the angle complement
    _lengths = lambda Xi, Xj: lengths(Xi, Xj, distance_eps=distance_eps)
    _angles = lambda Xi, Xj, Xk: np.pi - angles(Xi, Xj, Xk, distance_eps=distance_eps)
    _dihedrals = lambda Xi, Xj, Xk, Xl: dihedrals(Xi, Xj, Xk, Xl, distance_eps=distance_eps)

    # Compute internal coordinates associated with -[P]-[C4']-
    PC4p_L = _lengths(X_chain[:, 1:, :], X_chain[:, :-1, :])
    PC4p_A = _angles(X_chain[:, :-2, :], X_chain[:, 1:-1, :], X_chain[:, 2:, :])
    PC4p_D = _dihedrals(
        X_chain[:, :-3, :],
        X_chain[:, 1:-2, :],
        X_chain[:, 2:-1, :],
        X_chain[:, 3:, :],
    )

    # Compute internal coordinates associated with [C4']-[N]
    X_P, X_C4p, X_N = X.unbind(dim=2)
    X_P_next = X[:, 1:, 0, :]
    N_L = _lengths(X_C4p, X_N)
    N_A = _angles(X_P, X_C4p, X_N)
    N_D = _dihedrals(X_P_next, X_N[:, :-1, :], X_C4p[:, :-1, :], X_P[:, :-1, :])

    if C is None:
        C = torch.zeros_like(mask)

    # Mask nonphysical bonds and angles
    # Note: this could probably also be expressed as a Conv, unclear
    # which is faster and this probably not rate-limiting.
    C = C * (mask.type(torch.long))
    ii = torch.stack(2 * [C], dim=-1).view([num_batch, -1])
    L0, L1 = ii[:, :-1], ii[:, 1:]
    A0, A1, A2 = ii[:, :-2], ii[:, 1:-1], ii[:, 2:]
    D0, D1, D2, D3 = ii[:, :-3], ii[:, 1:-2], ii[:, 2:-1], ii[:, 3:]

    # Mask for linear backbone
    mask_L = torch.eq(L0, L1)
    mask_A = torch.eq(A0, A1) * torch.eq(A0, A2)
    mask_D = torch.eq(D0, D1) * torch.eq(D0, D2) * torch.eq(D0, D3)
    mask_L = mask_L.type(torch.float32)
    mask_A = mask_A.type(torch.float32)
    mask_D = mask_D.type(torch.float32)

    # Masks for branched nitrogen
    mask_N_D = torch.eq(C[:, :-1], C[:, 1:])
    mask_N_D = mask_N_D.type(torch.float32)
    mask_N_A = mask
    mask_N_L = mask

    def _pad_pack(D, A, L, N_D, N_A, N_L):
        # Pad and pack together the components
        D = F.pad(D, (1, 2))
        A = F.pad(A, (1, 1))
        L = F.pad(L, (0, 1))
        N_D = F.pad(N_D, (0, 1))
        D, A, L = (x.reshape(num_batch, num_residues, 2) for x in [D, A, L])
        _pack = lambda a, b: torch.cat([a, b.unsqueeze(-1)], dim=-1)
        L = _pack(L, N_L)
        A = _pack(A, N_A)
        D = _pack(D, N_D)
        return D, A, L

    D, A, L = _pad_pack(PC4p_D, PC4p_A, PC4p_L, N_D, N_A, N_L)
    mask_D, mask_A, mask_L = _pad_pack(mask_D, mask_A, mask_L, mask_N_D, mask_N_A, mask_N_L)
    mask_expand = mask.unsqueeze(-1)
    mask_D = mask_expand * mask_D
    mask_A = mask_expand * mask_A
    mask_L = mask_expand * mask_L

    D = mask_D * D
    A = mask_A * A
    L = mask_L * L

    if not return_masks:
        return D, A, L
    else:
        return D, A, L, mask_D, mask_A, mask_L


##################################
# Geometric Utility Functions
##################################

def normed_vec(V: torch.Tensor, distance_eps: float = DISTANCE_EPS) -> torch.Tensor:
    """Compute normalized vectors with smooth gradients near zero.

    This function normalizes vectors using a smoothed magnitude calculation to avoid
    gradient discontinuities at zero. The normalization is computed as:
        U = V / sqrt(|V|^2 + eps)
    
    This ensures differentiability everywhere and stable gradients during training,
    which is critical for deep learning applications.

    Args:
        V (torch.Tensor): Batch of vectors with shape (..., num_dims) where the last
            dimension contains the vector components
        distance_eps (float): Small smoothing parameter added to squared magnitude
            before taking square root. Defaults to DISTANCE_EPS (typically 1e-10).

    Returns:
        torch.Tensor: Batch of unit vectors with same shape as input (..., num_dims)

    Note:
        Without epsilon smoothing, gradient of |V| is undefined at V=0, causing
        numerical instability. The epsilon creates a smooth approximation.
    """
    # Unit vector from i to j
    mag_sq = (V**2).sum(dim=-1, keepdim=True)
    mag = torch.sqrt(mag_sq + distance_eps)
    U = V / mag
    return U


def normed_cross(
    V1: torch.Tensor, V2: torch.Tensor, distance_eps: float = DISTANCE_EPS
) -> torch.Tensor:
    """Normalized cross product between vectors.

    This normalization is computed as `U = V / sqrt(|V|^2 + eps)` to avoid cusps
    and gradient discontinuities.

    Args:
        V1 (Tensor): Batch of vectors with shape `(..., 3)`.
        V2 (Tensor): Batch of vectors with shape `(..., 3)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.

    Returns:
        C (Tensor): Batch of cross products `v_1 x v_2` with shape `(..., 3)`.
    """
    C = normed_vec(torch.cross(V1, V2, dim=-1), distance_eps=distance_eps)
    return C


def lengths(
    atom_i: torch.Tensor, atom_j: torch.Tensor, distance_eps: float = DISTANCE_EPS
) -> torch.Tensor:
    """Batched bond lengths given batches of atom i and j.

    Args:
        atom_i (Tensor): Atom `i` coordinates with shape `(..., 3)`.
        atom_j (Tensor): Atom `j` coordinates with shape `(..., 3)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.

    Returns:
        L (Tensor): Elementwise bond lengths `||x_i - x_j||` with shape `(...)`.
    """
    # Bond length of i-j
    dX = atom_j - atom_i
    L = torch.sqrt((dX**2).sum(dim=-1) + distance_eps)
    return L


def angles(
    atom_i: torch.Tensor,
    atom_j: torch.Tensor,
    atom_k: torch.Tensor,
    distance_eps: float = DISTANCE_EPS,
    degrees: bool = False,
) -> torch.Tensor:
    """Batched bond angles given atoms `i-j-k`.

    Args:
        atom_i (Tensor): Atom `i` coordinates with shape `(..., 3)`.
        atom_j (Tensor): Atom `j` coordinates with shape `(..., 3)`.
        atom_k (Tensor): Atom `k` coordinates with shape `(..., 3)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.
        degrees (bool, optional): If True, convert to degrees. Default: False.

    Returns:
        A (Tensor): Elementwise bond angles with shape `(...)`.
    """
    # Bond angle of i-j-k
    U_ji = normed_vec(atom_i - atom_j, distance_eps=distance_eps)
    U_jk = normed_vec(atom_k - atom_j, distance_eps=distance_eps)
    inner_prod = torch.einsum("bix,bix->bi", U_ji, U_jk)
    inner_prod = torch.clamp(inner_prod, -1, 1)
    A = torch.acos(inner_prod)
    if degrees:
        A = A * 180.0 / np.pi
    return A


def dihedrals(
    atom_i: torch.Tensor,
    atom_j: torch.Tensor,
    atom_k: torch.Tensor,
    atom_l: torch.Tensor,
    distance_eps: float = DISTANCE_EPS,
    degrees: bool = False,
) -> torch.Tensor:
    """Batched bond dihedrals given atoms `i-j-k-l`.

    Args:
        atom_i (Tensor): Atom `i` coordinates with shape `(..., 3)`.
        atom_j (Tensor): Atom `j` coordinates with shape `(..., 3)`.
        atom_k (Tensor): Atom `k` coordinates with shape `(..., 3)`.
        atom_l (Tensor): Atom `l` coordinates with shape `(..., 3)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.
        degrees (bool, optional): If True, convert to degrees. Default: False.

    Returns:
        D (Tensor): Elementwise bond dihedrals with shape `(...)`.
    """
    U_ij = normed_vec(atom_j - atom_i, distance_eps=distance_eps)
    U_jk = normed_vec(atom_k - atom_j, distance_eps=distance_eps)
    U_kl = normed_vec(atom_l - atom_k, distance_eps=distance_eps)
    normal_ijk = normed_cross(U_ij, U_jk, distance_eps=distance_eps)
    normal_jkl = normed_cross(U_jk, U_kl, distance_eps=distance_eps)
    # _inner_product = lambda a, b: torch.einsum("bix,bix->bi", a, b)
    _inner_product = lambda a, b: (a * b).sum(-1)
    cos_dihedrals = _inner_product(normal_ijk, normal_jkl)
    angle_sign = _inner_product(U_ij, normal_jkl)
    cos_dihedrals = torch.clamp(cos_dihedrals, -1, 1)
    D = torch.sign(angle_sign) * torch.acos(cos_dihedrals)
    if degrees:
        D = D * 180.0 / np.pi
    return D


##################################
# Feature Encoding Functions
##################################

def rbf_expansion(
    h: torch.Tensor,
    value_min: float = 0.0,
    value_max: float = 30.0,
    num_rbf: int = 32,
):
    """Encode scalar values using radial basis functions (RBF) for neural network input.

    This function expands scalar values (typically distances) into a higher-dimensional
    representation using Gaussian radial basis functions. RBF encoding helps neural networks
    learn smooth, continuous functions of distance by providing a localized, distributed
    representation.

    The RBF centers are uniformly spaced between value_min and value_max, and the width
    (standard deviation) is set to the spacing between centers.

    Args:
        h (torch.Tensor): Input scalar values to encode, shape (..., 1) or (...,)
        value_min (float): Minimum value for RBF centers. Defaults to 0.0.
        value_max (float): Maximum value for RBF centers. Defaults to 30.0 (Angstroms).
        num_rbf (int): Number of radial basis functions to use. Defaults to 32.

    Returns:
        torch.Tensor: RBF-encoded features with shape (..., num_rbf) where each
            dimension corresponds to one RBF kernel

    Note:
        For distance encoding in molecular structures, typical ranges are 0-30 Angstroms
        with 16-64 RBF functions, providing good coverage of typical interaction distances.
    """
    rbf_centers = torch.linspace(value_min, value_max, num_rbf)
    std = (rbf_centers[1] - rbf_centers[0]).item()
    shape = list(h.shape)
    shape_ones = [1 for _ in range(len(shape))] + [-1]
    rbf_centers = rbf_centers.view(shape_ones)
    h = torch.exp(-(((h.unsqueeze(-1) - rbf_centers) / std) ** 2))
    h = h.view(shape[:-1] + [-1])
    return h


def positional_encoding(inputs, num_posenc=32, period_range=(1.0, 1000.0)):
    """Encode scalar positions using sinusoidal positional encodings.

    This function generates sinusoidal positional encodings similar to those used in
    Transformer models. The encodings use multiple frequencies to capture positional
    information at different scales, enabling the model to learn position-dependent
    patterns in RNA sequences.

    The encoding uses sine and cosine functions at logarithmically spaced frequencies:
        PE(pos, 2i) = sin(pos / period_i)
        PE(pos, 2i+1) = cos(pos / period_i)

    Args:
        inputs (torch.Tensor): Input positions to encode, shape (..., num_positions)
        num_posenc (int): Total number of encoding dimensions (must be even, as half
            are sine and half are cosine). Defaults to 32.
        period_range (tuple): (min_period, max_period) defining the range of wavelengths
            used for encoding. Defaults to (1.0, 1000.0).

    Returns:
        torch.Tensor: Sinusoidal positional encodings with shape (..., num_posenc)

    Note:
        The use of multiple frequencies allows the model to attend to both local and
        long-range positional patterns, important for capturing RNA structural motifs
        at various length scales.
    """
    num_frequencies = num_posenc // 2
    log_bounds = np.log10(period_range)
    p = torch.logspace(log_bounds[0], log_bounds[1], num_frequencies, base=10.0)
    w = 2 * math.pi / p

    batch_dims = list(inputs.shape)[:-1]
    # (..., 1, num_out) * (..., num_in, 1)
    w = w.reshape(len(batch_dims) * [1] + [1, -1])
    h = w * inputs[..., None]
    h = torch.cat([h.cos(), h.sin()], -1).reshape(batch_dims + [-1])
    return h


def internal_vecs(X, mask_atoms):
    """Compute internal orientation vectors from RNA backbone coordinates.

    This function computes normalized orientation vectors that capture the local geometry
    of the RNA backbone in an SE(3)-equivariant manner. Four types of vectors are computed:
        1. P → C4': Phosphate to sugar direction
        2. C4' → N: Sugar to base direction
        3. Forward: C4'(i) → C4'(i+1) along chain
        4. Backward: C4'(i) → C4'(i-1) along chain

    These vectors provide geometric context for each nucleotide's local environment,
    capturing both intra-residue and inter-residue orientations.

    Args:
        X (torch.Tensor): Backbone atom coordinates with shape
            [num_conf, num_res, num_bb_atoms, 3] where num_bb_atoms is typically 3
            for (P, C4', N1/N9)
        mask_atoms (torch.Tensor): Atom validity mask with shape
            [num_conf, num_res, num_bb_atoms] indicating which atoms have valid coordinates

    Returns:
        torch.Tensor: Stacked normalized orientation vectors with shape
            [num_conf, num_res, 4, 3] where dimension 2 indexes the four vector types:
            [P→C4', C4'→N, forward, backward]

    Note:
        Missing atoms are handled by masking: vectors involving missing atoms are
        zeroed out rather than producing invalid values.
    """
    # Relative displacement vectors along backbone
    # X : num_conf x num_res x num_bb_atoms x 3
    # mask_atoms: num_conf x num_res x num_bb_atoms
    p, c4p, n = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    mask_p, mask_c4p, mask_n = (
        mask_atoms[:, :, 0, None],
        mask_atoms[:, :, 1, None],
        mask_atoms[:, :, 2, None],
    )
    n = (n - c4p) * mask_n * mask_c4p
    p = (p - c4p) * mask_p * mask_c4p
    forward = F.pad(c4p[:, 1:] - c4p[:, :-1], [0, 0, 0, 1]) * mask_c4p
    backward = F.pad(c4p[:, :-1] - c4p[:, 1:], [0, 0, 1, 0]) * mask_c4p
    return torch.cat(
        [
            normed_vec(p).unsqueeze_(-2),
            normed_vec(n).unsqueeze_(-2),
            normed_vec(forward).unsqueeze_(-2),
            normed_vec(backward).unsqueeze_(-2),
        ],
        dim=-2,
    )


def normalize(tensor, dim=-1):
    """Normalize tensor along specified dimension with NaN handling.

    This function normalizes a tensor by dividing by its L2 norm along the specified
    dimension, with special handling to avoid NaN values that can arise from division
    by zero or very small norms.

    Args:
        tensor (torch.Tensor): Input tensor to normalize
        dim (int): Dimension along which to compute the norm and normalize.
            Defaults to -1 (last dimension).

    Returns:
        torch.Tensor: Normalized tensor with same shape as input, where any NaN
            values (from zero-norm vectors) are replaced with zeros

    Note:
        This is safer than standard normalization for handling edge cases where
        vectors may have zero or near-zero magnitude, which can occur with missing
        data or degenerate geometries.
    """
    return torch.nan_to_num(torch.div(tensor, torch.linalg.norm(tensor, dim=dim, keepdim=True)))
