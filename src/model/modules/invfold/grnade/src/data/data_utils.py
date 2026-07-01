import os
from typing import Any, List, Literal, Optional

import biotite
# import cpdb
import numpy as np
import pandas as pd
import torch
from biotite.structure import apply_residue_wise
from biotite.structure import sasa as get_sasa
from biotite.structure.io import load_structure

from src.constants import FILL_VALUE, PURINES, PYRIMIDINES, RNA_ATOMS, RNA_NUCLEOTIDES
# from src.data.sec_struct_utils import pdb_to_sec_struct


# def pdb_to_tensor(
#     filepath: str,
#     return_sec_struct: bool = True,
#     return_sasa: bool = True,
#     keep_insertions: bool = True,
#     keep_pseudoknots: bool = True,
# ):
#     """
#     Reads a PDB file of an RNA structure and returns:
#     - sequence: str - RNA sequence
#     - coords: torch.FloatTensor of shape ``(length, 37, 3)`` - 3D coordinates
#     - sec_struct: str - secondary structure in dot-bracket notation
#     - sasa: np.array of shape ``(length, )`` - solvent accessible surface area

#     Credit: Arian Jamasb, graphein (https://github.com/a-r-j/graphein)

#     Args:
#         filepath (str): Path to PDB file.
#         return_sec_struct (bool, optional): Whether to return secondary structure.
#             Defaults to True.
#         return_sasa (bool, optional): Whether to return solvent accessible surface
#             area. Defaults to True.
#         keep_insertions (bool, optional): Whether to keep insertions in the
#             PDB file. Defaults to True.
#         keep_pseudoknots (bool, optional): Whether to keep pseudoknots in
#             secondary structure. Defaults to False.

#     Returns:
#         sequence (str): RNA sequence
#         coords (torch.FloatTensor): 3D coordinates
#         sec_struct (str): Secondary structure in dot-bracket notation
#         sasa (np.array): Solvent accessible surface area of shape
#     """

#     # read pdb to dataframe
#     df = cpdb.parse(filepath, df=True)
#     if not keep_insertions:
#         df = remove_insertions(df)

#     # create unique residue id
#     df["residue_id"] = (
#         df["chain_id"] + ":" + df["residue_name"] + ":" + df["residue_number"].astype(str)
#     )
#     if keep_insertions:
#         df["residue_id"] = df.residue_id + ":" + df.insertion

#     # get sequence
#     nt_list = [res.split(":")[1] for res in df.residue_id.unique()]
#     # replace non-standard nucleotides with placeholder
#     nt_list = [nt if nt in RNA_NUCLEOTIDES else "_" for nt in nt_list]
#     sequence = "".join(nt_list)
#     if len(sequence) <= 1:
#         return  # do not include single bases as data points

#     # get 3D coordinates (centered at origin)
#     coords = df_to_tensor(df, center=True)
#     assert coords.shape[0] == len(sequence), "Sequence and coordinates must be the same length"

#     sec_struct = None
#     if return_sec_struct:
#         # get secondary structure
#         sec_struct = pdb_to_sec_struct(filepath, sequence, keep_pseudoknots)
#         assert len(sec_struct) == len(
#             sequence
#         ), "Sequence and secondary structure must be the same length"

#     sasa = None
#     if return_sasa:
#         # get solvent accessibile surface area
#         atom_array = load_structure(filepath)
#         sasa = apply_residue_wise(atom_array, get_sasa(atom_array), np.nansum)
#         assert len(sasa) == len(sequence), "Sequence and SASA must be the same length"

#     return sequence, coords, sec_struct, sasa


def df_to_tensor(
    df: pd.DataFrame,
    atoms_to_keep: List[str] = RNA_ATOMS,
    fill_value: float = FILL_VALUE,
    center: bool = True,
):
    """Transforms a DataFrame of an RNA structure into a ``length x num_atoms x 3`` tensor.

    Credit: Arian Jamasb, graphein (https://github.com/a-r-j/graphein)

    :param df: DataFrame of protein structure.
    :type df: pd.DataFrame
    :param atoms_to_keep: List of atom types to retain in the tensor.
    :type atoms_to_keep: List[str]
    :param fill_value: Value to fill missing entries with. Defaults to ``1e-5``.
    :type fill_value: float
    :param center: Whether to center the structure at the origin. Defaults to ``True``.
    :type center: bool
    :returns: ``Length x Num_Atoms (default 37) x 3`` tensor.
    :rtype: torch.Tensor
    """
    if center:
        df.x_coord -= df.x_coord.mean()
        df.y_coord -= df.y_coord.mean()
        df.z_coord -= df.z_coord.mean()

    num_residues = len([res.split(":")[1] for res in df.residue_id.unique()])
    df = df.loc[df["atom_name"].isin(atoms_to_keep)]
    residue_indices = pd.factorize(np.array(df.residue_id))[0]
    atom_indices = df["atom_name"].map(lambda x: atoms_to_keep.index(x)).values

    positions = (torch.zeros((num_residues, len(atoms_to_keep), 3)) + fill_value).float()
    positions[residue_indices, atom_indices] = torch.tensor(
        df[["x_coord", "y_coord", "z_coord"]].values
    ).float()
    return positions


def remove_insertions(df: pd.DataFrame, keep: Literal["first", "last"] = "first") -> pd.DataFrame:
    """This function removes insertions from PDB DataFrames.

    Credit: Arian Jamasb, graphein (https://github.com/a-r-j/graphein)

    :param df: RNA Structure dataframe to remove insertions from.
    :type df: pd.DataFrame
    :param keep: Specifies which insertion to keep. Options are ``"first"`` or
        ``"last"``. Default is ``"first"``.
    :type keep: Literal["first", "last"]
    :return: RNA structure dataframe with insertions removed
    :rtype: pd.DataFrame
    """
    # Catches unnamed insertions
    duplicates = df.duplicated(
        subset=["chain_id", "residue_number", "atom_name", "alt_loc"],
        keep=keep,
    )
    df = df[~duplicates]

    return filter_dataframe(df, by_column="insertion", list_of_values=[""], boolean=True)


def filter_dataframe(
    dataframe: pd.DataFrame,
    by_column: str,
    list_of_values: List[Any],
    boolean: bool,
) -> pd.DataFrame:
    """Filter function for DataFrame.

    Filters the DataFrame such that the ``by_column`` values have to be
    in the ``list_of_values`` list if ``boolean == True``, or not in the list
    if ``boolean == False``.

    Credit: Arian Jamasb, graphein (https://github.com/a-r-j/graphein)

    :param dataframe: pd.DataFrame to filter.
    :type dataframe: pd.DataFrame
    :param by_column: str denoting column of DataFrame to filter.
    :type by_column: str
    :param list_of_values: List of values to filter with.
    :type list_of_values: List[Any]
    :param boolean: indicates whether to keep or exclude matching
        ``list_of_values``. ``True`` -> in list, ``False`` -> not in list.
    :type boolean: bool
    :returns: Filtered DataFrame.
    :rtype: pd.DataFrame
    """
    df = dataframe.copy()
    df = df[df[by_column].isin(list_of_values) == boolean]
    df.reset_index(inplace=True, drop=True)

    return df


def get_full_atom_coords(atom_tensor: torch.FloatTensor, fill_value: float = FILL_VALUE):
    """Converts an ``AtomTensor`` to a full atom representation.

    Return tuple of coords ``(N_atoms x 3)``, residue_index ``(N_atoms)``,
    atom_type ``(N_atoms x [0-27])`` with 27 possible RNA atoms.

    Credit: Arian Jamasb, graphein (https://github.com/a-r-j/graphein)

    :param atom_tensor: AtomTensor of shape``(N_residues, N_atoms, 3)``
    :type atom_tensor: torch.FloatTensor
    :param fill_value: Value used to fill missing values. Defaults to ``1e-5``.
    :return: Tuple of coords, residue_index, atom_type
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    # Get number of atoms per residue
    filled = atom_tensor[:, :, 0] != fill_value
    nz = filled.nonzero()
    residue_index = nz[:, 0]
    atom_type = nz[:, 1]
    coords = atom_tensor.reshape(-1, 3)
    coords = coords[coords != fill_value].reshape(-1, 3)
    return coords, residue_index, atom_type


def get_c4p_coords(
    atom_tensor: torch.FloatTensor,
    index: int = RNA_ATOMS.index("C4'"),
):
    """Returns tensor of C4' atom coordinates: ``(L x 3)``

    :param atom_tensor: AtomTensor of shape ``(N_residues, N_atoms, 3)``
    :type atom_tensor: torch.FloatTensor
    :param index: Index of C4' atom in dimension 1 of the AtomTensor.
    :type index: int
    """
    if atom_tensor.ndim == 2:
        # already C4' coords
        return atom_tensor
    elif atom_tensor.size(1) == 3:
        # backbone coords tensor to C4' coords
        return atom_tensor[:, 1, :]
    else:  # if atom_tensor.size(1) == len(RNA_ATOMS):
        # full atom tensor to C4' coords
        return atom_tensor[:, index, :]


def get_backbone_coords(
    atom_tensor: torch.FloatTensor,
    sequence: str,
    pyrimidine_bb_indices: List[int] = [
        RNA_ATOMS.index("P"),
        RNA_ATOMS.index("C4'"),
        RNA_ATOMS.index("N1"),
    ],
    purine_bb_indices: List[int] = [
        RNA_ATOMS.index("P"),
        RNA_ATOMS.index("C4'"),
        RNA_ATOMS.index("N9"),
    ],
    fill_value: float = 1e-5,
):
    """Returns tensor of backbone atom coordinates: ``(L x 3 x 3)``

    Note: One can easily change the backbone representation here by changing
          the indices of the atoms to include in the backbone. The running
          example in the docstrings uses a 3-bead coarse grained representation.

    :param atom_tensor: AtomTensor of shape ``(N_residues, N_atoms, 3)``
    :type atom_tensor: torch.FloatTensor
    :param pyrimidine_bb_indices: List of indices of ``[P, C4', N1]`` atoms (in
        order) for C and U nucleotides.
    :type pyrimidine_bb_indices: List[int]
    :param purine_bb_indices: List of indices of ``[P, C4', N9]`` atoms (in
        order) for A and G nucleotides.
    :type purine_bb_indices: List[int]
    :param fill_value: Value to fill missing entries with. Defaults to ``1e-5``.
    :type fill_value: float
    """
    # check that sequence is str
    assert isinstance(sequence, str), "Sequence must be a string"

    # get indices of purine/pyrimidine bases in sequence
    purine_indices = [i for i, base in enumerate(sequence) if base in PURINES]
    pyrimidine_indices = [i for i, base in enumerate(sequence) if base in PYRIMIDINES]

    # create tensor of backbone atoms
    backbone_tensor = (
        torch.zeros((atom_tensor.shape[0], len(purine_bb_indices), 3)) + fill_value
    ).float()
    backbone_tensor[purine_indices] = atom_tensor[purine_indices][:, purine_bb_indices, :]
    backbone_tensor[pyrimidine_indices] = atom_tensor[pyrimidine_indices][
        :, pyrimidine_bb_indices, :
    ]
    return backbone_tensor


def get_center(
    x: torch.FloatTensor,
    c4p_only: bool = True,
    fill_value: float = FILL_VALUE,
):
    """Returns the center of an RNA.

    :param x: Point Cloud to Center. Torch tensor of shape ``(Length , 3)`` or
        ``(Length, num atoms, 3)``.
    :param c4p_only: If ``True``, only the C4' atoms will be used to compute
        the center. Default is ``True``.
    :type c4p_only: bool
    :param fill_value: Value used to denote missing atoms. Default is ``1e-5``.
    :type fill_value: float
    :return: Torch tensor of shape ``(N,D)`` -- Center of Point Cloud
    :rtype: torch.FloatTensor
    """
    if x.ndim != 3:
        return x.mean(dim=0)
    if c4p_only:
        return get_c4p_coords(x).mean(dim=0)

    x_flat, _, _ = get_full_atom_coords(x, fill_value=fill_value)
    return x_flat.mean(dim=0)
