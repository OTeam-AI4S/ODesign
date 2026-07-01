import os
import subprocess
from datetime import datetime
from typing import Any, List, Literal, Optional

import biotite
import numpy as np
import wandb
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from biotite.structure import dot_bracket_from_structure
from biotite.structure.io import load_structure

from .constants import DOTBRACKET_TO_NUM, ETERNAFOLD_PATH, PROJECT_PATH, X3DNA_PATH


# def pdb_to_sec_struct(
#     pdb_file_path: str,
#     sequence: str,
#     keep_pseudoknots: bool = False,
#     x3dna_path: str = os.path.join(X3DNA_PATH, "bin/find_pair"),
#     max_len_for_biotite: int = 1000,
# ) -> str:
#     """Get secondary structure in dot-bracket notation from a PDB file.

#     Args:
#         pdb_file_path (str): Path to PDB file.
#         sequence (str): Sequence of RNA molecule.
#         keep_pseudoknots (bool, optional): Whether to keep pseudoknots in
#             secondary structure. Defaults to False.
#         x3dna_path (str, optional): Path to x3dna find_pair tool.
#         max_len_for_biotite (int, optional): Maximum length of sequence for
#             which to use biotite. Otherwise use X3DNA Defaults to 1000.
#     """
#     if len(sequence) < max_len_for_biotite:
#         try:
#             # get secondary structure using biotite
#             atom_array = load_structure(pdb_file_path)
#             sec_struct = dot_bracket_from_structure(atom_array)[0]
#             if not keep_pseudoknots:
#                 # replace all characters that are not '.', '(', ')' with '.'
#                 sec_struct = "".join(
#                     [dotbrac if dotbrac in [".", "(", ")"] else "." for dotbrac in sec_struct]
#                 )

#         except Exception as e:
#             # biotite fails for very short seqeunces
#             if "out of bounds for array" not in str(e):
#                 raise e
#             # get secondary structure using x3dna find_pair tool
#             # does not support pseudoknots
#             sec_struct = x3dna_to_sec_struct(pdb_to_x3dna(pdb_file_path, x3dna_path), sequence)

#     else:
#         # get secondary structure using x3dna find_pair tool
#         # does not support pseudoknots
#         sec_struct = x3dna_to_sec_struct(pdb_to_x3dna(pdb_file_path, x3dna_path), sequence)

#     return sec_struct


# def pdb_to_x3dna(
#     pdb_file_path: str, x3dna_path: str = os.path.join(X3DNA_PATH, "bin/find_pair")
# ) -> List[str]:
#     # Run x3dna find_pair tool
#     cmd = [
#         x3dna_path,
#         pdb_file_path,
#     ]
#     output = subprocess.run(cmd, check=True, capture_output=True).stdout.decode("utf-8")
#     output = output.split("\n")

#     # Delete temporary files
#     # os.remove("./bestpairs.pdb")
#     # os.remove("./bp_order.dat")
#     # os.remove("./col_chains.scr")
#     # os.remove("./col_helices.scr")
#     # os.remove("./hel_regions.pdb")
#     # os.remove("./ref_frames.dat")

#     return output


# def x3dna_to_sec_struct(output: List[str], sequence: str) -> str:
#     # Secondary structure in dot-bracket notation
#     num_base_pairs = int(output[3].split()[0])
#     sec_struct = ["."] * len(sequence)
#     for i in range(1, num_base_pairs + 1):
#         line = output[4 + i].split()
#         start, end = int(line[0]), int(line[1])
#         sec_struct[start - 1] = "("
#         sec_struct[end - 1] = ")"
#     return "".join(sec_struct)


# def predict_sec_struct(
#     sequence: Optional[str] = None,
#     fasta_file_path: Optional[str] = None,
#     eternafold_path: str = os.path.join(ETERNAFOLD_PATH, "src/contrafold"),
#     n_samples: int = 1,
# ) -> str:
#     """Predict secondary structure using EternaFold.

#     Notes:
#     - EternaFold does not support pseudoknots.
#     - EternaFold only supports single chains in a fasta file.
#     - When sampling multiple structures, EternaFold only supports nsamples=100.

#     Args:
#         sequence (str, optional): Sequence of RNA molecule. Defaults to None.
#         fasta_file_path (str, optional): Path to fasta file. Defaults to None.
#         eternafold_path (str, optional): Path to EternaFold. Defaults to ETERNAFOLD_PATH env variable.
#         n_samples (int, optional): Number of samples to take. Defaults to 1.
#     """
#     if sequence is not None:
#         assert fasta_file_path is None
#         # Write sequence to temporary fasta file
#         current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
#         try:
#             fasta_file_path = os.path.join(wandb.run.dir, f"temp_{current_datetime}.fasta")
#         except AttributeError:
#             fasta_file_path = os.path.join(PROJECT_PATH, f"temp_{current_datetime}.fasta")
#         SeqIO.write(SeqRecord(Seq(sequence), id="temp"), fasta_file_path, "fasta")

#     # Run EternaFold
#     if n_samples > 1:
#         assert n_samples == 100, "EternaFold using subprocess only supports nsamples=100"
#         cmd = [
#             eternafold_path,
#             "sample",
#             fasta_file_path,
#             # f" --nsamples {n_samples}",
#             # It seems like EternaFold using subprocess can only sample the default nsamples=100...
#             # Reason: unknown for now
#         ]
#     else:
#         cmd = [
#             eternafold_path,
#             "predict",
#             fasta_file_path,
#         ]

#     output = subprocess.run(cmd, check=True, capture_output=True).stdout.decode("utf-8")

#     # Delete temporary files
#     if sequence is not None:
#         os.remove(fasta_file_path)

#     if n_samples > 1:
#         return output.split("\n")[:-1]
#     else:
#         return [output.split("\n")[-2]]


def dotbracket_to_paired(sec_struct: str) -> np.ndarray:
    """Return whether each residue is paired (1) or unpaired (0) given secondary structure in dot-
    bracket notation."""
    is_paired = np.zeros(len(sec_struct), dtype=np.int8)
    for i, c in enumerate(sec_struct):
        if c == "(" or c == ")":
            is_paired[i] = 1
    return is_paired


def dotbracket_to_num(sec_struct: str) -> np.ndarray:
    """Convert secondary structure in dot-bracket notation to numerical representation."""
    return np.array([DOTBRACKET_TO_NUM[c] for c in sec_struct])


def dotbracket_to_adjacency(
    sec_struct: str,
    keep_pseudoknots: bool = False,
) -> np.ndarray:
    """Convert secondary structure in dot-bracket notation to adjacency matrix."""
    n = len(sec_struct)
    adj = np.zeros((n, n), dtype=np.int8)

    if keep_pseudoknots == False:
        stack = []
        for i, db_char in enumerate(sec_struct):
            if db_char == "(":
                stack.append(i)
            elif db_char == ")" and len(stack) > 0:
                j = stack.pop()
                adj[i, j] = 1
                adj[j, i] = 1
    else:
        stack = {"(": [], "[": [], "<": [], "{": []}
        pop = {")": "(", "]": "[", ">": "<", "}": "{"}
        for i, db_char in enumerate(sec_struct):
            if db_char in stack:
                stack[db_char].append((i, db_char))
            elif db_char in pop and len(stack[pop[db_char]]) > 0:
                forward_bracket = stack[pop[db_char]].pop()
                adj[forward_bracket[0], i] = 1
                adj[i, forward_bracket[0]] = 1
    return adj
