import os
import subprocess
import time
from typing import Any, List, Literal, Optional

import numpy as np
import pandas as pd
from Bio import SeqIO

from src.constants import DATA_PATH


def cluster_sequence_identity(
    input_sequences, identity_threshold=0.8, word_size=4, input_file="input", output_file="output"
):
    """Cluster sequences based on sequence similarity using CD-HIT.

     Args:
        input_sequences: List of sequences to cluster
        identity_threshold: Sequence identity threshold (0-1). Default: 0.8
        word_size: Word size for comparison. Default: 4
        input_file: Temporary input file name. Default: "input"
        output_file: Temporary output file name. Default: "output"

    Returns:
        dict: Mapping from sequence ID to cluster ID

    Notes:
    - https://manpages.ubuntu.com/manpages/impish/man1/cd-hit-est.1.html
    - How to chose word size? https://github.com/weizhongli/cdhit/wiki/3.-User's-Guide#user-content-CDHITEST
       -n 10, 11 for thresholds 0.95 ~ 1.0
       -n 8,9    for thresholds 0.90 ~ 0.95
       -n 7      for thresholds 0.88 ~ 0.9
       -n 6      for thresholds 0.85 ~ 0.88
       -n 5      for thresholds 0.80 ~ 0.85
       -n 4      for thresholds 0.75 ~ 0.8
    """
    t0 = time.time()

    # Write input sequences to the temporary input file
    SeqIO.write(input_sequences, input_file, "fasta")

    # Run CD-HIT-EST
    cmd = [
        "cd-hit-est",
        "-i",
        input_file,
        "-o",
        output_file,
        "-c",
        str(identity_threshold),  # Sequence identity threshold (e.g., 90%)
        "-n",
        str(word_size),  # Word size for sequence comparisson, larger is better (default: 2)
        "-M",
        str(0),  # Memory limit in MB (0 for unlimited)
        "-T",
        str(0),  # Number of threads (0 for using all CPUs)
    ]
    subprocess.run(cmd, check=True)

    # Read clustered sequences from the temporary output file
    # clustered_sequences = list(SeqIO.parse(output_file, "fasta"))

    # Process the clustering output
    seq_id_to_cluster = {}
    with open(output_file + ".clstr") as f:
        current_cluster = None
        for line in f:
            if line.startswith(">"):
                current_cluster = int(line.strip().split(" ")[1])
            else:
                seq_id = line.split(">")[1].split("...")[0]
                seq_id_to_cluster[seq_id] = current_cluster

    # Delete temporary files
    os.remove(input_file)
    os.remove(output_file)
    os.remove(output_file + ".clstr")

    print(f"Total CPU time {(time.time() - t0)/60:.2f} m")

    return seq_id_to_cluster


def parse_qtmclust_cluster_file(file_path: str) -> List[List[Any]]:
    # Return a list of lists, where each inner list is a cluster of structures
    clusters = {}
    with open(file_path) as file:
        for line in file:
            columns = line.strip().split("\t")
            valid_columns = [col for col in columns if col]  # filter out any empty columns
            # NOTE: the representative structure is the first (col=0) structure for a given cluster
            cluster_repr = valid_columns[0]
            clusters[cluster_repr] = valid_columns
    return list(clusters.values())


def run_qtmclust(
    chain_dir: str,
    chain_list_filepath: str,
    qtmclust_exec_path: str,
    output_cluster_filepath: Optional[str] = None,
    tm_cluster_threshold: float = 0.45,
    chain_ter_mode: Literal[0, 1, 2, 3] = 3,
    chain_split_mode: Literal[0, 1, 2] = 0,
) -> Optional[pd.DataFrame]:
    # Run qTMclust structural similarity clustering
    # For more information on `chain_ter_mode` and `chain_split_mode`, please see:
    # https://github.com/pylelab/USalign/blob/58b42af9d58436279c21b4f4074db87f072fcc21/qTMclust.cpp#L72
    # and
    # https://github.com/pylelab/USalign/blob/58b42af9d58436279c21b4f4074db87f072fcc21/qTMclust.cpp#L78
    cmd = [
        qtmclust_exec_path,
        "-dir",
        (chain_dir if chain_dir.endswith("/") else chain_dir + "/"),
        chain_list_filepath,
        "-TMcut",
        str(tm_cluster_threshold),
        "-ter",
        str(chain_ter_mode),
        "-split",
        str(chain_split_mode),
    ]
    if output_cluster_filepath is not None:
        cmd += ["-o", output_cluster_filepath]
    subprocess.run(" ".join(cmd), capture_output=True, shell=True)  # nosec
    if output_cluster_filepath is not None:
        output_clusters = parse_qtmclust_cluster_file(output_cluster_filepath)
        return output_clusters


def cluster_structure_similarity(
    input_pdb_files,
    similarity_threshold: float = 0.45,
    chain_list_filepath: str = "chain_list",
    output_cluster_filepath: str = "cluster.txt",
    chain_dir: str = os.path.join(DATA_PATH, "raw"),
    qtmclust_exec_path: str = "~/USalign/qTMclust",
):
    """Cluster structures based on their structural similarity using qTMclust.

    Credit: Alex Morehead

    Notes:
    - https://zhanggroup.org/US-align/
    - TM-score has values in (0,1] with 1 indicating an identical structure match,
      where a TM-score ≥0.5 (or 0.45) means the structures share the same global
      topology for proteins (or RNAs).
    """
    t0 = time.time()

    with open(chain_list_filepath, "w") as f:
        for pdb_file_index, pdb_file in enumerate(input_pdb_files):
            # record the name of each PDB file in a temporary text file input
            sample_name_without_extension = os.path.basename(os.path.splitext(pdb_file)[0])
            sample_name_postfix = "" if pdb_file_index == (len(input_pdb_files) - 1) else "\n"
            f.write(f"{sample_name_without_extension}{sample_name_postfix}")

    clustered_structures = run_qtmclust(
        chain_dir=chain_dir,
        chain_list_filepath=chain_list_filepath,
        qtmclust_exec_path=qtmclust_exec_path,
        output_cluster_filepath=output_cluster_filepath,
        tm_cluster_threshold=similarity_threshold,  # note: clusters two chains if their TM-score is `similarity_threshold` or greater
        chain_ter_mode=0,  # note: reads all chains
        chain_split_mode=0,  # note: parses all chains in a complex as a single chain
    )

    # Delete temporary files
    os.remove(chain_list_filepath)
    os.remove(output_cluster_filepath)

    print(f"Total CPU time {(time.time() - t0)/60:.2f} m")

    return clustered_structures
