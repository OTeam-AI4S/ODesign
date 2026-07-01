import biotite
import biotite.structure.graphics as graphics
import draw_rna.draw as d
import lovely_tensors as lt
import matplotlib.pyplot as plt
import numpy as np
from biotite.structure import base_pairs_from_dot_bracket, pseudoknots
from draw_rna.draw_utils import seq2col
from lovely_numpy import lo
from mpl_toolkits.mplot3d import Axes3D

lt.monkey_patch()


def print_rna_data(data):
    for key, value in data.items():
        if key == "sasa_list":
            print(f"{key}:")
            for sasa in value:
                print(f"\t{lo(sasa)}")
        elif key == "coords_list":
            print(f"{key}:")
            for coords in value:
                print(f"\t{lo(coords)}")
        elif key == "sec_struct_list":
            print(f"{key}:")
            for sec_struct in value:
                print(f"\t{sec_struct}")
        elif key == "eterna_sec_struct_list":
            print(f"{key}:")
            for sec_struct in value:
                print(f"\t{sec_struct}")
        elif key == "rmsds_list":
            print(f"{key}:")
            for pair, rmsd in value.items():
                print(f"\t{pair}, {rmsd}")
        else:
            print(f"{key}:\n\t{value}")


def draw_2d_struct(
    seq,
    secstruct,
    c=None,
    line=False,
    large_mode=False,
    cmap="viridis",
    rotation=0,
    vmin=None,
    vmax=None,
    alpha=None,
    ax=None,
):
    """Draw sequence with secondary structure.

    Inputs:
    c (string or array-like).  If string, characters must correspond to colors.
    If array-like obj, used as mapping for colormap (provided in cmap), or a string.
    line (bool): draw secstruct as line.
    large_mode: draw outer loop as straight line.
    rotation: rotate molecule (in degrees).

    Source: https://github.com/DasLab/draw_rna
    """
    if c is not None:
        assert len(c) == len(seq)
        if isinstance(c[0], float):
            d.draw_rna(
                seq,
                secstruct,
                c,
                line=line,
                ext_color_file=True,
                cmap_name=cmap,
                vmin=vmin,
                vmax=vmax,
                rotation=rotation,
                large_mode=large_mode,
                alpha=alpha,
                ax=ax,
            )
        else:
            d.draw_rna(
                seq,
                secstruct,
                c,
                line=line,
                cmap_name=cmap,
                large_mode=large_mode,
                vmin=vmin,
                vmax=vmax,
                rotation=rotation,
                alpha=alpha,
                ax=ax,
            )

    else:
        d.draw_rna(
            seq,
            secstruct,
            seq2col(seq),
            line=line,
            cmap_name=cmap,
            vmin=vmin,
            vmax=vmax,
            large_mode=large_mode,
            rotation=rotation,
            alpha=alpha,
            ax=ax,
        )

    if ax is None:
        plt.show()


def draw_2d_struct_biotite(ax, sequence, sec_struct, draw_pseudoknots=True, bases_to_color=[]):
    """Draw 2D RNA secondary structure with Biotite.

    Adapted from: https://www.biotite-python.org/latest/examples/gallery/structure/nucleotide/transfer_rnas.html

    Args:
        ax (matplotlib.axes.Axes): The axes to draw the secondary structure on.
        sequence (str): The RNA sequence.
        sec_struct (str): The secondary structure in dot-bracket notation.
        draw_pseudoknots (bool): Whether to draw pseudoknots.
        bases_to_color (list): The indices of bases to highlight with colors.
    """
    # Prepare the sequence
    base_labels = list(sequence)
    base_text = []
    base_box = []
    for i, base in enumerate(base_labels):
        if i in bases_to_color:
            base_box.append(
                {
                    "pad": 0,
                    "edgecolor": {
                        "C": "#2e8b57",
                        "A": "#ff8c00",
                        "U": "#5050ff",
                        "G": "#e00000",
                    }[base],
                    "facecolor": {
                        "C": "#2e8b5780",
                        "A": "#ff8c0080",
                        "U": "#5050ff80",
                        "G": "#e0000080",
                    }[base],
                    "boxstyle": "circle",
                }
            )
            base_text.append({"size": "small"})
        else:
            base_box.append({"pad": 0, "color": "white"})
            base_text.append({"size": "small", "color": "grey"})

    # Compute the base pairs and their pseudoknot order
    base_pairs = base_pairs_from_dot_bracket(sec_struct)
    pseudoknot_order = pseudoknots(base_pairs)[0]

    # Set the linestyle according to the pseudoknot order
    linestyles = np.full(base_pairs.shape[0], "-", dtype=object)
    linestyles[pseudoknot_order == 1] = "--"
    linestyles[pseudoknot_order >= 2] = ":"

    # Color canonical Watson-Crick base pairs with a grey and
    # non-canonical base pairs with a lighter orange
    colors = np.full(base_pairs.shape[0], "grey")  # biotite.colors["brightorange"]
    for i, (base1, base2) in enumerate(base_pairs):
        name1 = base_labels[base1]
        name2 = base_labels[base2]
        if sorted([name1, name2]) in [["A", "U"], ["C", "G"]]:
            colors[i] = biotite.colors["dimorange"]

    # Plot the secondary structure
    graphics.plot_nucleotide_secondary_structure(
        ax,
        base_labels,
        base_pairs,
        len(base_labels),
        base_text=base_text,
        base_box=base_box,
        layout_type=1,  # https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/man/RNAplot.html#cmdoption-t
        draw_pseudoknots=draw_pseudoknots,
        pseudoknot_order=pseudoknot_order,
        bond_linestyle=linestyles,
        bond_color=colors,
        annotation_positions=list(range(4, len(base_labels), 5)),
        annotation_text={"size": "small", "color": "grey"},
        border=0,  # Add margin to compensate for reduced axis limits in shared axis
    )
