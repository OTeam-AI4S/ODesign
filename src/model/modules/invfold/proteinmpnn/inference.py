from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from .wrapper import MPNNFamilyWrapper


@torch.no_grad()
def proteinmpnn_inference(
    model: MPNNFamilyWrapper,
    sample_input: Dict[str, Any],
    design_modality: str,
    topk: int = 5,
    temp: float = 0.2,
    use_beam: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[list[str], list[float], str, torch.Tensor, Dict[str, Any]]:
    """
    Unified protein-design entrypoint for the MPNN family.

    Notes
    -----
    - Only valid when ODesign `design_modality == "protein"`.
    - The concrete backend is selected *per sample* using
      `sample_input["mpnn_model_type"]`:
        * "protein_mpnn"
        * "ligand_mpnn"
    - `use_beam` is kept only for call-site compatibility and is ignored.
    """
    kind = str(design_modality).strip().lower()
    if kind != "protein":
        raise ValueError(f"proteinmpnn_inference only supports protein design, got {design_modality!r}")

    _ = use_beam
    return model.sample(
        sample_input=sample_input,
        topk=topk,
        temp=temp,
        device=device,
    )
