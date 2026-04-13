from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from .wrapper import GRNAdeWrapper


@torch.no_grad()
def grnade_inference(
    model: GRNAdeWrapper,
    sample_input: Dict[str, Any],
    design_modality: str,
    topk: int = 5,
    temp: float = 0.2,
    use_beam: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[list[str], list[float], str, torch.Tensor, Dict[str, Any]]:
    kind = str(design_modality).strip().lower()
    if kind != "rna":
        raise ValueError(f"grnade_inference only supports RNA, got {design_modality!r}")

    if use_beam:
        # gRNAde here is used in native autoregressive stochastic sampling mode.
        pass

    return model.sample(
        sample_input=sample_input,
        topk=topk,
        temp=temp,
        device=device,
    )
