from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Dynamic loader for gRNAde library living under this directory as ./src
# This avoids clashing with ODesign's own top-level `src` package.
# -----------------------------------------------------------------------------

_GRNADE_ALIAS = "_grnade_src"


def _rewrite_imports(source: str, alias: str) -> str:
    source = source.replace("from src.", f"from {alias}.")
    source = source.replace("import src.", f"import {alias}.")
    source = source.replace("from src import", f"from {alias} import")
    source = source.replace("import src as", f"import {alias} as")
    return source


def _ensure_package(name: str, path: Optional[Path] = None) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__package__ = name
    if path is not None:
        mod.__path__ = [str(path)]
        mod.__file__ = str(path / "__init__.py")
    sys.modules[name] = mod
    return mod


def _load_module(alias_name: str, file_path: Path, package_name: str) -> types.ModuleType:
    if alias_name in sys.modules:
        return sys.modules[alias_name]
    module = types.ModuleType(alias_name)
    module.__file__ = str(file_path)
    module.__package__ = package_name
    sys.modules[alias_name] = module
    source = file_path.read_text(encoding="utf-8")
    source = _rewrite_imports(source, _GRNADE_ALIAS)
    code = compile(source, str(file_path), "exec")
    exec(code, module.__dict__)
    return module


def _load_grnade_runtime(grnade_root: Path):
    src_root = grnade_root / "src"
    if not src_root.exists():
        raise FileNotFoundError(
            f"gRNAde source root not found: {src_root}. "
            f"Expected grnade/src next to wrapper.py."
        )

    if f"{_GRNADE_ALIAS}.models" in sys.modules and f"{_GRNADE_ALIAS}.data.featurizer" in sys.modules:
        models_mod = sys.modules[f"{_GRNADE_ALIAS}.models"]
        feat_mod = sys.modules[f"{_GRNADE_ALIAS}.data.featurizer"]
        return models_mod.gRNAde, feat_mod.RNAGraphFeaturizer

    # create alias package skeleton
    _ensure_package(_GRNADE_ALIAS, src_root)
    _ensure_package(f"{_GRNADE_ALIAS}.data", src_root / "data")

    # load dependencies in order
    _load_module(f"{_GRNADE_ALIAS}.constants", src_root / "constants.py", _GRNADE_ALIAS)
    _load_module(f"{_GRNADE_ALIAS}.layers", src_root / "layers.py", _GRNADE_ALIAS)
    _load_module(f"{_GRNADE_ALIAS}.data.data_utils", src_root / "data" / "data_utils.py", f"{_GRNADE_ALIAS}.data")
    _load_module(f"{_GRNADE_ALIAS}.data.sec_struct_utils", src_root / "data" / "sec_struct_utils.py", f"{_GRNADE_ALIAS}.data")
    feat_mod = _load_module(f"{_GRNADE_ALIAS}.data.featurizer", src_root / "data" / "featurizer.py", f"{_GRNADE_ALIAS}.data")
    models_mod = _load_module(f"{_GRNADE_ALIAS}.models", src_root / "models.py", _GRNADE_ALIAS)
    return models_mod.gRNAde, feat_mod.RNAGraphFeaturizer


# -----------------------------------------------------------------------------
# Wrapper
# -----------------------------------------------------------------------------


_RNA_ALPHABET = "AGCU"
_RNA_TO_ID = {b: i for i, b in enumerate(_RNA_ALPHABET)}


@dataclass
class GRNAdeConfig:
    ckpt_path: str
    grnade_root: Optional[str] = None
    device: Optional[str] = None
    # Model defaults from uploaded models.py
    node_in_dim: Tuple[int, int] = (15, 4)
    node_h_dim: Tuple[int, int] = (128, 16)
    edge_in_dim: Tuple[int, int] = (132, 3)
    edge_h_dim: Tuple[int, int] = (64, 4)
    num_layers: int = 4
    drop_rate: float = 0.5
    out_dim: int = 4
    # Featurizer defaults from uploaded featurizer.py
    radius: float = 4.5
    top_k: int = 32
    num_rbf: int = 32
    num_posenc: int = 32
    max_num_conformers: int = 1
    noise_scale: float = 0.0
    drop_prob_3d: float = 0.0


class GRNAdeWrapper(torch.nn.Module):
    def __init__(self, config: GRNAdeConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        self.device = torch.device(device) if device is not None else torch.device(
            config.device if config.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.grnade_root = Path(config.grnade_root) if config.grnade_root is not None else Path(__file__).resolve().parent
        gRNAde, RNAGraphFeaturizer = _load_grnade_runtime(self.grnade_root)

        # self.featurizer = RNAGraphFeaturizer(
        #     split="test",
        #     radius=config.radius,
        #     top_k=config.top_k,
        #     num_rbf=config.num_rbf,
        #     num_posenc=config.num_posenc,
        #     max_num_conformers=config.max_num_conformers,
        #     noise_scale=config.noise_scale,
        #     drop_prob_3d=config.drop_prob_3d,
        #     device=str(self.device),
        # )

        self.featurizer = RNAGraphFeaturizer(
            split="test",
            radius=config.radius,
            top_k=config.top_k,
            num_rbf=config.num_rbf,
            num_posenc=config.num_posenc,
            max_num_conformers=config.max_num_conformers,
            noise_scale=config.noise_scale,
            drop_prob_3d=config.drop_prob_3d,
            device="cpu",
        )
        
        self.model = gRNAde(
            node_in_dim=config.node_in_dim,
            node_h_dim=config.node_h_dim,
            edge_in_dim=config.edge_in_dim,
            edge_h_dim=config.edge_h_dim,
            num_layers=config.num_layers,
            drop_rate=config.drop_rate,
            out_dim=config.out_dim,
        )

        checkpoint = torch.load(config.ckpt_path, map_location=self.device)
        state_dict = self._extract_state_dict(checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state = checkpoint[key]
                    # strip lightning prefix if needed
                    cleaned = {}
                    for k, v in state.items():
                        nk = k[len("model."):] if k.startswith("model.") else k
                        cleaned[nk] = v
                    return cleaned
            # raw state_dict fallback
            if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                return checkpoint
        raise ValueError("Unrecognized gRNAde checkpoint format")

    @staticmethod
    def _validate_sample(sample_input: Dict[str, Any]) -> None:
        required = ["seq", "P", "C4", "N", "design_mask"]
        missing = [k for k in required if k not in sample_input]
        if missing:
            raise KeyError(f"GRNAdeWrapper missing required keys in sample_input: {missing}")
        if str(sample_input.get("type", "rna")).lower() != "rna":
            raise ValueError("GRNAdeWrapper only supports RNA samples")

    @staticmethod
    def _canonicalize_rna_seq(seq: str) -> str:
        out = []
        for ch in str(seq).upper():
            if ch == "T":
                ch = "U"
            if ch not in _RNA_TO_ID:
                ch = "_"
            out.append(ch)
        return "".join(out)

    @staticmethod
    def _build_partial_seq(seq: str, design_mask: np.ndarray) -> str:
        seq = GRNAdeWrapper._canonicalize_rna_seq(seq)
        return "".join("_" if bool(d) else base for base, d in zip(seq, design_mask.tolist()))

    def _create_partial_seq_logit_bias(self, partial_seq: str, model_out_dim: int = 4) -> torch.Tensor:
        # copied from uploaded design.py semantics
        partial_ids = []
        for residue in partial_seq:
            if residue in self.featurizer.letter_to_num:
                partial_ids.append(self.featurizer.letter_to_num[residue])
            else:
                partial_ids.append(len(self.featurizer.letter_to_num.keys()))
        partial_ids = torch.as_tensor(partial_ids, device=self.device, dtype=torch.long)
        logit_bias = F.one_hot(partial_ids, num_classes=model_out_dim + 1).float()
        logit_bias = logit_bias[:, :-1] * 100.0
        return logit_bias

    def _sample_to_raw_rna(self, sample_input: Dict[str, Any]) -> Dict[str, Any]:
        seq = self._canonicalize_rna_seq(sample_input["seq"])
        P = np.asarray(sample_input["P"], dtype=np.float32)
        C4 = np.asarray(sample_input["C4"], dtype=np.float32)
        N = np.asarray(sample_input["N"], dtype=np.float32)
        if not (len(P) == len(C4) == len(N) == len(seq)):
            raise ValueError("RNA sample fields P/C4/N/seq have inconsistent lengths")
        coords = np.stack([P, C4, N], axis=1).astype(np.float32)  # [L,3,3]
        sec_struct = "." * len(seq)
        return {
            "sequence": seq,
            "coords_list": [coords],
            "sec_struct_list": [sec_struct],
        }

    @staticmethod
    def _decode_ids(ids: torch.Tensor, num_to_letter: Dict[int, str]) -> str:
        return "".join(num_to_letter.get(int(i), "N") for i in ids.tolist())

    @staticmethod
    def _sum_selected_log_probs(sampled_ids: torch.Tensor, log_probs: torch.Tensor, design_mask: torch.Tensor) -> torch.Tensor:
        gathered = torch.gather(log_probs, dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze(-1)
        return (gathered * design_mask).sum(dim=-1)

    @torch.no_grad()
    def sample(
        self,
        sample_input: Dict[str, Any],
        topk: int = 5,
        temp: float = 0.2,
        device: Optional[torch.device] = None,
    ) -> Tuple[List[str], List[float], str, torch.Tensor, Dict[str, Any]]:
        self._validate_sample(sample_input)
        dev = torch.device(device) if device is not None else self.device
        if dev != self.device:
            self.device = dev
            self.model.to(dev)
            self.featurizer.device = str(dev)

        raw_rna = self._sample_to_raw_rna(sample_input)
        featurized = self.featurizer(raw_rna).to(dev)

        design_mask = torch.as_tensor(np.asarray(sample_input["design_mask"], dtype=bool), device=dev)
        if design_mask.ndim != 1:
            design_mask = design_mask.reshape(-1)

        partial_seq = self._build_partial_seq(sample_input["seq"], np.asarray(sample_input["design_mask"], dtype=bool))
        logit_bias = self._create_partial_seq_logit_bias(partial_seq, self.model.out_dim)

        sampled_ids, logits = self.model.sample(
            featurized,
            n_samples=int(topk),
            temperature=float(temp),
            logit_bias=logit_bias,
            return_logits=True,
        )

        log_probs = F.log_softmax(logits, dim=-1)
        scores_t = self._sum_selected_log_probs(sampled_ids.long(), log_probs, design_mask.float().unsqueeze(0).expand(sampled_ids.shape[0], -1))

        pred_seqs = [self._decode_ids(row, self.featurizer.num_to_letter) for row in sampled_ids]
        scores = [float(x) for x in scores_t.detach().cpu().tolist()]
        native_seq = self._canonicalize_rna_seq(sample_input["seq"])
        probs_best = torch.softmax(logits[0], dim=-1).detach().cpu()
        meta = {
            "sampled_ids": sampled_ids.detach().cpu(),
            "logits": logits.detach().cpu(),
            "partial_seq": partial_seq,
            "raw_rna": raw_rna,
        }
        return pred_seqs, scores, native_seq, probs_best, meta


def build_default_grnade_wrapper(
    ckpt_path: str,
    device: Optional[torch.device] = None,
    grnade_root: Optional[str] = None,
) -> GRNAdeWrapper:
    cfg = GRNAdeConfig(
        ckpt_path=ckpt_path,
        grnade_root=grnade_root,
    )
    return GRNAdeWrapper(config=cfg, device=device)
