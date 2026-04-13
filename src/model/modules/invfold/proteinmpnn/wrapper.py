from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .data_utils import featurize
from .model_utils import ProteinMPNN

_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


@dataclass
class MPNNFamilyConfig:
    protein_ckpt_path: str
    ligand_ckpt_path: str
    hidden_dim: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    ligand_mpnn_cutoff_for_score: float = 8.0
    ligand_mpnn_use_atom_context: bool = True
    ligand_mpnn_use_side_chain_context: bool = False


class MPNNFamilyWrapper(torch.nn.Module):
    """
    Unified wrapper for the LigandMPNN family.

    Expected protein sample schema (emitted by parse_invfold after your edits):
      - type == "protein"
      - seq: str                              # local target-chain full sequence
      - design_mask: bool[L_target]
      - chain_ids: np.ndarray[str]            # local target-chain chain ids
      - res_atom_indices: list[np.ndarray]    # local target-chain atom mapping
      - target_global_indices: np.ndarray[int]# indices of the target chain in whole-structure order
      - mpnn_model_type: "protein_mpnn" | "ligand_mpnn"
      - has_ligand_context: bool
      - mpnn_input_dict: dict with whole-structure keys:
            X, mask, R_idx, chain_labels, chain_letters, S
            optional: Y, Y_t, Y_m, xyz_37, xyz_37_m, side_chain_mask
    """

    def __init__(self, config: MPNNFamilyConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.models: Dict[str, ProteinMPNN] = {}
        self.model_meta: Dict[str, Dict[str, Any]] = {}
        self._load_all_models()

    def _load_all_models(self) -> None:
        self.models["protein_mpnn"], self.model_meta["protein_mpnn"] = self._load_model(
            model_type="protein_mpnn",
            ckpt_path=self.config.protein_ckpt_path,
        )
        self.models["ligand_mpnn"], self.model_meta["ligand_mpnn"] = self._load_model(
            model_type="ligand_mpnn",
            ckpt_path=self.config.ligand_ckpt_path,
        )

    def _load_model(self, model_type: str, ckpt_path: str) -> Tuple[ProteinMPNN, Dict[str, Any]]:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{model_type} checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        num_edges = int(checkpoint["num_edges"])
        atom_context_num = int(checkpoint.get("atom_context_num", 1 if model_type != "ligand_mpnn" else 16))

        model = ProteinMPNN(
            node_features=self.config.hidden_dim,
            edge_features=self.config.hidden_dim,
            hidden_dim=self.config.hidden_dim,
            num_encoder_layers=self.config.num_encoder_layers,
            num_decoder_layers=self.config.num_decoder_layers,
            k_neighbors=num_edges,
            device=self.device,
            atom_context_num=atom_context_num,
            model_type=model_type,
            ligand_mpnn_use_side_chain_context=bool(self.config.ligand_mpnn_use_side_chain_context),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        meta = {
            "ckpt_path": ckpt_path,
            "num_edges": num_edges,
            "atom_context_num": atom_context_num,
            "model_type": model_type,
        }
        return model, meta

    def to(self, *args, **kwargs):  # type: ignore[override]
        super().to(*args, **kwargs)
        if args:
            self.device = torch.device(args[0])
        elif "device" in kwargs:
            self.device = torch.device(kwargs["device"])
        for model in self.models.values():
            model.to(*args, **kwargs)
        return self

    def eval(self):  # type: ignore[override]
        super().eval()
        for model in self.models.values():
            model.eval()
        return self

    @staticmethod
    def _to_tensor(x: Any, device: torch.device) -> Any:
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, np.ndarray):
            if x.dtype.kind in ("U", "S", "O"):
                return x
            return torch.from_numpy(x).to(device)
        if isinstance(x, Mapping):
            return {k: MPNNFamilyWrapper._to_tensor(v, device) for k, v in x.items()}
        if isinstance(x, list):
            return [MPNNFamilyWrapper._to_tensor(v, device) for v in x]
        if isinstance(x, tuple):
            return tuple(MPNNFamilyWrapper._to_tensor(v, device) for v in x)
        return x

    @staticmethod
    def _ensure_long_1d(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.long().reshape(-1)
        return torch.as_tensor(x, dtype=torch.long).reshape(-1)

    @staticmethod
    def _ensure_bool_1d(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.bool().reshape(-1)
        return torch.as_tensor(x, dtype=torch.bool).reshape(-1)

    @staticmethod
    def _seq_from_ids(ids: Sequence[int]) -> str:
        return "".join(_ALPHABET[int(i)] if 0 <= int(i) < len(_ALPHABET) else "X" for i in ids)

    @staticmethod
    def _sum_selected_log_probs(sampled_ids: torch.Tensor, log_probs: torch.Tensor, select_mask: torch.Tensor) -> torch.Tensor:
        gathered = torch.gather(log_probs, dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze(-1)
        return (gathered * select_mask).sum(dim=-1)

    @staticmethod
    def _validate_sample(sample_input: Dict[str, Any]) -> None:
        required = [
            "type",
            "seq",
            "design_mask",
            "target_global_indices",
            "mpnn_model_type",
            "mpnn_input_dict",
        ]
        missing = [k for k in required if k not in sample_input]
        if missing:
            raise KeyError(f"MPNNFamilyWrapper missing required keys in sample_input: {missing}")
        if str(sample_input.get("type", "")).lower() != "protein":
            raise ValueError("MPNNFamilyWrapper only supports protein design samples")

    def _build_feature_dict(
        self,
        sample_input: Dict[str, Any],
        topk: int,
        temp: float,
        device: torch.device,
    ) -> Tuple[str, Dict[str, Any], torch.Tensor, torch.Tensor]:
        self._validate_sample(sample_input)

        model_type = str(sample_input["mpnn_model_type"]).strip().lower()
        if model_type not in self.models:
            raise ValueError(f"Unsupported mpnn_model_type: {model_type!r}")

        base_input = copy.deepcopy(sample_input["mpnn_input_dict"])
        base_input = self._to_tensor(base_input, device)

        if "S" not in base_input or "X" not in base_input or "mask" not in base_input:
            raise KeyError("sample_input['mpnn_input_dict'] must at least contain X, mask, and S")

        total_len = int(base_input["S"].shape[0])
        target_global_indices = self._ensure_long_1d(sample_input["target_global_indices"]).to(device)
        if target_global_indices.numel() == 0:
            raise ValueError("target_global_indices is empty")

        local_design_mask = self._ensure_bool_1d(sample_input["design_mask"]).to(device)
        if local_design_mask.numel() != target_global_indices.numel():
            raise ValueError(
                f"design_mask length ({local_design_mask.numel()}) != target_global_indices length ({target_global_indices.numel()})"
            )

        global_chain_mask = torch.zeros(total_len, dtype=torch.float32, device=device)
        global_chain_mask[target_global_indices[local_design_mask]] = 1.0
        base_input["chain_mask"] = global_chain_mask

        meta = self.model_meta[model_type]
        feature_dict = featurize(
            base_input,
            cutoff_for_score=float(self.config.ligand_mpnn_cutoff_for_score),
            use_atom_context=bool(self.config.ligand_mpnn_use_atom_context),
            number_of_ligand_atoms=int(meta["atom_context_num"]),
            model_type=model_type,
        )

        L = int(feature_dict["X"].shape[1])
        feature_dict["batch_size"] = int(topk)
        feature_dict["temperature"] = float(temp)
        feature_dict["bias"] = torch.zeros((1, L, 21), dtype=torch.float32, device=device)
        feature_dict["symmetry_residues"] = [[]]
        feature_dict["symmetry_weights"] = [[]]
        feature_dict["randn"] = torch.randn((int(topk), L), dtype=torch.float32, device=device)

        return model_type, feature_dict, target_global_indices, local_design_mask

    @torch.no_grad()
    def sample(
        self,
        sample_input: Dict[str, Any],
        topk: int = 5,
        temp: float = 0.2,
        device: Optional[torch.device] = None,
    ) -> Tuple[List[str], List[float], str, torch.Tensor, Dict[str, Any]]:
        dev = torch.device(device) if device is not None else self.device
        model_type, feature_dict, target_global_indices, local_design_mask = self._build_feature_dict(
            sample_input=sample_input,
            topk=topk,
            temp=temp,
            device=dev,
        )

        model = self.models[model_type]
        meta_model = self.model_meta[model_type]
        output_dict = model.sample(feature_dict)

        S_sample = output_dict["S"]
        log_probs = output_dict["log_probs"]
        sampling_probs = output_dict.get("sampling_probs", torch.softmax(log_probs, dim=-1))
        native_seq = str(sample_input["seq"])

        target_ids = target_global_indices.long()
        design_global_mask = torch.zeros(log_probs.shape[1], dtype=torch.float32, device=dev)
        design_global_mask[target_ids[local_design_mask]] = 1.0
        design_global_mask = design_global_mask.unsqueeze(0).repeat(S_sample.shape[0], 1)

        if model_type == "ligand_mpnn" and "mask_XY" in feature_dict:
            score_mask = design_global_mask * feature_dict["mask_XY"].float().repeat(S_sample.shape[0], 1)
            if float(score_mask.sum()) == 0.0:
                score_mask = design_global_mask
        else:
            score_mask = design_global_mask

        logp_sums = self._sum_selected_log_probs(S_sample, log_probs, score_mask)

        pred_seqs: List[str] = []
        scores: List[float] = []
        meta_candidates: List[Dict[str, Any]] = []
        probs_local_all: List[torch.Tensor] = []

        for b_ix in range(S_sample.shape[0]):
            seq_ids_all = S_sample[b_ix]
            seq_ids_target = seq_ids_all[target_ids].detach().cpu().tolist()
            seq_target = self._seq_from_ids(seq_ids_target)
            score = float(logp_sums[b_ix].detach().cpu().item())
            pred_seqs.append(seq_target)
            scores.append(score)
            probs_local_all.append(sampling_probs[b_ix, target_ids].detach().cpu())
            meta_candidates.append(
                {
                    "seq": seq_target,
                    "score": score,
                    "target_length": int(target_ids.numel()),
                    "model_type": model_type,
                }
            )

        order = np.argsort(np.asarray(scores))[::-1].tolist()
        pred_seqs = [pred_seqs[i] for i in order]
        scores = [scores[i] for i in order]
        meta_candidates = [meta_candidates[i] for i in order]
        probs_best = probs_local_all[order[0]] if len(order) > 0 else torch.empty((0, 21), dtype=torch.float32)

        meta = {
            "candidates": meta_candidates,
            "native_seq": native_seq,
            "model_type": model_type,
            "has_ligand_context": bool(sample_input.get("has_ligand_context", False)),
            "target_global_indices": target_ids.detach().cpu(),
            "local_design_mask": local_design_mask.detach().cpu(),
            "ckpt_path": meta_model["ckpt_path"],
            "num_edges": int(meta_model["num_edges"]),
            "atom_context_num": int(meta_model["atom_context_num"]),
            "used_atom_context": bool(model_type == "ligand_mpnn" and self.config.ligand_mpnn_use_atom_context),
        }
        return pred_seqs, scores, native_seq, probs_best, meta


def build_default_wrapper(
    protein_ckpt_path: Optional[str] = None,
    ligand_ckpt_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    ligand_mpnn_cutoff_for_score: float = 8.0,
    ligand_mpnn_use_atom_context: bool = True,
    ligand_mpnn_use_side_chain_context: bool = False,
) -> MPNNFamilyWrapper:
    ckpt_root = os.getenv("CKPT_ROOT_DIR", "")
    if protein_ckpt_path is None:
        protein_ckpt_path = os.path.join(ckpt_root, "v_48_020.pt")
    if ligand_ckpt_path is None:
        ligand_ckpt_path = os.path.join(ckpt_root, "ligandmpnn_v_32_010_25.pt")

    cfg = MPNNFamilyConfig(
        protein_ckpt_path=protein_ckpt_path,
        ligand_ckpt_path=ligand_ckpt_path,
        ligand_mpnn_cutoff_for_score=float(ligand_mpnn_cutoff_for_score),
        ligand_mpnn_use_atom_context=bool(ligand_mpnn_use_atom_context),
        ligand_mpnn_use_side_chain_context=bool(ligand_mpnn_use_side_chain_context),
    )
    return MPNNFamilyWrapper(config=cfg, device=device)
