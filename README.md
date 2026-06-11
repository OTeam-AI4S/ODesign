# ODesign Antibody — Windows Native

> **Windows-native fork** of the `ODesign-antibody` branch from [OTeam-AI4S/ODesign](https://github.com/OTeam-AI4S/ODesign/tree/ODesign-antibody). Designs VHH nanobodies and scFv antibodies against specified antigens. Uses a different invfold module than the main branch (no ProteinMPNN/LigandMPNN/grnade) and a separate antibody-specific checkpoint.

This branch reuses the conda environment from the [main branch's Windows fork](../../tree/main) — set that up first, then come back here.

---

- [Setup](#setup)
- [Input JSON Format](#input-json-format)
  - [Example: scFv design](#example-scfv-design)
  - [Parameter reference](#parameter-reference)
  - [Helper Script — identify_cdr.py](#helper-script--identify_cdrpy)
- [Run Inference](#run-inference)
- [Pre-staged Examples](#pre-staged-examples)
- [Differences from upstream antibody branch](#differences-from-upstream-antibody-branch)
- [License](#license)

---

# Setup

## Step 1 — Reuse the main branch's environment

If you haven't installed the main branch first, do that now — see the [main README](../../tree/main#installation). The conda env (`odesign`), CUDA Toolkit 12.1, VS Build Tools, and `requirements.txt` are all shared.

```cmd
conda activate odesign
```

## Step 2 — Download antibody checkpoints

The antibody branch needs different checkpoints than main: `ab.pt` (antibody-specific ODesign weights) and `oinvfold_protein.ckpt`. From Git Bash, inside this folder:

```bash
bash ./ckpt/get_odesign_ckpt.sh
```

This pulls from [The-Institute-for-AI-Molecular-Design/ODesign-AB](https://huggingface.co/The-Institute-for-AI-Molecular-Design/ODesign-AB) on Hugging Face. Re-run is idempotent.

## Step 3 — Reuse CCD data from main

Same `components.v20240608.cif` / `.rdkit_mol.pkl` as main. Either symlink (admin shell) or copy:

```cmd
REM Symlink (run as Administrator):
mklink /D data "..\ODesign\data"

REM Or copy if you can't elevate:
xcopy /E /I "..\ODesign\data" "data"
```

## Step 4 — Optional: install abnumber

Only needed if you want to use `identify_cdr.py` to auto-generate input JSONs from PDB structures. Pre-staged examples don't need it.

```cmd
pip install abnumber
```

# Input JSON Format

The antibody workflow accepts an antibody framework sequence with CDRs masked by hyphens (`-`). The model designs the masked regions while preserving the framework. This is different from the main branch, which uses sequence ranges.

## Example: scFv design

```json
[
    {
        "name": "abtest",
        "antigen": "./examples/antigen/6oq5_tcdb_truncated.pdb",
        "hotspot": "A/538,A/151,A/152,A/148,A/539",
        "chains": [
            {
                "chain_type": "proteinChain",
                "im": "antibody",
                "sequence": "EVQLVESGGGLVQPGGSLRLSCAAS-YIHWVRQAPGKGLEWVARI-TRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSR-WGQGTLVTVSS",
                "length": "6-7,6-7,9-15"
            },
            {
                "chain_type": "proteinChain",
                "im": "antibody",
                "sequence": "DIQMTQSPSSLSASVGDRVTITC-WYQQKPGKAPKLLIY-GVPSRFSGSRSGTDFTLTISSLQPEDFATYYC-FGQGTKVEIK",
                "length": "6-7,6-7,9-15"
            },
            {
                "chain_type": "proteinChain",
                "im": "antigen",
                "sequence": "A/100-550"
            }
        ]
    }
]
```

## Parameter reference

| Parameter | Description |
| :--- | :--- |
| `name` | Name of the sample. |
| `antigen` | Path to the antigen structure file (`.cif` or `.pdb`). |
| `hotspot` | Hotspot residues on the antigen, format `Chain/ResidueIndex` (comma-separated for multiple). |
| `chains` | List of chain definitions (antibody chains + antigen reference). |

**Per-chain configuration:**

- **`im` (Identity Mode):**
  - `antibody` — model reads the sequence from the `sequence` field below.
  - `antigen` — model reads the structure from the top-level `antigen` path.
- **`sequence`:**
  - For **antigen** chains: residue range (e.g., `A/100-550`).
  - For **antibody** chains: the framework sequence with each CDR replaced by a single `-` hyphen.
- **`length`:** *(antibody chains only)* target length range for each masked CDR. Comma-separated, in order of the hyphens. E.g., `6-7,6-7,9-15` means CDR1 length 6–7, CDR2 length 6–7, CDR3 length 9–15. Set to `""` or omit to keep the original length.

## Helper Script — identify_cdr.py

Auto-generates a compatible input JSON from a PDB or CIF structure of an existing antibody:

```cmd
pip install abnumber
python .\scripts\identify_cdr.py path\to\antibody.pdb --output examples\my_input.json --chothia --odesign
```

Uses `abnumber` to identify CDRs (default scheme IMGT; Chothia available with `--chothia`) and emits a JSON with the framework + masked CDRs in the format above.

# Run Inference

Use the pre-built Windows runner:

```cmd
.\run_antibody.bat
```

Edit the `SET` lines at the top of `run_antibody.bat` to control the run:

| Variable | Default | Description |
|---|---|---|
| `input_json_path` | `./examples/nanobody.json` | Path to input JSON. |
| `exp_name` | `ab_test` | Custom label for the output folder. |
| `seeds` | `[42]` | Random seeds. |
| `N_sample` | `5` | Samples per seed. **Drop to 1 for the first run on 8 GB GPUs.** |
| `num_workers` | `0` | Keep at 0 on Windows for first run. |
| `use_msa` | `false` | Antibody design typically doesn't use MSA on the framework chain itself. |

> ⚠️ **PowerShell users:** Run as `.\run_antibody.bat`, not the bare name.

Or invoke `scripts/inference.py` directly with Hydra overrides:

```cmd
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set CUDA_VISIBLE_DEVICES=0

python .\scripts\inference.py ^
    data_root_dir=./data ^
    ckpt_root_dir=./ckpt ^
    exp.input_json_path=./examples/nanobody.json ^
    exp.exp_name=nb_test ^
    exp.seeds=[42] ^
    exp.model.sample_diffusion.N_sample=1 ^
    exp.use_msa=false ^
    exp.num_workers=0
```

Output lands in `outputs\<exp_name>\<timestamp>\<sample_name>\seed_<S>\predictions\<sample>_seed_<S>_bb_<B>_seq_<Q>.cif`.

# Pre-staged Examples

| File | Description | Size |
|---|---|---|
| `examples/nanobody.json` | Single VHH nanobody chain + antigen. **Lightest example — recommended first run.** | smallest |
| `examples/scfv.json` | Heavy + light chain scFv + antigen. | largest |
| `examples/scfv_H.json` | Heavy chain only + antigen. | medium |

All three target the antigen at `examples/antigen/6oq5_tcdb_truncated.pdb`. Edit the `hotspot` field to retarget.

# Differences from upstream antibody branch

This is a Windows-friendly fork. Specifically:

- **Three source patches** (same as main branch): try/except around the OpenFold CUDA softmax kernel import, try/except around DeepSpeed's `DS4Sci_EvoformerAttention` import (CUTLASS not installed natively), and `tempfile.gettempdir()` instead of hardcoded `/tmp` paths in `msa_utils.py`.
- **`run_antibody.bat`** — Windows-native runner (the upstream antibody branch ships no `.sh` runner at all).
- **No code-level changes** beyond the three patches above. Model weights, model architecture, training data, and inference logic are unmodified.

The full changelog and rationale for each patch is in the main branch's [`PROGRESS_NOTES.md`](../../tree/main/PROGRESS_NOTES.md).

# License

Inherits the [Apache 2.0 License](LICENSE) from upstream. Windows-fork patches are released under the same license.
