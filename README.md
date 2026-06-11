<div align="center">
  <img src="imgs/odesign_logo.svg" alt="ODesign" width="60%">
</div>

<div align="center">

[![Web Server](https://img.shields.io/badge/Web_Server-Access-9A8CF0?logo=internet-explorer)](https://odesign.lglab.ac.cn/)
[![Technical Report](https://img.shields.io/badge/Technical_Report-Download-9A8CF0?logo=adobe-acrobat-reader)](https://odesign1.github.io/static/pdfs/technical_report.pdf)
[![Project Page](https://img.shields.io/badge/Project_Page-Access-9A8CF0?logo=adobe-acrobat-reader)](https://odesign1.github.io/)
[![Windows Native](https://img.shields.io/badge/Windows-Native-0078D6?logo=windows)](#installation)
[![Upstream](https://img.shields.io/badge/Forked_from-OTeam--AI4S/ODesign-blue?logo=github)](https://github.com/OTeam-AI4S/ODesign)
</div>

> # 🪟 Windows-native fork
>
> This repository is a **Windows-native fork** of [OTeam-AI4S/ODesign](https://github.com/OTeam-AI4S/ODesign). The upstream targets Linux + Docker; this fork is for Windows users who need to use WDDM shared GPU memory to spill past their dedicated VRAM cap (e.g. running 4 GB+ checkpoints on an 8 GB consumer GPU). Tested end-to-end on **Windows 11 + RTX 4070 (8 GB dedicated + 8 GB shared) + CUDA 12.1**.
>
> If you're on Linux or you have a Linux-friendly Docker setup, the upstream repo is a smoother experience. Use this fork if Windows is a hard constraint.

---

- [About this fork](#about-this-fork)
- [Installation](#installation)
- [Available Models](#available-models)
- [Inference](#inference)
  - [Input Format](#input-format)
  - [Run Inference](#run-inference)
  - [Run All Examples (smoke test)](#run-all-examples-smoke-test)
  - [Output Format](#output-format)
  - [Usage](#usage)
    - [Protein Generation](#protein-generation)
    - [Ligand Generation](#ligand-generation)
    - [Nucleic Acid Generation](#nucleic-acid-generation)
    - [Cyclic Peptide Generation](#cyclic-peptide-generation)
    - [Partial Diffusion](#partial-diffusion)
- [MSA Mode on Windows (Limited)](#msa-mode-on-windows-limited)
- [Training](#training)
- [Linux / Docker / Apptainer Users](#linux--docker--apptainer-users)
- [Cite](#cite)
- [Acknowledgements](#acknowledgements)
- [License](#license)

🎉 [ODesign](https://odesign1.github.io/static/pdfs/technical_report.pdf) is an all-atom generative world model for all-to-all biomolecular interaction design. ODesign allows scientists to specify epitopes on arbitrary targets and generate diverse classes of binding partners with fine-grained control.

A no-install hosted version is available at https://odesign.lglab.ac.cn — use that if you only need occasional inference and don't want to manage a local install.

For questions about the model, contact the upstream authors at [odesign@lglab.ac.cn](mailto:odesign@lglab.ac.cn). For questions about this Windows fork, open an issue here.

<div align="center">
  <img src="imgs/odesign_video.gif" alt="ODesign Video" width="100%">
</div>


# About this fork

What's different from upstream:

| Component | Upstream | This fork | Reason |
|---|---|---|---|
| `requirements.txt` — `triton==2.3.1` | required | commented out | no Windows wheel; not imported by `src/` |
| `requirements.txt` — `pyg-lib` | required | commented out | no Windows wheel; not imported by `src/` |
| `requirements.txt` — `biotite` | `1.0.1` | `1.2.0` | `inference_utils.py` imports `biotite.interface.rdkit` (added in 1.1+) |
| `requirements.txt` — `setuptools` | `75.8.2` | `69.5.1` | `pytorch_lightning==1.9.0` needs `pkg_resources.declare_namespace`, removed in setuptools 70+ |
| `requirements.txt` — `pyparsing` | `3.2.1` | `3.1.1` | `prody` requires `<=3.1.1` |
| `requirements.txt` — `prody`, `addict` | missing | added | imported by `invfold/` modules but absent from upstream pin list |
| `ckpt/get_odesign_ckpt.sh` | 9 URLs | 10 URLs | added LigandMPNN checkpoint hardcoded by `infer_runner.py:89` |
| `inference_demo.sh` `num_workers` | `4` | `0` | Windows uses spawn-based DataLoaders; pickle issues common at >0 |
| `inference_demo.sh` allocator env | not set | `expandable_segments:True` | reduces fragmentation when spilling into shared GPU memory |
| `src/utils/openfold_local/utils/kernel/attention_core.py` | hard import | try/except | OpenFold's optional CUDA softmax kernel isn't built natively |
| `src/utils/openfold_local/model/primitives.py` | hard import | try/except | DeepSpeed `DS4Sci_EvoformerAttention` JIT-compiles on first import; needs CUTLASS, not installed natively |
| `src/utils/data/msa_utils.py` | hardcoded `/tmp/...` | `tempfile.gettempdir()` | defensive, in case the `precomputed_msa_dir` path isn't taken |
| `inference_demo.bat` | absent | added | Windows-native runner equivalent to `inference_demo.sh` |
| `run_all_examples.bat` | absent | added | sanity-check runner across all 11 pre-staged examples |

For full install state, supplemental fixes during first inference, and known-issue triage, see:

- [`ODESIGN_INSTALL_CHECKLIST.md`](ODESIGN_INSTALL_CHECKLIST.md) — phase-by-phase install with checkboxes
- [`PROGRESS_NOTES.md`](PROGRESS_NOTES.md) — what was patched and why, with verification commands

# Installation

The full step-by-step is in [`ODESIGN_INSTALL_CHECKLIST.md`](ODESIGN_INSTALL_CHECKLIST.md). The summary below assumes you're starting fresh.

### Step 1 — Windows tooling (one-time)

Install in this order. Run in an **elevated terminal** (Run as Administrator) for CUDA Toolkit and VS Build Tools.

| Tool | Version | Notes |
|---|---|---|
| **NVIDIA Driver** | ≥ 550 | Any recent driver supports WDDM shared memory. |
| **CUDA Toolkit** | **12.1.1** (not 12.2/12.4) | Must match PyTorch's pinned wheel. [Download](https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_531.14_windows.exe) |
| **Visual Studio 2022 Build Tools** | latest | Pick the *Desktop development with C++* workload. Required for any source-build wheel. |
| **Miniconda** | any recent | [Miniconda3-py310](https://repo.anaconda.com/miniconda/Miniconda3-py310_latest-Windows-x86_64.exe) |
| **Git for Windows** | any recent | Provides Git Bash, which runs the project's `.sh` scripts unmodified. |

Verify in a fresh `cmd.exe`:

```cmd
nvcc --version              :: should show release 12.1
nvidia-smi                  :: should show your GPU + driver
conda --version             :: any version
git --version               :: any version
```

### Step 2 — Clone

```cmd
cd /d "C:\path\to\where\you\want\it"
git clone https://github.com/<your-username>/ODesign.git
cd ODesign
```

### Quick path: one-shot installer

This fork ships `install.bat`, which runs Steps 3–6 below in the right order with the right env vars (notably `DS_BUILD_OPS=0` etc. so DeepSpeed installs in JIT mode on Windows). From the **x64 Native Tools Command Prompt for VS 2022** with the conda base env active:

```cmd
.\install.bat
```

The remaining manual step is the CCD data download (Step 6 below); the script prints the exact `gdown` command at the end. If you'd rather do it by hand, follow Steps 3–6 below explicitly.

### Step 3 — Create the conda environment

Open the **"x64 Native Tools Command Prompt for VS 2022"** (Start menu → Visual Studio 2022 folder). Use this shell — it puts MSVC and CUDA on PATH for source-build wheels.

```cmd
conda create -n odesign python=3.10 -y
conda activate odesign

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

Sanity check (must print `2.3.1+cu121 True 12.1`):

```cmd
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

### Step 4 — Install the rest of the requirements

```cmd
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip install gdown
```

If `deepspeed==0.17.4` fails to install, fall back to `0.15.4` or `0.14.0` — any version that *imports* on Windows is enough since the JIT-compiled ops are guarded by try/except in this fork.

If `torch_scatter`/`torch_sparse`/`torch_cluster`/`torch_spline_conv` fall back to source build, that's expected; the x64 Native Tools shell has the right env. Slow (~5-10 min each) but reliable.

### Step 5 — Download checkpoints

From **Git Bash**:

```bash
bash ./ckpt/get_odesign_ckpt.sh
```

This pulls all 10 required checkpoints (~17 GB total, including the LigandMPNN checkpoint added in this fork). Re-run is idempotent.

### Step 6 — Download CCD data

```cmd
mkdir data
gdown --folder https://drive.google.com/drive/folders/1wPmwIrC3G52q1JFY0RXY95tjKDl7YEln -O data
```

You need `components.v20240608.cif` and `components.v20240608.cif.rdkit_mol.pkl` to land in `data\`. The other file in that Drive folder (`odesign_full_data.tar.gz`, ~850 GB unzipped) is the **training** dataset and is NOT needed for inference.

# Available Models

ODesign provides four pre-trained model variants. Each model supports a specific modality and design mode:

| Model Name                  | Design Modality | Design Mode          | Hugging Face                                                                         |
| --------------------------- | ----------------- | ---------------------- | ------------------------------------------------------------------------------------ |
| `odesign_base_prot_flex`    | protein           | flexible-receptor | [odesign_base_prot_flex.pt](https://huggingface.co/The-Institute-for-AI-Molecular-Design/ODesign/resolve/main/ckpt/odesign_base_prot_flex.pt?download=true) |
| `odesign_base_prot_rigid`   | protein           | rigid-receptor    | [odesign_base_prot_rigid.pt](https://huggingface.co/The-Institute-for-AI-Molecular-Design/ODesign/resolve/main/ckpt/odesign_base_prot_rigid.pt?download=true) |
| `odesign_base_ligand_rigid` | ligand            | rigid-receptor    | [odesign_base_ligand_rigid.pt](https://huggingface.co/The-Institute-for-AI-Molecular-Design/ODesign/resolve/main/ckpt/odesign_base_ligand_rigid.pt?download=true) |
| `odesign_base_na_rigid`     | nucleic acid      | rigid-receptor    | [odesign_base_na_rigid.pt](https://huggingface.co/The-Institute-for-AI-Molecular-Design/ODesign/resolve/main/ckpt/odesign_base_na_rigid.pt?download=true) |

<div align="center">
  <img src="imgs/odesign_design_mode.jpg" alt="ODesign Design Mode" width="85%">
</div>

OInvFold checkpoints for different modalities are at [Hugging Face](https://huggingface.co/The-Institute-for-AI-Molecular-Design/OInvFold/tree/main).

# Inference

## Input Format

See **Section B.1 & B.2** of the [Supplementary Information](https://odesign1.github.io/static/pdfs/technical_report.pdf) for details. Example input JSONs for each task are in the [`examples`](examples/) directory.

A `ligand` chain can be specified by SMILES via the `smiles` field instead of a `ref_file`. See [`examples/protein_design/lig_binding_prot_smiles/odesign_input.json`](examples/protein_design/lig_binding_prot_smiles/odesign_input.json). This works out of the box on this fork (biotite 1.2.0 is in `requirements.txt`).

## Run Inference

After Steps 1–6 of [Installation](#installation), launch inference with the Windows runner:

```cmd
.\inference_demo.bat
```

(or, equivalently from Git Bash: `bash inference_demo.sh`)

Edit the `SET` lines at the top of `inference_demo.bat` to control the run:

| Argument              | Description                                                                                                                                             | Example                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| `infer_model_name`     | Model. Options: `odesign_base_prot_flex`, `odesign_base_prot_rigid`, `odesign_base_ligand_rigid`, `odesign_base_na_rigid`. | `odesign_base_prot_flex`            |
| `design_modality`        | Required for nucleic-acid models. Options: `protein`, `ligand`, `dna`, `rna`. | `rna` |
| `data_root_dir`        | Where the CCD data lives. | `./data` |
| `ckpt_root_dir`        | Where the checkpoints live. | `./ckpt` |
| `input_json_path`      | Path to the input design JSON. | `./examples/.../odesign_input.json` |
| `exp_name`             | Custom label for the output folder. Auto-generated if empty. | `protein_binding_protein_design` |
| `seeds`                | Random seeds. Multiple supported. | `[42]` or `[42, 123]` |
| `N_sample`             | Samples per seed. **Start at `1` for first run on 8 GB GPUs.** | `5` |
| `use_msa`              | Use MSA. Only `true` if input JSON has `msa.precomputed_msa_dir` (see [MSA Mode](#msa-mode-on-windows-limited)). | `false` |
| `num_workers`          | DataLoader workers. **Keep at 0 on Windows for first run.** | `0` |
| `CUDA_VISIBLE_DEVICES` | GPU index. | `0` |

> ⚠️ **PowerShell users:** `inference_demo.bat` won't run as a bare command. Use `.\inference_demo.bat`. Or switch to `cmd.exe`.
>
> ⚠️ **Single-GPU only.** NCCL (PyTorch's distributed backend used when `world_size > 1`) is Linux-only. The commented-out `torchrun` multi-GPU block at the bottom of `inference_demo.sh` will not work on Windows — leave it commented. The single-GPU path is the only one this fork supports.

## Run All Examples (smoke test)

This fork ships `run_all_examples.bat`, which walks all 11 pre-staged examples with `N_sample=1`, dumps each to its own output folder under `outputs\test_*\`, and prints a pass/fail summary at the end. Useful as an end-to-end stack check after a fresh install.

```cmd
.\run_all_examples.bat
```

Wall-clock ~30 minutes total (~2-3 min per example) on an RTX 4070.

## Output Format

When inference completes, results are saved under `outputs\<exp_name>\<timestamp>\`:

```
outputs
└── <exp_name>
    └── <timestamp>
        ├── .hydra
        ├── errors
        ├── <sample_name_1>
        │   ├── seed_XXX
        │   │   ├── predictions
        │   │   │   ├── <sample_name_1>_seed_XXX_bb_0_seq_0.cif
        │   │   │   ├── <sample_name_1>_seed_XXX_bb_0_seq_1.cif
        │   │   │   └── ...
        │   │   └── traceback.pkl
        │   ├── seed_YYY
        │   │   └── ...
        ├── <sample_name_2>
        │   └── ...
        └── run.log
```

| Folder / File   | Description                                                                                                                   |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `<exp_name>/`    | Folder corresponding to your `exp_name`.                                |
| `<timestamp>/`   | Auto-generated run folder. |
| `.hydra/`        | Hydra config snapshot — useful for reproducibility. |
| `errors/`        | Error logs (empty on a clean run). |
| `<sample_name>/` | Named after the `name` field in the input JSON. |
| `seed_<value>/`   | Outputs for a specific random seed. |
| `predictions/`   | Inverse-folded molecular designs. |
| `*.cif`         | Designed molecules in CIF format. Open in PyMOL / ChimeraX / [Mol\* viewer](https://molstar.org/viewer/). |
| `traceback.pkl` | Serialized atom-array traceback. |
| `run.log`       | Full inference log. |

The naming pattern is `<name>_seed_<S>_bb_<B>_seq_<Q>.cif` — backbone B, sequence Q. Sequences per backbone defaults to 1; override via `exp.invfold_topk`.

## Usage

### Protein Generation

#### Protein-binding Protein

<img src="imgs/protein/protein_binding_protein.png" alt="Protein-binding Protein" width="400px" align="right"/>

ODesign generates proteins that bind specific protein targets. Provide a reference structure of the target and specify hotspot residues that define the binding interface. The model generates a new protein chain that interacts at the specified hotspot.

Edit `inference_demo.bat`:
```
SET infer_model_name=odesign_base_prot_flex
SET input_json_path=./examples/protein_design/prot_binding_prot/odesign_input.json
```
Then `.\inference_demo.bat`.

#### Ligand-binding Protein

<img src="imgs/protein/ligand_binding_protein.png" alt="Ligand-binding Protein" width="400px" align="right"/>

Generates proteins that bind small-molecule ligands. Provide a reference with the ligand; specify hotspot atoms.

Edit:
```
SET infer_model_name=odesign_base_prot_flex
SET input_json_path=./examples/protein_design/lig_binding_prot/odesign_input.json
```

#### Atom Scaffold

<img src="imgs/protein/tip_atom.png" alt="Atom Scaffold" width="150" align="right"/>

Scaffold proteins around specific atoms or functional groups. Useful for designing proteins around specific chemical moieties.

Edit:
```
SET infer_model_name=odesign_base_prot_flex
SET input_json_path=./examples/protein_design/atom_scaffold/odesign_input.json
```

#### Motif Scaffold
<p align="center">
<img src="imgs/protein/motif_scaffolding.png" alt="Motif Scaffold" width="400px" align="middle"/>
</p>

Scaffold functional motifs by generating surrounding protein structure. Useful for stabilizing motifs or building new folds around known functional elements.

Edit:
```
SET infer_model_name=odesign_base_prot_flex
SET input_json_path=./examples/protein_design/motif_scaffold/odesign_input.json
```

### Ligand Generation

#### Protein-binding Ligand

<img src="imgs/ligand/protein_binding_ligand.png" alt="Protein-binding Ligand" width="150" align="right"/>

Generates small-molecule ligands that bind specific protein targets. Provide a reference with the protein; specify hotspot residues.

Edit:
```
SET infer_model_name=odesign_base_ligand_rigid
SET design_modality=ligand
SET input_json_path=./examples/ligand_design/prot_binding_lig/odesign_input.json
```

### Nucleic Acid Generation

#### Backbone Generation
<p align="center">
<img src="imgs/na/rna_backbone.png" alt="RNA Backbone" width="300px" align="middle"/>
</p>

Generate nucleic-acid backbones of specified length. To switch to DNA, change `chain_type` in the JSON to `"dnaChain"` and set `design_modality=dna`.

Edit:
```
SET infer_model_name=odesign_base_na_rigid
SET design_modality=rna
SET input_json_path=./examples/na_design/rna_bb/odesign_input.json
```

#### Protein-binding Nucleic Acid
<p align="center">
<img src="imgs/na/protein_binding_rna.png" alt="Protein-binding RNA" width="400px" align="middle"/>
</p>

Generate NA molecules that bind protein targets. Provide a reference with the protein; specify hotspot residues. To switch to DNA, change `chain_type` in the JSON and set `design_modality=dna`.

Edit:
```
SET infer_model_name=odesign_base_na_rigid
SET design_modality=rna
SET input_json_path=./examples/na_design/prot_binding_rna/odesign_input.json
```

### Cyclic Peptide Generation

#### Protein-binding Cyclic Peptide

Generate cyclic peptides that bind protein targets. Provide a reference with the target; specify hotspot residues.

Edit:
```
SET infer_model_name=odesign_base_prot_flex
SET input_json_path=./examples/cyclic_peptide_design/odesign_input.json
```

### Partial Diffusion

ODesign can partially modify existing binding molecules to enhance stability, modulate specificity, or improve expressibility. Provide a reference with the target molecule and specify the `partial_diff` field in the input JSON to indicate regions to modify. See **Section B.3 Partial Diffusion** of the [Supplementary Information](https://odesign1.github.io/static/pdfs/technical_report.pdf).

Edit:
```
SET infer_model_name=odesign_base_prot_rigid
SET input_json_path=./examples/protein_design/prot_binding_prot_partial_diff/odesign_input.json
SET enable_partial_diff=true
```

# MSA Mode on Windows (Limited)

Setting `use_msa=true` triggers ODesign's MSA pipeline, which on Linux calls `jackhmmer` (HMMER), `reformat.pl` (HHsuite), and `subprocess.check_call(..., executable="/bin/bash")`. None of those exist on native Windows. Trying to port them is a multi-day rabbit hole.

**The supported workflow on Windows: pre-compute MSAs externally and inject via `precomputed_msa_dir`.** ODesign's `msa_featurizer.py` checks for `msa.precomputed_msa_dir` in the input JSON *before* invoking the broken pipeline; if present, the entire jackhmmer/HHsuite path is skipped and the model just reads two `.a3m` files.

**Required files** in `<msa_dir>/`:
- `non_pairing.a3m` (always required)
- `pairing.a3m` (only required when the chain is *not* a homomer/monomer)

**Three options for producing the `.a3m` files**, ranked by ease:

1. **ColabFold MMseqs2 web API** (easiest — free, web-based). Upload sequence, get back a `.a3m`, rename to ODesign's expected names.
2. **Native MMseqs2 on Windows** (binaries available; needs a local UniRef30 / ColabFold-Env DB).
3. **Run jackhmmer once under WSL** and copy the output back.

Then add to your input JSON:

```json
{
  "chain_type": "proteinChain",
  "sequence": "MVKVGVNG...",
  "msa": {
    "precomputed_msa_dir": "./data/msa/chain1",
    "pairing_db": "uniprot"
  }
}
```

Set `use_msa=true` in `inference_demo.bat`.

For full Phase I instructions including a sample ColabFold API client script, see [`ODESIGN_INSTALL_CHECKLIST.md`](ODESIGN_INSTALL_CHECKLIST.md#phase-i--flipping-msa-on-the-proper-fix).

# Training

Training requires the upstream ~850 GB training dataset (`odesign_full_data.tar.gz` from [Google Drive](https://drive.google.com/drive/folders/1wPmwIrC3G52q1JFY0RXY95tjKDl7YEln)) and is best done on Linux at the moment. **Native-Windows training is not validated by this fork** — at minimum you'll need:

- WDDM-compatible spilling won't work for a typical training crop (640 tokens) on a consumer GPU; expect to need an A100/H100-class accelerator.
- Pre-computed MSAs for the entire training corpus, since the on-the-fly MSA search is broken on Windows (see [MSA Mode](#msa-mode-on-windows-limited)).

If you have those constraints solved, `train_demo.sh` works under Git Bash. Set `ckpt_root_dir` to a folder containing the pre-trained folding model checkpoint ([protenix_base_default_v0.5.0.pt](https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_base_default_v0.5.0.pt)) for ODesign initialization.

# Linux / Docker / Apptainer Users

**Use the [upstream repository](https://github.com/OTeam-AI4S/ODesign) instead.** This fork's patches make sense only for Windows-native installs and aren't tested against the upstream Linux/Docker/Apptainer paths. The `Dockerfile` and `odesign.def` left in this repo are the upstream files preserved for parity but should not be used here — clone upstream for a supported Linux experience.

# Cite

If you use ODesign in your work, please cite the upstream technical report:

```
@misc{zhang2025odesign,
      title={ODesign: A World Model for Biomolecular Interaction Design},
      author={Odin Zhang and Xujun Zhang and Haitao Lin and Cheng Tan and Qinghan Wang and Yuanle Mo and Qiantai Feng and Gang Du and Yuntao Yu and Zichang Jin and Ziyi You and Peicong Lin and Yijie Zhang and Yuyang Tao and Shicheng Chen and Jack Xiaoyu Chen and Chenqing Hua and Weibo Zhao and Runze Ma and Yunpeng Xia and Kejun Ying and Jun Li and Yundian Zeng and Lijun Lang and Peichen Pan and Hanqun Cao and Zihao Song and Bo Qiang and Jiaqi Wang and Pengfei Ji and Lei Bai and Jian Zhang and Chang-yu Hsieh and Pheng Ann Heng and Siqi Sun and Tingjun Hou and Shuangjia Zheng},
      year={2025},
      eprint={2510.22304},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2510.22304},
}
```

# Acknowledgements

The original ODesign project is by Lingang Laboratory, Zhejiang University, The Chinese University of Hong Kong, and Shanghai Artificial Intelligence Laboratory. ODesign builds upon [Protenix](https://github.com/bytedance/Protenix) and [OpenFold](https://github.com/aqlaboratory/openfold). All credit for the model, the architecture, and the training corpus belongs to the upstream authors.

This Windows fork is unaffiliated with the upstream authors. It exists only to make the model runnable natively on Windows for users with consumer GPUs that benefit from WDDM shared memory. No model weights are modified.

# License

Both source code and model parameters are released under the [Apache 2.0 License](LICENSE), inherited from upstream. The Windows-fork patches are released under the same license.
