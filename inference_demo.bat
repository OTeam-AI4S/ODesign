@echo off
REM ====================================================================
REM ODesign Inference Demo (Windows native, cmd.exe)
REM Mirror of inference_demo.sh — edit the SET lines below, then run:
REM     inference_demo.bat
REM (Claude, 2026-04-28)
REM ====================================================================

REM 1. Inference model name (REQUIRED)
REM    odesign_base_prot_flex / odesign_base_prot_rigid / odesign_base_ligand_rigid / odesign_base_na_rigid
SET infer_model_name=odesign_base_prot_flex

REM 2. Design modality — leave empty unless using an NA model
SET design_modality=

REM 3. Data root (must contain components.v20240608.cif and the rdkit pkl)
SET data_root_dir=./data

REM 4. Checkpoint root
SET ckpt_root_dir=./ckpt

REM 5. Input JSON (REQUIRED)
SET input_json_path=./examples/protein_design/prot_binding_prot/odesign_input.json

REM 6. Experiment name
SET exp_name=protein_binding_protein_design

REM 7. Seeds
SET seeds=[42]

REM 8. Samples per seed
SET N_sample=1

REM 9. Use precomputed MSA (set true ONLY if input JSON has msa.precomputed_msa_dir; see Phase I in checklist)
SET use_msa=false

REM 10. DataLoader workers — keep 0 on Windows for first run
SET num_workers=0

REM 11. CUDA setup
SET PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SET CUDA_VISIBLE_DEVICES=0

REM Partial diffusion (advanced)
SET enable_partial_diff=false
SET snr=0.1

REM ---- Auto-assign design_modality from model name ----
echo %infer_model_name% | findstr /C:"prot" >nul && SET design_modality=protein
echo %infer_model_name% | findstr /C:"ligand" >nul && SET design_modality=ligand
echo %infer_model_name% | findstr /C:"na" >nul && (
    if "%design_modality%"=="" (
        echo ERROR: Please set design_modality to "dna" or "rna" for an NA model.
        exit /b 1
    )
)

if "%exp_name%"=="" SET exp_name=infer_%infer_model_name%

echo -----------------------------------------------------------
echo  Start ODesign Inference (Windows native)
echo -----------------------------------------------------------
echo Model            : %infer_model_name%
echo Modality         : %design_modality%
echo Input JSON       : %input_json_path%
echo Exp name         : %exp_name%
echo Seeds            : %seeds%
echo N_sample         : %N_sample%
echo Use MSA          : %use_msa%
echo Workers          : %num_workers%
echo Data root        : %data_root_dir%
echo Ckpt root        : %ckpt_root_dir%
echo CUDA device      : %CUDA_VISIBLE_DEVICES%
echo Allocator        : %PYTORCH_CUDA_ALLOC_CONF%
echo -----------------------------------------------------------

python ./scripts/inference.py ^
    exp=train_%infer_model_name% ^
    data_root_dir=%data_root_dir% ^
    ckpt_root_dir=%ckpt_root_dir% ^
    exp.infer_model_name=%infer_model_name% ^
    exp.design_modality=%design_modality% ^
    exp.input_json_path=%input_json_path% ^
    exp.exp_name=%exp_name% ^
    exp.seeds=%seeds% ^
    exp.model.sample_diffusion.N_sample=%N_sample% ^
    exp.use_msa=%use_msa% ^
    exp.num_workers=%num_workers% ^
    exp.model.inference_noise_schedulers.coordinate.partial_diffusion.enable=%enable_partial_diff% ^
    exp.model.inference_noise_schedulers.coordinate.partial_diffusion.snr=%snr%

if errorlevel 1 (
    echo.
    echo FAILED. Check the traceback above; consult ODESIGN_INSTALL_CHECKLIST.md Appendix A.
    exit /b 1
)

echo -----------------------------------------------------------
echo  ODesign inference completed.
echo -----------------------------------------------------------
