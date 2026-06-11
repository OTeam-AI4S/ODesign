@echo off
REM ====================================================================
REM ODesign Antibody Inference (Windows native)
REM (Claude, 2026-04-28)
REM
REM 1. Generate input JSON from a PDB if needed:
REM      python .\scripts\identify_cdr.py path\to\antibody.pdb --output examples\my_input.json --chothia --odesign
REM 2. Edit the SET lines below to point at your JSON, then run:
REM      run_antibody.bat
REM ====================================================================

SET input_json_path=./examples/nanobody.json
SET exp_name=ab_test
SET data_root_dir=./data
SET ckpt_root_dir=./ckpt
SET seeds=[42]
SET N_sample=1
SET num_workers=0
SET use_msa=false

SET PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SET CUDA_VISIBLE_DEVICES=0

echo -----------------------------------------------------------
echo  Start ODesign Antibody Inference (Windows)
echo -----------------------------------------------------------
echo Input JSON  : %input_json_path%
echo Exp name    : %exp_name%
echo Workers     : %num_workers%
echo CUDA device : %CUDA_VISIBLE_DEVICES%
echo -----------------------------------------------------------

python .\scripts\inference.py ^
    data_root_dir=%data_root_dir% ^
    ckpt_root_dir=%ckpt_root_dir% ^
    exp.input_json_path=%input_json_path% ^
    exp.exp_name=%exp_name% ^
    exp.seeds=%seeds% ^
    exp.model.sample_diffusion.N_sample=%N_sample% ^
    exp.use_msa=%use_msa% ^
    exp.num_workers=%num_workers%

if errorlevel 1 (
    echo.
    echo FAILED. Check traceback; see ODESIGN_INSTALL_CHECKLIST.md Appendix A.
    exit /b 1
)

echo -----------------------------------------------------------
echo  Antibody inference completed.
echo -----------------------------------------------------------
