@echo off
REM ===========================================================================
REM ODesign — Re-run the three examples that failed in run_all_examples.bat
REM
REM Tests Patch 6 (vendored gRNAde data files) and Patch 7 (negative-residue
REM parser fix). Each example runs with N_sample=1 and dumps to its own folder
REM under .\outputs\fix_*\<timestamp>\.
REM
REM Usage (from PowerShell or cmd, with the odesign conda env active):
REM     .\inference_failure_fixes.bat
REM
REM Wall-clock estimate: ~10-15 min total (~3-5 min per example).
REM (Claude, 2026-04-28)
REM ===========================================================================

setlocal enabledelayedexpansion

set TOTAL=0
set PASSED=0
set FAILED=0
set FAILED_LIST=

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set CUDA_VISIBLE_DEVICES=0

echo ===========================================================================
echo  ODesign — Re-running previously-failed examples
echo  Started: %date% %time%
echo ===========================================================================
echo.
echo  Re-runs the RNA examples that hit the gRNAde import chain. Targets
echo  Patches 6 (vendored gRNAde files), 8 (cpdb-protein), 9 (pre-rewritten
echo  imports). The ptr example is skipped here -- already confirmed working
echo  via Patch 7.
echo.
echo    - na_prot_binding_rna  (Patches 6, 8, 9)
echo    - na_rna_bb            (Patches 6, 8, 9)
echo.

REM --- Patch 7 (ptr) verified previously, skipping. Uncomment to retest:
REM call :run_test ptr                  odesign_base_prot_flex     protein "./examples/ptr_design/ptr.json"

REM --- Patches 6, 8, 9 verification: gRNAde RNA invfold ---
call :run_test na_prot_binding_rna  odesign_base_na_rigid      rna     "./examples/na_design/prot_binding_rna/odesign_input.json"
call :run_test na_rna_bb            odesign_base_na_rigid      rna     "./examples/na_design/rna_bb/odesign_input.json"

REM --- Summary ---
echo.
echo ===========================================================================
echo  TEST SUMMARY
echo ===========================================================================
echo  Total:  %TOTAL%
echo  Passed: %PASSED%
echo  Failed: %FAILED%
if not "!FAILED_LIST!"=="" (
    echo  Failed tests:!FAILED_LIST!
)
echo  Finished: %date% %time%
echo  Outputs in: .\outputs\fix_*\
echo.
echo  Verify CIFs were actually produced (the runner reports PASS based on Python
echo  exit code, but per-sample featurization errors don't fail the script):
echo    dir .\outputs\fix_ptr\*\ptr\seed_42\predictions\*.cif
echo    dir .\outputs\fix_na_prot_binding_rna\*\*\seed_42\predictions\*.cif
echo    dir .\outputs\fix_na_rna_bb\*\*\seed_42\predictions\*.cif
echo ===========================================================================

if %FAILED% gtr 0 exit /b 1
exit /b 0


REM ---------------------------------------------------------------------------
REM Subroutine: run a single example
REM   %1 = test name (used for exp_name and output folder, prefixed "fix_")
REM   %2 = model name
REM   %3 = design modality
REM   %4 = input JSON path (quoted)
REM ---------------------------------------------------------------------------
:run_test
set /a TOTAL+=1
set name=%~1
set model=%~2
set modality=%~3
set jsonpath=%~4

echo.
echo ---------------------------------------------------------------------------
echo  Test %TOTAL%: %name%
echo  Model: %model% / Modality: %modality%
echo  Input: %jsonpath%
echo ---------------------------------------------------------------------------

python ./scripts/inference.py ^
    exp=train_%model% ^
    exp.infer_model_name=%model% ^
    exp.design_modality=%modality% ^
    exp.input_json_path=%jsonpath% ^
    exp.exp_name=fix_%name% ^
    exp.seeds=[42] ^
    exp.model.sample_diffusion.N_sample=1 ^
    exp.use_msa=false ^
    exp.num_workers=0 ^
    exp.model.inference_noise_schedulers.coordinate.partial_diffusion.enable=false ^
    exp.model.inference_noise_schedulers.coordinate.partial_diffusion.snr=0.1

if errorlevel 1 (
    set /a FAILED+=1
    set FAILED_LIST=!FAILED_LIST! %name%
    echo  ^>^> FAILED: %name%
) else (
    set /a PASSED+=1
    echo  ^>^> PASSED: %name%
)
exit /b 0
