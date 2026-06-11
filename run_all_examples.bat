@echo off
REM ===========================================================================
REM ODesign — Run ALL pre-staged examples with N_sample=1 (Windows native)
REM
REM Sanity-check script: walks every example shipped with the repo, runs each
REM with a single sample, dumps each to its own output folder, and prints a
REM pass/fail summary at the end.
REM
REM Usage (from x64 Native Tools or PowerShell or cmd, with conda env active):
REM     .\run_all_examples.bat
REM
REM Wall-clock estimate: ~3 minutes per example * 11 examples = ~30 min.
REM Outputs land under .\outputs\test_<name>\<timestamp>\
REM ===========================================================================

setlocal enabledelayedexpansion

set TOTAL=0
set PASSED=0
set FAILED=0
set FAILED_LIST=

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set CUDA_VISIBLE_DEVICES=0

echo ===========================================================================
echo  ODesign full-example smoke test
echo  Started: %date% %time%
echo ===========================================================================

REM --- Protein design family (all use odesign_base_prot_flex + modality protein) ---
call :run_test prot_binding_prot       odesign_base_prot_flex     protein "./examples/protein_design/prot_binding_prot/odesign_input.json"
call :run_test motif_scaffold          odesign_base_prot_flex     protein "./examples/protein_design/motif_scaffold/odesign_input.json"
call :run_test atom_scaffold           odesign_base_prot_flex     protein "./examples/protein_design/atom_scaffold/odesign_input.json"
call :run_test lig_binding_prot        odesign_base_prot_flex     protein "./examples/protein_design/lig_binding_prot/odesign_input.json"
call :run_test lig_binding_prot_smiles odesign_base_prot_flex     protein "./examples/protein_design/lig_binding_prot_smiles/odesign_input.json"
call :run_test cyclic_peptide          odesign_base_prot_flex     protein "./examples/cyclic_peptide_design/odesign_input.json"
call :run_test ptr                     odesign_base_prot_flex     protein "./examples/ptr_design/ptr.json"

REM --- Partial-diffusion variant (same model, special flag) ---
call :run_partial_diff prot_binding_prot_partial_diff odesign_base_prot_flex protein "./examples/protein_design/prot_binding_prot_partial_diff/odesign_input.json"

REM --- Ligand design family (ligand model + ligand modality) ---
call :run_test ligand_design   odesign_base_ligand_rigid  ligand "./examples/ligand_design/prot_binding_lig/odesign_input.json"

REM --- Nucleic-acid design family (NA model + rna modality) ---
call :run_test na_prot_binding_rna  odesign_base_na_rigid  rna "./examples/na_design/prot_binding_rna/odesign_input.json"
call :run_test na_rna_bb            odesign_base_na_rigid  rna "./examples/na_design/rna_bb/odesign_input.json"

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
echo  Outputs in: .\outputs\test_*\
echo ===========================================================================

if %FAILED% gtr 0 exit /b 1
exit /b 0


REM ---------------------------------------------------------------------------
REM Subroutine: run a single example
REM   %1 = test name (used for exp_name and output folder)
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
    exp.exp_name=test_%name% ^
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


REM ---------------------------------------------------------------------------
REM Subroutine: run partial-diffusion variant (sets the partial_diff flag)
REM ---------------------------------------------------------------------------
:run_partial_diff
set /a TOTAL+=1
set name=%~1
set model=%~2
set modality=%~3
set jsonpath=%~4

echo.
echo ---------------------------------------------------------------------------
echo  Test %TOTAL%: %name%  (partial diffusion mode, SNR=0.1)
echo  Model: %model% / Modality: %modality%
echo  Input: %jsonpath%
echo ---------------------------------------------------------------------------

python ./scripts/inference.py ^
    exp=train_%model% ^
    exp.infer_model_name=%model% ^
    exp.design_modality=%modality% ^
    exp.input_json_path=%jsonpath% ^
    exp.exp_name=test_%name% ^
    exp.seeds=[42] ^
    exp.model.sample_diffusion.N_sample=1 ^
    exp.use_msa=false ^
    exp.num_workers=0 ^
    exp.model.inference_noise_schedulers.coordinate.partial_diffusion.enable=true ^
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
