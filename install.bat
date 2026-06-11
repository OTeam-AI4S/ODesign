@echo off
REM ===========================================================================
REM ODesign — Windows Native Install (Phases C, D, E)
REM ---------------------------------------------------------------------------
REM Runs the install steps that follow Phase A (Windows tooling) and Phase B
REM (clone + source patches). After this completes you can launch:
REM     .\inference_demo.bat            (default protein binder example)
REM     .\run_all_examples.bat          (smoke-test all 11 examples)
REM
REM Prerequisites you must have done first (Phase A in ODESIGN_INSTALL_CHECKLIST.md):
REM   - NVIDIA driver >= 550 + RTX-class GPU
REM   - CUDA Toolkit 12.1.1 installed (nvcc --version reports release 12.1)
REM   - Visual Studio 2022 Build Tools with C++ workload
REM   - Miniconda
REM   - Git for Windows (provides Git Bash for the checkpoint download script)
REM
REM Run this from the "x64 Native Tools Command Prompt for VS 2022" so that
REM MSVC and CUDA_HOME are on PATH for any source-build wheels that fall back.
REM
REM (Claude, 2026-04-28)
REM ===========================================================================

setlocal

echo ===========================================================================
echo  ODesign Windows install starting at %date% %time%
echo ===========================================================================
echo.

REM ---------------------------------------------------------------------------
REM Step 1: sanity checks
REM ---------------------------------------------------------------------------
echo [1/6] Sanity checks...
where nvcc >nul 2>&1 || (
    echo   [ERROR] nvcc not found on PATH. Install CUDA Toolkit 12.1.1 first.
    echo          See ODESIGN_INSTALL_CHECKLIST.md Phase A.
    exit /b 1
)
where conda >nul 2>&1 || (
    echo   [ERROR] conda not found. Install Miniconda first.
    exit /b 1
)
where git >nul 2>&1 || (
    echo   [WARN ] git not found. You'll need it for the checkpoint download in step 5.
)
nvcc --version | findstr "release 12.1" >nul || (
    echo   [WARN ] nvcc reports a CUDA version that isn't 12.1. PyTorch is pinned to
    echo          torch==2.3.1+cu121, so a mismatch may cause subtle issues.
)
echo   OK
echo.

REM ---------------------------------------------------------------------------
REM Step 2: ensure conda env "odesign" exists and is active
REM ---------------------------------------------------------------------------
echo [2/6] Conda environment setup...
conda env list | findstr /B "odesign " >nul
if errorlevel 1 (
    echo   Creating fresh "odesign" env with Python 3.10...
    call conda create -n odesign python=3.10 -y || exit /b 1
) else (
    echo   "odesign" env already exists, skipping creation.
)
call conda activate odesign || exit /b 1
echo   Active env: %CONDA_DEFAULT_ENV%
echo.

REM ---------------------------------------------------------------------------
REM Step 3: PyTorch from NVIDIA's cu121 index
REM ---------------------------------------------------------------------------
echo [3/6] Installing PyTorch 2.3.1 + cu121...
python -c "import torch; assert torch.__version__.startswith('2.3.1') and torch.version.cuda=='12.1'" >nul 2>&1
if not errorlevel 1 (
    echo   PyTorch 2.3.1+cu121 already installed, skipping.
) else (
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 ^
        --index-url https://download.pytorch.org/whl/cu121 || exit /b 1
)
echo   Sanity check:
python -c "import torch; print('   torch:', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| cuda version:', torch.version.cuda)" || exit /b 1
echo.

REM ---------------------------------------------------------------------------
REM Step 4: rest of requirements.txt with DeepSpeed JIT-mode env vars
REM ---------------------------------------------------------------------------
echo [4/6] Installing remaining requirements.txt deps...
echo   Setting DeepSpeed JIT-mode env vars to skip op pre-compilation on Windows.
SET DS_BUILD_OPS=0
SET DS_BUILD_AIO=0
SET DS_BUILD_GDS=0
SET DS_BUILD_INFERENCE_CUTLASS=0
SET DS_BUILD_RAGGED_OPS=0
SET BUILD_OP_PLATFORM=0

pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
if errorlevel 1 (
    echo.
    echo   [WARN ] Initial requirements.txt install failed. Retrying with deepspeed
    echo           downgrade ladder (0.15.4 -^> 0.14.0) since DeepSpeed Windows
    echo           install can be flaky across versions...
    pip install deepspeed==0.15.4 || pip install deepspeed==0.14.0
    pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.3.1+cu121.html || (
        echo   [ERROR] Install still failed. Inspect the trace above and consult
        echo           ODESIGN_INSTALL_CHECKLIST.md Appendix A.
        exit /b 1
    )
)
echo   OK
echo.

REM ---------------------------------------------------------------------------
REM Step 5: import sanity checks for the patched packages
REM ---------------------------------------------------------------------------
echo [5/6] Verifying critical imports...
python -c "import torch, deepspeed, protenix, biotite.interface.rdkit, prody, addict, cpdb, gdown; print('   all imports OK')" || (
    echo   [ERROR] One of the critical packages failed to import. See traceback above.
    exit /b 1
)
echo.

REM ---------------------------------------------------------------------------
REM Step 6: download checkpoints (delegates to bash script)
REM ---------------------------------------------------------------------------
echo [6/6] Downloading model checkpoints (HuggingFace + ipd.uw.edu, ~17 GB total)...
where bash >nul 2>&1
if errorlevel 1 (
    if exist "C:\Program Files\Git\bin\bash.exe" (
        "C:\Program Files\Git\bin\bash.exe" ./ckpt/get_odesign_ckpt.sh
    ) else (
        echo   [WARN ] bash not on PATH and Git for Windows not at default location.
        echo           Skipping. Run this manually from Git Bash:
        echo               bash ./ckpt/get_odesign_ckpt.sh
    )
) else (
    bash ./ckpt/get_odesign_ckpt.sh
)
echo.

REM ---------------------------------------------------------------------------
REM Done
REM ---------------------------------------------------------------------------
echo ===========================================================================
echo  Install complete.
echo ===========================================================================
echo.
echo  Next steps:
echo.
echo  1. Download CCD data from Google Drive (one-time, ~530 MB):
echo        gdown --folder https://drive.google.com/drive/folders/1wPmwIrC3G52q1JFY0RXY95tjKDl7YEln -O data
echo.
echo  2. Run the default protein-binder example:
echo        .\inference_demo.bat
echo.
echo  3. Or smoke-test all 11 examples (~30 min wall-clock):
echo        .\run_all_examples.bat
echo.
echo  Finished at %date% %time%
echo ===========================================================================
