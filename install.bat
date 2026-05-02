@echo off
setlocal
echo ==========================================
echo    Irodori-TTS-Turbo Installer
echo ==========================================

echo [1/3] Checking for uv...
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo uv not found. Installing uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%PATH%;%USERPROFILE%\.cargo\bin"
) else (
    echo uv is already installed.
)

echo.
echo Please select your hardware:
echo [1] NVIDIA GPU (CUDA 12.8) - Standard
echo [2] Intel GPU / Core Ultra (XPU) - Accelerated
echo [3] CPU only / Mac (MPS) / Other
set /p CHOICE="Enter choice (1-3) [default: 1]: "

if "%CHOICE%"=="" set CHOICE=1

echo.
echo [2/3] Setting up environment...

if "%CHOICE%"=="1" (
    echo Installing for NVIDIA GPU (CUDA 12.8)...
    uv sync --extra-index-url https://download.pytorch.org/whl/cu128
) else if "%CHOICE%"=="2" (
    echo Installing for Intel GPU / XPU...
    uv sync --extra-index-url https://download.pytorch.org/whl/xpu
) else (
    echo Installing for CPU/Standard...
    uv sync
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo [Error] Installation failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ==========================================
echo    Installation Complete!
echo    Run run.bat to start the WebUI.
echo ==========================================
pause
