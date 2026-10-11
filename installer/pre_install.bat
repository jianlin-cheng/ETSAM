@echo off
:: ETSAM installer: pre-install check (Windows).
::
:: The installer currently supports NVIDIA GPUs only, so it stops early, with a
:: clear message, on computers without a working NVIDIA driver. Support for
:: other accelerators can be added here and in post_install.bat; it needs testing
:: on such hardware, and contributions are welcome:
:: https://github.com/jianlin-cheng/ETSAM
::
:: Set ETSAM_SKIP_GPU_CHECK=1 to skip this check (e.g. for testing in CI).

if "%ETSAM_SKIP_GPU_CHECK%"=="1" (
    echo ETSAM: skipping the GPU check ^(ETSAM_SKIP_GPU_CHECK=1^).
    exit /b 0
)

echo ETSAM: checking for an NVIDIA GPU driver...

:: The NVIDIA driver installs the CUDA driver library nvcuda.dll, which PyTorch
:: needs. (wmic is not used: it is removed from recent Windows versions.)
if exist "%SystemRoot%\System32\nvcuda.dll" (
    echo ETSAM: NVIDIA driver found.
    exit /b 0
)

echo ==========================================================
echo ETSAM: this installer requires an NVIDIA GPU with a working
echo driver, and none was found. Install the NVIDIA driver and run
echo this installer again. Installation stopped.
echo Other GPUs are not supported by the installer yet; see
echo https://jianlin-cheng.github.io/ETSAM/installation/
echo ==========================================================
exit /b 1
