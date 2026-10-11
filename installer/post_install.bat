@echo off
:: ETSAM installer: post-install step (Windows).
::
:: Installs PyTorch for the computer's accelerator; each case below installs it
:: its own way. NVIDIA and CPU use conda-forge; for NVIDIA GPUs, conda picks the
:: CUDA build that matches the installed driver. Only NVIDIA GPUs are supported
:: for now (pre_install.bat enforces this); other accelerators can be added as
:: new cases below, and may need PyTorch's own pip packages instead.
:: Contributions are welcome.
::
:: ETSAM_ACCELERATOR overrides the choice: nvidia (default) or cpu (for testing).

:: PyTorch version ETSAM is validated with (see post_install.sh).
set "PYTORCH_VERSION=2.7.1"
:: PyTorch is installed with the installer's own conda (_conda.exe, which stays in
:: the installation folder). The conda in this environment (26.9) cannot create
:: the .exe commands of new packages here: it looks for the launchers of the
:: conda-launchers package in that package's record, and the installer writes the
:: records of the packages it contains without the file list it needs.
set "CONDA=%PREFIX%\_conda.exe"
if "%ETSAM_ACCELERATOR%"=="" set "ETSAM_ACCELERATOR=nvidia"

if "%ETSAM_ACCELERATOR%"=="nvidia" (
    echo ETSAM: installing PyTorch with CUDA support ^(downloads about 2-3 GB^)...
    "%CONDA%" install --yes --quiet --prefix "%PREFIX%" --override-channels -c conda-forge "pytorch-gpu=%PYTORCH_VERSION%"
    if errorlevel 1 exit /b 1
) else if "%ETSAM_ACCELERATOR%"=="cpu" (
    echo ETSAM: installing the CPU-only PyTorch build...
    "%CONDA%" install --yes --quiet --prefix "%PREFIX%" --override-channels -c conda-forge "pytorch-cpu=%PYTORCH_VERSION%"
    if errorlevel 1 exit /b 1
) else (
    rem Possible future cases, untested: rocm, xpu. These may install PyTorch's
    rem own pip packages instead, e.g. "%PREFIX%\python.exe" -m pip install ...
    echo ETSAM: unknown ETSAM_ACCELERATOR "%ETSAM_ACCELERATOR%" ^(use nvidia or cpu^).
    exit /b 1
)

:: No "conda clean" here: the installer clears the package cache (pkgs\, where
:: this script runs from) itself after this script.

:: The model weights (1.9 GB), into the installation folder, where ETSAM finds them.
:: ETSAM_SKIP_WEIGHTS=1 skips this (CI); ETSAM then downloads them on first use.
if not "%ETSAM_SKIP_WEIGHTS%"=="1" (
    echo ETSAM: downloading the model weights ^(1.9 GB^)...
    set "ETSAM_CHECKPOINT_DIR=%PREFIX%\checkpoints"
    "%PREFIX%\python.exe" -c "import etsam; etsam.ensure_checkpoints()"
    if errorlevel 1 exit /b 1
)

:: Make the etsam commands available in the terminal. The folder commands\ holds
:: only etsam.cmd and etsam-postprocess.cmd, so putting it on the PATH adds nothing
:: else (not this environment's python, pip or conda; its Scripts\ folder holds
:: those). Each command runs ETSAM with this environment's Python in isolated
:: mode (-I), so that the user's Python settings and packages are not used.
:: Each command also puts this environment's DLL folders on its own PATH (setlocal
:: keeps that from the calling terminal), as "conda activate" does: cuDNN loads
:: its parts (cudnn_graph64_9.dll, ...) from Library\bin through the PATH, and
:: without it ETSAM crashes when it first runs the model on the GPU.
:: The folder goes on the user's PATH; pre_uninstall.bat removes it again.
:: PowerShell is called by its full path: the installer runs this script with a
:: PATH that does not include PowerShell's folder.
:: ETSAM_NO_PATH=1 skips adding it to the PATH.
if not exist "%PREFIX%\commands" mkdir "%PREFIX%\commands"
set "DLL_PATH=%PREFIX%;%PREFIX%\Library\mingw-w64\bin;%PREFIX%\Library\usr\bin;%PREFIX%\Library\bin"
> "%PREFIX%\commands\etsam.cmd" echo @setlocal
>> "%PREFIX%\commands\etsam.cmd" echo @set "PATH=%DLL_PATH%;%%PATH%%"
>> "%PREFIX%\commands\etsam.cmd" echo @"%PREFIX%\python.exe" -I -c "import sys, etsam; sys.argv[0] = 'etsam'; sys.exit(etsam.main())" %%*
> "%PREFIX%\commands\etsam-postprocess.cmd" echo @setlocal
>> "%PREFIX%\commands\etsam-postprocess.cmd" echo @set "PATH=%DLL_PATH%;%%PATH%%"
>> "%PREFIX%\commands\etsam-postprocess.cmd" echo @"%PREFIX%\python.exe" -I -c "import sys, postprocess; sys.argv[0] = 'etsam-postprocess'; sys.exit(postprocess._run_cli())" %%*
if not "%ETSAM_NO_PATH%"=="1" (
    "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -Command "$d = $env:PREFIX + '\commands'; $p = [Environment]::GetEnvironmentVariable('Path', 'User'); if (-not $p) { $p = '' }; if (($p -split ';') -notcontains $d) { [Environment]::SetEnvironmentVariable('Path', (($d + ';' + $p).TrimEnd(';')), 'User') }"
    if errorlevel 1 exit /b 1
    echo ETSAM: added %PREFIX%\commands to the PATH. To use the etsam command, open a new terminal.
)

:: Marker that the ETSAM plugins (ChimeraX, napari) use to find this environment.
echo %INSTALLER_VER% %ETSAM_ACCELERATOR%> "%PREFIX%\.etsam-installer"
echo ETSAM: installation complete.
exit /b 0
