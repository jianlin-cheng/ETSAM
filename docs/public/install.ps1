# Install ETSAM on Windows with one command (in PowerShell):
#
#   powershell -ExecutionPolicy Bypass -c "irm https://jianlin-cheng.github.io/ETSAM/install.ps1 | iex"
#
# Downloads the ETSAM installer (ETSAM-<version>-Windows-x86_64.exe) from the latest GitHub
# release and runs it without questions: ETSAM goes into %USERPROFILE%\etsam_env,
# with PyTorch and the model weights, and the etsam commands are added to the
# user's PATH.
#
# If ETSAM is already installed (found through %USERPROFILE%\.conda\environments.txt,
# the list of conda environments, and the .etsam-installer file the installer leaves
# in its folder), the script says so and asks whether to upgrade (the latest
# release is newer) or reinstall (it is the same version). Both uninstall the old
# installation and install the new one: the Windows installer cannot update a folder
# in place. The downloaded model weights are kept, and downloaded again only if their
# download URL changed.
#
# Environment variables:
#   ETSAM_PREFIX     install folder (default: %USERPROFILE%\etsam_env). Without it, a
#                    new installation asks for the folder, showing the default.
#   ETSAM_INSTALLER  another installer to run: a URL or a local file
#   ETSAM_YES        1 to not ask anything: use the default folder, and upgrade or
#                    reinstall when ETSAM is installed already
$ErrorActionPreference = 'Stop'

$repo = 'https://github.com/jianlin-cheng/ETSAM'
$marker = '.etsam-installer'   # written by the installer; first word: its version

$prefixSet = [bool]$env:ETSAM_PREFIX
$prefix = if ($env:ETSAM_PREFIX) { $env:ETSAM_PREFIX } else { Join-Path $env:USERPROFILE 'etsam_env' }
if ($prefix -match ' ') {
    # conda environments cannot be in a folder with spaces on Windows
    $prefix = Join-Path $env:SystemDrive 'etsam_env'
}

# The release files have the version in their names (ETSAM-1.2.3-Windows-x86_64.exe), so
# the latest release is looked up first: .../releases/latest redirects to
# .../releases/tag/v1.2.3.
function Get-LatestTag {
    $url = curl.exe -fsLI -o NUL -w '%{url_effective}' "$repo/releases/latest"
    if ($LASTEXITCODE -eq 0 -and $url -match '/releases/tag/([^/]+)$') { return $Matches[1] }
    return $null
}

$latestTag = $null
if ($env:ETSAM_INSTALLER) {
    $source = $env:ETSAM_INSTALLER
} else {
    $latestTag = Get-LatestTag
    if (-not $latestTag) {
        Write-Error "ETSAM: could not find the latest release at $repo/releases/latest (no internet?). Download the installer by hand, see https://jianlin-cheng.github.io/ETSAM/installation/"
    }
    $source = "$repo/releases/download/$latestTag/ETSAM-$($latestTag -replace '^v','')-Windows-x86_64.exe"
}

# --- Is ETSAM installed already? -----------------------------------------------
# The folder we would install into, or (if no folder was chosen) a conda
# environment listed in environments.txt, counts when it has the installer's marker
# file. Listed folders that no longer exist, or are not made by this installer, do not.
function Find-Installed {
    if (Test-Path -PathType Leaf (Join-Path $prefix $marker)) { return $prefix }
    $list = Join-Path $env:USERPROFILE '.conda\environments.txt'
    if (-not $prefixSet -and (Test-Path -PathType Leaf $list)) {
        foreach ($line in Get-Content $list) {
            $dir = $line.Trim()
            if ($dir -and (Test-Path -PathType Leaf (Join-Path $dir $marker))) { return $dir }
        }
    }
    return $null
}

# Version of the installer we would run, or $null if it cannot be told.
function Get-LatestVersion {
    if ($latestTag) { return $latestTag -replace '^v', '' }
    return $null
}

# Compares two versions such as 1.2.0 or 1.2.0rc1 (a pre-release is older than its
# release): returns -1, 0 or 1, or $null if one of them cannot be read.
function Compare-EtsamVersion([string]$a, [string]$b) {
    $keys = @()
    foreach ($text in $a, $b) {
        if ($text.Trim() -notmatch '^v?(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?') { return $null }
        $numbers, $kind, $count = $Matches[1], $Matches[2], $Matches[3]   # before the next -match
        if ($numbers -notmatch '\.') { $numbers += '.0' }
        $rank = switch ($kind) { 'a' { 1 } 'b' { 2 } 'rc' { 3 } default { 4 } }
        $keys += [pscustomobject]@{ Version = [version]$numbers; Rank = $rank; Number = [int]('0' + $count) }
    }
    foreach ($property in 'Version', 'Rank', 'Number') {
        if ($keys[0].$property -lt $keys[1].$property) { return -1 }
        if ($keys[0].$property -gt $keys[1].$property) { return 1 }
    }
    return 0
}

# Asks where to install a new ETSAM. Enter keeps the default; without a console, or
# with ETSAM_YES=1, the default is used.
function Read-InstallFolder {
    if ($env:ETSAM_YES -eq '1' -or -not [Environment]::UserInteractive) { return }
    while ($true) {
        try { $answer = Read-Host "ETSAM: install folder [$prefix] (press Enter to accept, or type another folder)" } catch { return }
        $answer = "$answer".Trim().Trim('"')
        if (-not $answer) { return }
        if ($answer -match ' ') {
            # conda environments cannot be in a folder with spaces on Windows
            Write-Host 'ETSAM: the folder cannot contain spaces. Try another.'
            continue
        }
        if ($answer -eq '~' -or $answer.StartsWith('~\') -or $answer.StartsWith('~/')) {
            $answer = $env:USERPROFILE + $answer.Substring(1)
        }
        $script:prefix = [IO.Path]::GetFullPath($answer)
        return
    }
}

$action = 'install'
$installed = Find-Installed
if (-not $installed -and -not $prefixSet) {
    Read-InstallFolder
    $prefixSet = $true   # the chosen folder is the only one to check from here on
    $installed = Find-Installed
}
if ($installed) {
    $prefix = $installed
    $oldVersion = ((Get-Content (Join-Path $prefix $marker) -TotalCount 1) -split ' ')[0].Trim()
    $newVersion = Get-LatestVersion
    $order = if ($newVersion) { Compare-EtsamVersion $oldVersion $newVersion } else { $null }
    if ($null -eq $order) {
        $action = 'reinstall'
        Write-Host "ETSAM $oldVersion is already installed in $prefix."
        $question = 'Reinstall it? [y/N]'
    } elseif ($order -eq 0) {
        $action = 'reinstall'
        Write-Host "ETSAM $oldVersion (the latest version) is already installed in $prefix."
        $question = 'Reinstall it? [y/N]'
    } elseif ($order -lt 0) {
        $action = 'upgrade'
        Write-Host "ETSAM $oldVersion is installed in $prefix. The latest version is $newVersion."
        $question = "Upgrade to $newVersion? [Y/n]"
    } else {
        $action = 'reinstall'
        Write-Host "ETSAM $oldVersion is installed in $prefix, which is newer than the latest release ($newVersion)."
        $question = "Reinstall it as version ${newVersion}? [y/N]"
    }
    Write-Host 'The downloaded model weights are kept (they are downloaded again only if their URL changed).'
    Write-Host 'Close ChimeraX, napari and any other program that uses ETSAM first.'

    if ($env:ETSAM_YES -eq '1') {
        Write-Host "ETSAM: ETSAM_YES=1, so continuing with the $action."
    } else {
        try { $answer = Read-Host $question } catch { $answer = $null }
        if ($null -eq $answer) {
            Write-Error "ETSAM: no console to ask on. Run this again in a PowerShell window, or set ETSAM_YES=1 to $action without asking."
        }
        $yes = if ($action -eq 'upgrade') { $answer -eq '' -or $answer -match '^[Yy]' } else { $answer -match '^[Yy]' }
        if (-not $yes) {
            Write-Host 'ETSAM: nothing changed.'
            return
        }
    }
} elseif (Test-Path $prefix) {
    Write-Error "ETSAM: $prefix already exists, but it was not made by the ETSAM installer. Choose another folder with ETSAM_PREFIX, or delete it first."
}

$installer = Join-Path ([IO.Path]::GetTempPath()) 'ETSAM-Windows-x86_64.exe'
if (Test-Path -PathType Leaf $source) {
    Copy-Item $source $installer -Force
} else {
    Write-Host "ETSAM: downloading the installer from $source"
    # curl.exe is part of Windows 10 and 11, and shows the download progress.
    curl.exe -fL --progress-bar -o $installer $source
    if ($LASTEXITCODE -ne 0) { Write-Error 'ETSAM: downloading the installer failed.' }
}

# The model weights are moved aside while the old installation is removed (a rename
# next to the installation folder: no copy, no extra disk space), and moved back into
# the new one before ETSAM checks them. $keep also survives a failed run.
$keep = "$prefix.checkpoints-keep"
if ($installed) {
    Write-Host "ETSAM: removing the old installation in $prefix (keeping the model weights)"
    if (Test-Path (Join-Path $prefix 'checkpoints')) {
        if (Test-Path $keep) { Remove-Item -Recurse -Force $keep }
        Move-Item (Join-Path $prefix 'checkpoints') $keep
    }
    # The uninstaller also removes the commands folder from the PATH and the
    # entries in the list of apps. /_?= (the last argument, unquoted) makes it run
    # in this folder and wait, instead of copying itself elsewhere and returning.
    $uninstaller = Join-Path $prefix 'Uninstall-ETSAM.exe'
    $failure = $null
    try {
        if (Test-Path $uninstaller) {
            Start-Process -FilePath $uninstaller -ArgumentList @('/S', "/_?=$prefix") -Wait | Out-Null
        }
        if (Test-Path $prefix) { Remove-Item -Recurse -Force $prefix }   # what is left, e.g. the uninstaller
    } catch {
        $failure = $_.Exception.Message
    }
    if ($failure -or (Test-Path $prefix)) {
        # put the weights back where they were
        if ((Test-Path $keep) -and (Test-Path $prefix) -and -not (Test-Path (Join-Path $prefix 'checkpoints'))) {
            Move-Item $keep (Join-Path $prefix 'checkpoints')
        }
        Remove-Item $installer -ErrorAction SilentlyContinue
        Write-Error "ETSAM: could not remove the old installation in $prefix ($failure). Close the programs that use ETSAM (ChimeraX, napari, terminals) and try again."
    }
}

$userSkipsWeights = $env:ETSAM_SKIP_WEIGHTS -eq '1'
if (Test-Path $keep) {
    $env:ETSAM_SKIP_WEIGHTS = '1'   # restored and checked below, instead of downloaded
}

Write-Host "ETSAM: installing into $prefix (this downloads PyTorch$(if (-not (Test-Path $keep)) { ' and the model weights' }), and takes several minutes)"
# /D must be the last argument, without quotes.
$arguments = @('/S', '/InstallationType=JustMe', '/AddToPath=0', '/RegisterPython=0', "/D=$prefix")
$process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru
Remove-Item $installer -ErrorAction SilentlyContinue
if ($process.ExitCode -ne 0 -or -not (Test-Path (Join-Path $prefix $marker))) {
    # The installer keeps the output of the step that failed in .step.log.
    $log = Join-Path $prefix '.step.log'
    if (Test-Path $log) {
        Write-Host '--- output of the failed installation step ---'
        Get-Content $log -Tail 40 | Write-Host
        Write-Host '----------------------------------------------'
    }
    if (Test-Path $keep) { Write-Host "The model weights are kept in $keep and used by the next installation." }
    Write-Error "ETSAM: the installation failed (exit code $($process.ExitCode)). To see the full output, run the installer by hand: $source"
}

if (Test-Path $keep) {
    $checkpoints = Join-Path $prefix 'checkpoints'
    if (Test-Path $checkpoints) { Remove-Item -Recurse -Force $checkpoints }
    Move-Item $keep $checkpoints
    if (-not $userSkipsWeights) {
        # keeps the weights unless their download URL changed; the DLL folders are
        # put on the PATH as the etsam command does
        $env:ETSAM_CHECKPOINT_DIR = $checkpoints
        $env:PATH = "$prefix;$prefix\Library\mingw-w64\bin;$prefix\Library\usr\bin;$prefix\Library\bin;$env:PATH"
        & (Join-Path $prefix 'python.exe') -I -c 'import etsam; etsam.ensure_checkpoints()'
        if ($LASTEXITCODE -ne 0) { Write-Error 'ETSAM: checking the model weights failed.' }
    }
}
Write-Host "ETSAM: $(switch ($action) { 'upgrade' { 'upgraded' } 'reinstall' { 'reinstalled' } default { 'installed' } }). Open a new terminal and run: etsam --help"
