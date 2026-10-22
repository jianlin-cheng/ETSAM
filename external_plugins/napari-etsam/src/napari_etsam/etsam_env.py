"""
Find, or install, the ETSAM environment.

Shared, identical copy in the ChimeraX plugin (chimerax.etsam) and the napari
plugin (napari_etsam); keep them in sync.

ETSAM (PyTorch and its bundled ``sam2`` package) is not installed into the host
application. It lives in its own conda environment, created by the ETSAM
installer (or by hand, as for the command line), and the plugins run
``etsam_worker.py`` with that environment's Python.

Environments are found the way ``conda env list`` does: from conda's registry
``~/.conda/environments.txt`` (the installer registers itself there) or, if that
has none, from ``conda env list --json``. Environments made by the installer
(marked by ``.etsam-installer``) are preferred over an environment named
``etsam``. Set ETSAM_PYTHON to use a specific Python instead.

If none is found, ``install()`` downloads the installer for this platform and
runs it silently. The installer supports NVIDIA GPUs on Linux and Windows
(x86-64); it stops with a clear message on other computers.

A plugin release needs at least the ETSAM release of the same version
(``needs_update``). ``update()`` reinstalls an installer-made environment with the
latest release, as install.sh and install.ps1 do for a reinstall.
"""

from __future__ import annotations

import json
import os
import platform
import queue
import re
import threading
import time
from collections import deque
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

INSTALL_DOCS = "https://jianlin-cheng.github.io/ETSAM/installation/"
RELEASES = "https://github.com/jianlin-cheng/ETSAM/releases"
ENV_NAME = "etsam"
MARKER = ".etsam-installer"  # written by the installer's post-install script
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "etsam_worker.py")
INSTALL_MESSAGE = (
    "ETSAM is not installed. Click Install ETSAM to download and set it up. "
    "This downloads about 5 GB and may take 10–15 minutes. "
    "Keep the application open until installation finishes. "
    "ETSAM requires Linux or Windows and an NVIDIA GPU. macOS is not supported."
)
NOT_FOUND = (
    "ETSAM is not installed. ETSAM requires Linux or Windows and an NVIDIA GPU. "
    "macOS is not supported."
)
UPDATE_MESSAGE = (
    "ETSAM {installed} is older than this plugin ({plugin}), which cannot be used "
    "until ETSAM is updated. Click Update ETSAM to install the latest release "
    "(about 3 GB download, 10–15 minutes)."
)


def env_python(prefix: str) -> str:
    if sys.platform == "win32":
        return os.path.join(prefix, "python.exe")
    return os.path.join(prefix, "bin", "python")


def _registered_prefixes() -> list[str]:
    """Environment prefixes known to conda, as listed by ``conda env list``."""
    prefixes = []
    try:
        with open(os.path.expanduser(os.path.join("~", ".conda", "environments.txt"))) as f:
            prefixes = [line.strip() for line in f if line.strip()]
    except OSError:
        pass
    return prefixes


def _conda_env_list() -> list[str]:
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda") or shutil.which("mamba")
    if not conda:
        return []
    try:
        out = subprocess.run(
            [conda, "env", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            **subprocess_options(),
        )
        return json.loads(out.stdout).get("envs", [])
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def _pick(prefixes: list[str]) -> str | None:
    usable = [p for p in prefixes if os.path.exists(env_python(p))]
    for prefix in usable:  # made by the ETSAM installer
        if os.path.exists(os.path.join(prefix, MARKER)):
            return env_python(prefix)
    for prefix in usable:  # e.g. "conda env create -f environment.yml"
        if os.path.basename(os.path.normpath(prefix)).lower() == ENV_NAME:
            return env_python(prefix)
    return None


def find_python() -> str | None:
    """Python of the ETSAM environment, or None if none is found."""
    override = os.environ.get("ETSAM_PYTHON")
    if override:
        return os.path.expanduser(override)
    return _pick(_registered_prefixes()) or _pick(_conda_env_list())


# --------------------------------------------------------------------------- #
# Versions
# --------------------------------------------------------------------------- #


def etsam_version(python: str) -> str | None:
    """Version of ETSAM in the environment of ``python``, or None if it cannot be read."""
    try:
        out = subprocess.run(
            [python, "-c", "import importlib.metadata as m; print(m.version('etsam'))"],
            capture_output=True,
            text=True,
            timeout=120,
            **subprocess_options(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def _release(version: str | None) -> tuple[int, ...] | None:
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version or "")
    return tuple(int(part) for part in match.groups()) if match else None


def needs_update(plugin_version: str, installed_version: str | None) -> bool:
    """
    Whether the installed ETSAM is older than this plugin release. Development builds
    of the plugin (such as 2.0.1.dev3) are not checked: there is no release to update to.
    """
    if not re.fullmatch(r"\d+\.\d+\.\d+", plugin_version or ""):
        return False
    installed = _release(installed_version)
    return installed is None or installed < _release(plugin_version)


def update(python: str, log=None) -> None:
    """
    Reinstall the installer-made ETSAM environment of ``python`` with the latest release
    (as install.sh and install.ps1 do for a reinstall).
    """
    prefix = os.path.dirname(python) if sys.platform == "win32" else os.path.dirname(os.path.dirname(python))
    if not os.path.isfile(os.path.join(prefix, MARKER)):
        raise RuntimeError(
            f"ETSAM in {prefix} was not set up by the ETSAM installer, so it cannot be "
            f"updated from here. Update it the way it was installed (see {INSTALL_DOCS})."
        )
    workdir = _workdir()
    try:
        installer = _installer(workdir, log)
        _reinstall(installer, prefix, python, log)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _reinstall(installer: str, prefix: str, python: str, log) -> None:
    """Remove the installation in ``prefix`` and install it again, with the model weights
    moved aside meanwhile (a rename next to the folder: no copy, no extra disk space)."""
    checkpoints = os.path.join(prefix, "checkpoints")
    keep = prefix + ".checkpoints-keep"
    if os.path.isdir(checkpoints):
        shutil.rmtree(keep, ignore_errors=True)
        os.replace(checkpoints, keep)
    if log:
        log("Removing the old ETSAM installation…")
    uninstaller = os.path.join(prefix, "Uninstall-ETSAM.exe")
    if os.path.exists(uninstaller):
        # Windows: also removes the commands from the PATH and the list of apps.
        # /_?= (the last argument, unquoted) makes it run in this folder and wait.
        subprocess.run(f'"{uninstaller}" /S /_?={prefix}', **subprocess_options())
    shutil.rmtree(prefix, ignore_errors=True)  # what is left, e.g. the uninstaller
    if os.path.exists(prefix):
        if os.path.isdir(keep) and not os.path.exists(checkpoints):
            os.replace(keep, checkpoints)
        raise RuntimeError(
            f"Could not remove the old ETSAM installation in {prefix}. Close the "
            "programs that use ETSAM (such as terminals running etsam) and try again."
        )
    if log:
        log("Installing ETSAM (this downloads PyTorch, about 3 GB, and takes a few minutes)…")
    _run_installer(installer, prefix, log,
                   environment={"ETSAM_SKIP_WEIGHTS": "1"} if os.path.isdir(keep) else None)
    if os.path.isdir(keep):
        shutil.rmtree(checkpoints, ignore_errors=True)
        os.replace(keep, checkpoints)
        # The new release reuses the kept weights, and downloads the ones it needs that
        # are not there: new files, or files from a new download URL.
        environment = {
            "ETSAM_CHECKPOINT_DIR": checkpoints,
            "PATH": os.pathsep.join([
                prefix, os.path.join(prefix, "Library", "mingw-w64", "bin"),
                os.path.join(prefix, "Library", "usr", "bin"), os.path.join(prefix, "Library", "bin"),
                os.environ.get("PATH", ""),
            ]),
        }
        _run([python, "-I", "-c", "import etsam; etsam.ensure_checkpoints()"], log,
             environment=environment, name="Checking the model weights",
             message="Checking the model weights")


# --------------------------------------------------------------------------- #
# Installing
# --------------------------------------------------------------------------- #


def installer_platform() -> str | None:
    """
    Platform part of the ETSAM installer's file name (ETSAM-<version>-<platform>) for this
    computer, or None if there is none.
    """
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        return None
    if sys.platform.startswith("linux"):
        return "Linux-x86_64.sh"
    if sys.platform == "win32":
        return "Windows-x86_64.exe"
    return None  # macOS and other platforms: not supported by the installer yet


def can_install() -> bool:
    return bool(os.environ.get("ETSAM_INSTALLER") or installer_platform())


def latest_installer_url(platform_name: str) -> str:
    """
    Download URL of the ETSAM installer of the latest release. The files carry the version
    in their names (ETSAM-1.2.3-Linux-x86_64.sh), so the latest release is looked up first:
    .../releases/latest redirects to .../releases/tag/v1.2.3.
    """
    try:
        request = urllib.request.Request(f"{RELEASES}/latest", method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            tag = response.url.rstrip("/").rsplit("/", 1)[-1]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(
                f"The ETSAM release repository is unavailable at {RELEASES}. "
                "It may be private, moved, or have no published release. "
                "Automatic installation requires a publicly accessible release. "
                "For a local installer, set ETSAM_INSTALLER to its file path before "
                "starting the application."
            ) from e
        raise RuntimeError(
            f"GitHub could not provide the latest ETSAM release (HTTP {e.code}). "
            f"Check {RELEASES} or try again later."
        ) from e
    except OSError as e:  # includes URLError
        raise RuntimeError(
            f"Could not find the latest ETSAM release ({e}). Check your internet connection."
        ) from e
    if "/releases/tag/" not in response.url:
        raise RuntimeError(f"Could not find the latest ETSAM release at {RELEASES}/latest.")
    return f"{RELEASES}/download/{tag}/ETSAM-{tag.removeprefix('v')}-{platform_name}"


def default_prefix() -> str:
    prefix = os.path.join(os.path.expanduser("~"), "etsam_env")
    if sys.platform == "win32" and " " in prefix:
        # conda environments must not contain spaces on Windows
        prefix = os.path.join(os.environ.get("SYSTEMDRIVE", "C:") + os.sep, "etsam_env")
    return prefix


def install(log=None, prefix: str | None = None) -> str:
    """
    Download the ETSAM installer and run it silently; return the new Python.

    ``log(message)`` receives progress. ETSAM_INSTALLER may name another
    installer (a URL or a local file). Raises RuntimeError with the installer's
    output if it fails.
    """
    prefix = prefix or default_prefix()
    if os.path.exists(prefix):
        raise RuntimeError(
            f"Cannot install ETSAM into {prefix}: the folder already exists. Remove it, or "
            f"install ETSAM yourself (see {INSTALL_DOCS}), then try again."
        )
    workdir = _workdir()
    try:
        installer = _installer(workdir, log)
        if log:
            log("Installing ETSAM (this downloads PyTorch, about 3 GB, and takes a few minutes)…")
        _run_installer(installer, prefix, log)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return env_python(prefix)


def _workdir() -> str:
    cache = os.path.join(os.path.expanduser("~"), ".cache", "etsam-installer")
    os.makedirs(cache, exist_ok=True)
    return tempfile.mkdtemp(dir=cache)


def _installer(workdir: str, log) -> str:
    """The installer of the latest release for this computer, or ETSAM_INSTALLER."""
    source = os.environ.get("ETSAM_INSTALLER")
    if not source:
        name = installer_platform()
        if name is None:
            raise RuntimeError(
                f"The ETSAM installer does not support this computer yet. See {INSTALL_DOCS} "
                "for other ways to install ETSAM."
            )
        if log:
            log("Looking up the latest ETSAM release…")
        source = latest_installer_url(name)
    return _fetch(source, workdir, log)


def _run_installer(installer: str, prefix: str, log, environment=None) -> None:
    """Run the installer silently into the new folder ``prefix``."""
    if installer.endswith(".exe"):
        # NSIS: /D must come last and must not be quoted.
        cmd = (
            f'"{installer}" /S /Q /InstallationType=JustMe /AddToPath=0 '
            f"/RegisterPython=0 /D={prefix}"
        )
    else:
        cmd = ["bash", installer, "-b", "-p", prefix]
    _run(cmd, log, prefix=prefix, environment=environment)
    if not os.path.exists(env_python(prefix)) or not os.path.isfile(os.path.join(prefix, MARKER)):
        raise RuntimeError(
            "The ETSAM installer exited without completing setup. "
            f"See {os.path.join(prefix, '.step.log')} for details."
        )


def _fetch(source: str, workdir: str, log) -> str:
    if os.path.exists(source):  # a local installer file
        return os.path.abspath(source)
    path = os.path.join(workdir, source.rsplit("/", 1)[-1])
    if log:
        log("Downloading the ETSAM installer…")
    with urllib.request.urlopen(source, timeout=60) as response, open(path, "wb") as f:
        total = int(response.headers.get("Content-Length") or 0)
        done, last = 0, -1
        while chunk := response.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            percent = 100 * done // total if total else -1
            if log and total and percent // 10 != last // 10:
                last = percent
                log(f"Downloading the ETSAM installer: {percent}%")
    return path


def _run(cmd, log, prefix=None, environment=None, name="The ETSAM installer",
         message="Installing ETSAM: extracting and preparing the environment") -> None:
    # NSIS buffers script output until a step finishes. /Q prevents its own console;
    # read .step.log as well so PyTorch and weight-download progress reaches the UI live.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        **subprocess_options(environment),
    )
    lines = queue.Queue()

    def read_output():
        try:
            for line in proc.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    threading.Thread(target=read_output, name="ETSAM installer output", daemon=True).start()
    output = deque(maxlen=100)
    started = last_update = time.monotonic()
    last_message = message
    step_path = os.path.join(prefix, ".step.log") if prefix and sys.platform == "win32" else None
    step_file = None
    step_pending = ""
    stream_done = False
    checkpoint_stage = None

    def report(line):
        nonlocal last_message, last_update, checkpoint_stage
        line = line.strip()
        if not line:
            return
        output.append(line)
        if line.startswith("ETSAM:"):
            checkpoint_stage = None
            message = line.split(":", 1)[1].strip()
            message = message[:1].upper() + message[1:]
        elif line.startswith("==> Downloading"):
            checkpoint_stage = next(
                (stage for stage in (1, 2) if f"etsam_stage{stage}_" in line), None
            )
            message = (
                f"Downloading ETSAM Stage {checkpoint_stage} of 2 Checkpoint: starting…"
                if checkpoint_stage else line
            )
        elif "MB" in line and "%" in line:
            message = (
                f"Downloading ETSAM Stage {checkpoint_stage} of 2 Checkpoint: {line}"
                if checkpoint_stage else line
            )
        elif line.startswith(("Collecting package metadata", "Solving environment",
                              "Preparing transaction", "Executing transaction")):
            message = line
        else:
            return
        if message != last_message:
            last_message, last_update = message, time.monotonic()
            if log:
                log(message)

    if log:
        log(last_message + "…")
    try:
        while not stream_done or proc.poll() is None:
            try:
                line = lines.get(timeout=1)
                if line is None:
                    stream_done = True
                elif step_path is None:
                    report(line)
                else:
                    output.append(line.rstrip())  # Windows progress comes from the live log
            except queue.Empty:
                pass
            if step_path:
                try:
                    if step_file is None:
                        step_file = open(step_path, encoding="utf-8", errors="replace")
                    if os.path.getsize(step_path) < step_file.tell():
                        step_file.seek(0)  # NSIS reuses the file for each installer step
                        step_pending = ""
                    step_pending += step_file.read()
                    entries = step_pending.split(chr(10))
                    step_pending = entries.pop()
                    for entry in entries:
                        report(entry)
                except OSError:
                    pass  # the log is created after extraction and may be locked briefly
            if log and time.monotonic() - last_update >= 15:
                elapsed = int(time.monotonic() - started)
                log(f"{last_message} (setup running for {elapsed // 60}m {elapsed % 60:02d}s)")
                last_update = time.monotonic()
    finally:
        if step_file is not None:
            step_file.close()
        proc.stdout.close()
    if proc.wait() != 0:
        reasons = [
            line.split(":", 1)[1].strip() if line.startswith("ETSAM:") else line
            for line in output
            if (line.startswith("ETSAM:") or "Error" in line or "ERROR" in line)
            and not line.startswith("ERROR: executing")
        ]
        summary = reasons[-1] if reasons else "see the details"
        raise RuntimeError(
            f"{name} failed: {summary}\n\n" + "\n".join(list(output)[-25:])
        )


def subprocess_options(environment=None) -> dict:
    """Environment without the host application's Python settings (plus ``environment``),
    and no console windows."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "VIRTUAL_ENV"}
    env.update(environment or {})
    env["PYTHONUNBUFFERED"] = "1"
    options = {"env": env}
    if sys.platform == "win32":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    return options
