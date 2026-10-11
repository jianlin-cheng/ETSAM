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
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request

INSTALL_DOCS = "https://jianlin-cheng.github.io/ETSAM/installation/"
RELEASES = "https://github.com/jianlin-cheng/ETSAM/releases"
ENV_NAME = "etsam"
MARKER = ".etsam-installer"  # written by the installer's post-install script
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "etsam_worker.py")
NOT_FOUND = (
    f"ETSAM is not installed (no '{ENV_NAME}' conda environment was found). Install it with "
    f"the ETSAM installer, see {INSTALL_DOCS} , then try again. (To use a specific "
    "environment, set ETSAM_PYTHON to its Python executable.)"
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

    cache = os.path.join(os.path.expanduser("~"), ".cache", "etsam-installer")
    os.makedirs(cache, exist_ok=True)
    workdir = tempfile.mkdtemp(dir=cache)
    try:
        installer = _fetch(source, workdir, log)
        if log:
            log("Installing ETSAM (this downloads PyTorch, about 3 GB, and takes a few minutes)…")
        if installer.endswith(".exe"):
            # NSIS: /D must come last and must not be quoted.
            cmd = (
                f'"{installer}" /S /InstallationType=JustMe /AddToPath=0 '
                f"/RegisterPython=0 /D={prefix}"
            )
        else:
            cmd = ["bash", installer, "-b", "-p", prefix]
        _run(cmd, log)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    python = env_python(prefix)
    if not os.path.exists(python):
        raise RuntimeError(f"The ETSAM installer finished, but {python} does not exist.")
    return python


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


def _run(cmd, log) -> None:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        **subprocess_options(),
    )
    output = []
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            output.append(line)
            if log and line.startswith("ETSAM:"):  # the installer's own messages
                message = line.split(":", 1)[1].strip()
                log(message[:1].upper() + message[1:])
    if proc.wait() != 0:
        # The last informative line: an ETSAM message or an error (not the
        # installer's generic "ERROR: executing post_install.sh failed").
        reasons = [
            line.split(":", 1)[1].strip() if line.startswith("ETSAM:") else line
            for line in output
            if (line.startswith("ETSAM:") or "Error" in line or "ERROR" in line)
            and not line.startswith("ERROR: executing")
        ]
        summary = reasons[-1] if reasons else "see the details"
        raise RuntimeError(f"The ETSAM installer failed: {summary}\n\n" + "\n".join(output[-25:]))


def subprocess_options() -> dict:
    """Environment without the host application's Python settings, and no console windows."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "VIRTUAL_ENV"}
    env["PYTHONUNBUFFERED"] = "1"
    options = {"env": env}
    if sys.platform == "win32":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    return options
