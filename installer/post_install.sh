#!/bin/sh
# ETSAM installer: post-install step (Linux and macOS).
#
# The installer ships ETSAM and its dependencies except PyTorch. Here PyTorch is
# installed for the computer's accelerator; each case below installs it its own
# way. All use conda-forge; for NVIDIA GPUs, conda picks the CUDA build that
# matches the installed driver, so no CUDA version needs to be chosen by hand.
# On macOS (Apple silicon), conda-forge's PyTorch includes Metal (MPS) support.
# Other accelerators may need PyTorch's own pip packages instead.
#
# Linux supports NVIDIA GPUs and macOS Apple silicon (pre_install.sh enforces
# this). Other accelerators can be added as new cases below; they need testing
# on such hardware, and contributions are welcome.
#
# ETSAM_ACCELERATOR overrides the choice: "nvidia" (Linux default), "mps" (macOS
# default) or "cpu" (used to test the installer on Linux machines without a GPU,
# e.g. in CI). Afterwards the model weights are downloaded into the installation
# folder.
set -e

# PyTorch version ETSAM is validated with. Pinned so every installation gets
# the same, tested version (without a pin, conda may choose an older build that
# fits the installed libraries). Bump it together with a validation run.
PYTORCH_VERSION="2.7.1"
CONDA="$PREFIX/bin/conda"
if [ "$(uname -s)" = "Darwin" ]; then
    ACCELERATOR="${ETSAM_ACCELERATOR:-mps}"
else
    ACCELERATOR="${ETSAM_ACCELERATOR:-nvidia}"
fi

case "$ACCELERATOR" in
    nvidia)
        echo "ETSAM: installing PyTorch with CUDA support (downloads about 2-3 GB)..."
        "$CONDA" install --yes --quiet --prefix "$PREFIX" --override-channels -c conda-forge \
            "pytorch-gpu=$PYTORCH_VERSION"
        ;;
    mps)
        # macOS has one PyTorch build; it runs on the CPU and the Apple GPU.
        echo "ETSAM: installing PyTorch with Apple silicon GPU (MPS) support (downloads a few hundred MB)..."
        "$CONDA" install --yes --quiet --prefix "$PREFIX" --override-channels -c conda-forge \
            "pytorch=$PYTORCH_VERSION"
        ;;
    cpu)
        echo "ETSAM: installing the CPU-only PyTorch build..."
        "$CONDA" install --yes --quiet --prefix "$PREFIX" --override-channels -c conda-forge \
            "pytorch-cpu=$PYTORCH_VERSION"
        ;;
    # Possible future cases (untested; see the PyTorch installation guide,
    # https://pytorch.org/get-started/locally/). These may install PyTorch's own
    # pip packages into the environment, e.g. "$PREFIX/bin/python" -m pip install ...:
    #   rocm)  AMD GPUs: ROCm build of PyTorch
    #   xpu)   Intel GPUs: XPU build of PyTorch
    *)
        echo "ETSAM: unknown ETSAM_ACCELERATOR '$ACCELERATOR' (use nvidia, mps or cpu)."
        exit 1
        ;;
esac

"$CONDA" clean --yes --all --quiet

# The model weights (1.9 GB), into the installation folder, where ETSAM finds them
# (checkpoints/ in its Python environment). ETSAM_SKIP_WEIGHTS=1 skips this (CI);
# ETSAM then downloads them on first use instead.
if [ "${ETSAM_SKIP_WEIGHTS:-0}" != "1" ]; then
    echo "ETSAM: downloading the model weights (1.9 GB)..."
    ETSAM_CHECKPOINT_DIR="$PREFIX/checkpoints" "$PREFIX/bin/python" -c "import etsam; etsam.ensure_checkpoints()"
fi

# Make the etsam commands available in the terminal. The folder commands/ holds
# only the etsam and etsam-postprocess commands, so putting it on the PATH adds
# nothing else (not this environment's python, pip or conda). Each command runs
# ETSAM with this environment's Python in isolated mode (-I), so that the user's
# Python settings and packages (PYTHONPATH, ~/.local/lib/python...) are not used.
#
# There is no single file that all shells read, so the folder is added in the
# startup files of the common shells:
#   ~/.profile   login shells (sh, dash, bash) and most desktop sessions, so
#                terminals opened from the desktop get it whatever the shell
#   ~/.bashrc    new bash terminals
#   ~/.zshrc     new zsh terminals (zsh does not read ~/.profile), if zsh is used;
#                on macOS always (zsh is its default shell, and a new Mac has no
#                ~/.zshrc yet)
#   ~/.bash_profile   bash login shells on macOS (Terminal opens login shells,
#                and bash then skips ~/.profile), if it exists
#   ~/.config/fish/conf.d/etsam.fish   fish, if fish is used
# Each gets one line, marked "added by the ETSAM installer", that adds the folder
# only if it is not on the PATH yet; installing again replaces the line.
# ETSAM_NO_PATH=1 skips adding it to the PATH.
COMMANDS_DIR="$PREFIX/commands"
mkdir -p "$COMMANDS_DIR"
write_command() {  # $1: command name, $2: Python code that runs it
    printf '#!/bin/sh\n# %s: runs ETSAM with the Python of its own environment (%s).\nexec "%s/bin/python" -I -c "%s" "$@"\n' \
        "$1" "$PREFIX" "$PREFIX" "$2" > "$COMMANDS_DIR/$1"
    chmod +x "$COMMANDS_DIR/$1"
}
write_command etsam "import sys, etsam; sys.argv[0] = 'etsam'; sys.exit(etsam.main())"
write_command etsam-postprocess "import sys, postprocess; sys.argv[0] = 'etsam-postprocess'; sys.exit(postprocess._run_cli())"

if [ "${ETSAM_NO_PATH:-0}" != "1" ]; then
    MARK="added by the ETSAM installer"
    DARWIN=0
    [ "$(uname -s)" = "Darwin" ] && DARWIN=1
    for rc in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.bash_profile"; do
        if [ "$rc" = "$HOME/.zshrc" ] && [ ! -f "$rc" ] && [ "$DARWIN" = 0 ]; then
            continue  # zsh is not used
        fi
        if [ "$rc" = "$HOME/.bash_profile" ] && { [ ! -f "$rc" ] || [ "$DARWIN" = 0 ]; }; then
            continue  # bash reads ~/.profile instead
        fi
        if [ -f "$rc" ]; then
            # Remove the line of an earlier installation (perhaps in another folder).
            grep -v "$MARK" "$rc" > "$rc.etsam-tmp" || true
            cat "$rc.etsam-tmp" > "$rc"
            rm -f "$rc.etsam-tmp"
        fi
        printf 'case ":$PATH:" in *":%s:"*) ;; *) export PATH="%s:$PATH" ;; esac  # %s\n' \
            "$COMMANDS_DIR" "$COMMANDS_DIR" "$MARK" >> "$rc"
    done
    if [ -d "$HOME/.config/fish" ]; then
        mkdir -p "$HOME/.config/fish/conf.d"
        printf '# %s\nif not contains -- "%s" $PATH\n    set -gx PATH "%s" $PATH\nend\n' \
            "$MARK" "$COMMANDS_DIR" "$COMMANDS_DIR" > "$HOME/.config/fish/conf.d/etsam.fish"
    fi
    echo "ETSAM: added $COMMANDS_DIR to the PATH. To use the etsam command, open a new terminal"
    echo "       (if it is not found there, log out and log in again)."
fi

# Marker that the ETSAM plugins (ChimeraX, napari) use to find this environment.
echo "${INSTALLER_VER:-unknown} $ACCELERATOR" > "$PREFIX/.etsam-installer"
echo "ETSAM: installation complete."
