#!/bin/sh
# Install ETSAM on Linux or macOS (Apple silicon) with one command:
#
#   curl -fsSL https://jianlin-cheng.github.io/ETSAM/install.sh | sh
#
# Downloads the ETSAM installer (ETSAM-<version>-Linux-x86_64.sh, or
# ETSAM-<version>-macOS-arm64.sh on a Mac) from the latest GitHub release and runs
# it without questions: ETSAM goes into ~/etsam_env, with PyTorch
# and the model weights, and the etsam commands are added to the PATH. The
# installer is self-extracting, so it is saved to a temporary file and run from
# there (it cannot be piped into sh itself).
#
# If ETSAM is already installed (found through ~/.conda/environments.txt, the
# list of conda environments, and the .etsam-installer file the installer leaves
# in its folder), the script says so and asks whether to upgrade (the latest
# release is newer) or reinstall (it is the same version). An upgrade updates the
# installation in place; a reinstall removes it and installs it again. The
# downloaded model weights are kept in both cases, and downloaded again only if
# their download URL changed.
#
# Environment variables:
#   ETSAM_PREFIX     install folder (default: ~/etsam_env). Without it, a new
#                    installation asks for the folder, showing the default.
#   ETSAM_INSTALLER  another installer to run: a URL or a local file
#   ETSAM_YES        1 to not ask anything: use the default folder, and upgrade or
#                    reinstall when ETSAM is installed already
set -eu

REPO="https://github.com/jianlin-cheng/ETSAM"
PREFIX_SET="${ETSAM_PREFIX:-}"
PREFIX="${ETSAM_PREFIX:-$HOME/etsam_env}"
MARKER=".etsam-installer"   # written by the installer; first word: its version

case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) PLATFORM="Linux-x86_64" ;;
    # uname -m says x86_64 in a Rosetta shell on Apple silicon: ask the hardware.
    Darwin-*) [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ] && PLATFORM="macOS-arm64" ;;
esac
if [ -z "${PLATFORM:-}" ]; then
    echo "ETSAM: this installer is for Linux on x86-64 computers and Macs with Apple silicon." >&2
    echo "       For other systems, see https://jianlin-cheng.github.io/ETSAM/installation/" >&2
    exit 1
fi

# --- Which installer? ----------------------------------------------------------
# The release files have the version in their names (ETSAM-1.2.3-Linux-x86_64.sh), so
# the latest release is looked up first: .../releases/latest redirects to
# .../releases/tag/v1.2.3.
latest_tag() {
    url=""
    if command -v curl >/dev/null 2>&1; then
        url="$(curl -fsSLI -o /dev/null -w '%{url_effective}' "$REPO/releases/latest" 2>/dev/null || true)"
    elif command -v wget >/dev/null 2>&1; then
        url="$(wget -q --server-response --spider --max-redirect=0 "$REPO/releases/latest" 2>&1 \
            | sed -n 's/^ *[Ll]ocation: *//p' | head -n 1 | tr -d '\r')"
    fi
    case "$url" in
        */releases/tag/*) echo "${url##*/}" ;;
    esac
}

LATEST_TAG=""
if [ -n "${ETSAM_INSTALLER:-}" ]; then
    SOURCE="$ETSAM_INSTALLER"
else
    LATEST_TAG="$(latest_tag)"
    if [ -z "$LATEST_TAG" ]; then
        echo "ETSAM: could not find the latest release at $REPO/releases/latest (no internet, or no curl or wget?)." >&2
        echo "       Download the installer by hand, see https://jianlin-cheng.github.io/ETSAM/installation/" >&2
        exit 1
    fi
    SOURCE="$REPO/releases/download/$LATEST_TAG/ETSAM-${LATEST_TAG#v}-$PLATFORM.sh"
fi

# --- Is ETSAM installed already? ---------------------------------------------
# The folder we would install into, or (if no folder was chosen) a conda
# environment listed in environments.txt, counts when it has the installer's marker
# file. Listed folders that no longer exist, or are not made by this installer, do not.
find_installed() {
    if [ -f "$PREFIX/$MARKER" ]; then
        echo "$PREFIX"
        return
    fi
    [ -z "$PREFIX_SET" ] && [ -f "$HOME/.conda/environments.txt" ] || return 0
    while IFS= read -r dir; do
        if [ -n "$dir" ] && [ -f "$dir/$MARKER" ]; then
            echo "$dir"
            return
        fi
    done < "$HOME/.conda/environments.txt"
}

# Version of the installer we would run, or nothing if it cannot be told.
latest_version() {
    if [ -f "$SOURCE" ]; then
        # the installer's header (a text script at the start of the file) has it
        head -n 200 "$SOURCE" | sed -n "s/^export INSTALLER_VER='\(.*\)'.*/\1/p" | head -n 1
    elif [ -n "$LATEST_TAG" ]; then
        echo "${LATEST_TAG#v}"
    fi
}

# Version for "sort -V": a pre-release (1.2.0rc1) sorts before its release (1.2.0).
sort_key() {
    # (sed -E: BSD sed on macOS has no \| in basic regular expressions)
    printf '%s\n' "$1" | sed -E 's/([0-9])(a|b|rc)([0-9])/\1~\2\3/'
}

# Asks where to install a new ETSAM (on the terminal: stdin is the script itself when
# run as "curl ... | sh"). Enter keeps the default; without a terminal, or with
# ETSAM_YES=1, the default is used.
ask_prefix() {
    [ "${ETSAM_YES:-0}" = "1" ] && return 0
    ( exec </dev/tty ) 2>/dev/null || return 0
    while :; do
        printf 'ETSAM: install folder [%s] (press Enter to accept, or type another folder): ' "$PREFIX" > /dev/tty
        read -r ANSWER < /dev/tty || ANSWER=""
        case "$ANSWER" in
            "") return 0 ;;
            "~") ANSWER="$HOME" ;;
            "~/"*) ANSWER="$HOME/${ANSWER#"~/"}" ;;
        esac
        case "$ANSWER" in
            /*) ;;
            *) ANSWER="$PWD/$ANSWER" ;;
        esac
        case "$ANSWER" in
            *" "*) echo "ETSAM: the folder cannot contain spaces. Try another." > /dev/tty; continue ;;
        esac
        PREFIX="$ANSWER"
        return 0
    done
}

ACTION=install
INSTALLED="$(find_installed)"
if [ -z "$INSTALLED" ] && [ -z "$PREFIX_SET" ]; then
    ask_prefix
    PREFIX_SET="$PREFIX"   # the chosen folder is the only one to check from here on
    INSTALLED="$(find_installed)"
fi
if [ -n "$INSTALLED" ]; then
    PREFIX="$INSTALLED"
    OLD_VERSION="$(head -n 1 "$PREFIX/$MARKER" | cut -d ' ' -f 1)"
    NEW_VERSION="$(latest_version)"
    if [ -z "$NEW_VERSION" ]; then
        ACTION=reinstall
        echo "ETSAM $OLD_VERSION is already installed in $PREFIX."
        QUESTION="Reinstall it? [y/N]"
    elif [ "$NEW_VERSION" = "$OLD_VERSION" ]; then
        ACTION=reinstall
        echo "ETSAM $OLD_VERSION (the latest version) is already installed in $PREFIX."
        QUESTION="Reinstall it? [y/N]"
    elif [ "$(printf '%s\n%s\n' "$(sort_key "$OLD_VERSION")" "$(sort_key "$NEW_VERSION")" | sort -V | head -n 1)" = "$(sort_key "$OLD_VERSION")" ]; then
        ACTION=upgrade
        echo "ETSAM $OLD_VERSION is installed in $PREFIX. The latest version is $NEW_VERSION."
        QUESTION="Upgrade to $NEW_VERSION? [Y/n]"
    else
        ACTION=reinstall
        echo "ETSAM $OLD_VERSION is installed in $PREFIX, which is newer than the latest release ($NEW_VERSION)."
        QUESTION="Reinstall it as version $NEW_VERSION? [y/N]"
    fi
    echo "The downloaded model weights are kept (they are downloaded again only if their URL changed)."

    if [ "${ETSAM_YES:-0}" = "1" ]; then
        echo "ETSAM: ETSAM_YES=1, so continuing with the $ACTION."
    elif ( exec </dev/tty ) 2>/dev/null; then
        # stdin is the script itself when run as "curl ... | sh": ask on the terminal
        printf '%s ' "$QUESTION" > /dev/tty
        read -r ANSWER < /dev/tty || ANSWER=""
        case "$ACTION:$ANSWER" in
            upgrade:|upgrade:[Yy]*|reinstall:[Yy]*) ;;
            *) echo "ETSAM: nothing changed."; exit 0 ;;
        esac
    else
        echo "ETSAM: no terminal to ask on. Run this again in a terminal, or set ETSAM_YES=1 to $ACTION without asking." >&2
        exit 1
    fi
elif [ -e "$PREFIX" ]; then
    echo "ETSAM: $PREFIX already exists, but it was not made by the ETSAM installer." >&2
    echo "       Choose another folder with ETSAM_PREFIX, or delete it first (rm -r \"$PREFIX\")." >&2
    exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
INSTALLER="$TMP_DIR/ETSAM-$PLATFORM.sh"

if [ -f "$SOURCE" ]; then
    cp "$SOURCE" "$INSTALLER"
else
    echo "ETSAM: downloading the installer from $SOURCE"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --progress-bar -o "$INSTALLER" "$SOURCE"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$INSTALLER" "$SOURCE"
    else
        echo "ETSAM: neither curl nor wget is installed; install one of them first." >&2
        exit 1
    fi
fi

# The model weights are moved aside while the old installation is removed (a rename
# next to the installation folder: no copy, no extra disk space), and moved back
# into the new one before ETSAM checks them. KEEP also survives a failed run.
KEEP="$PREFIX.checkpoints-keep"
if [ "$ACTION" = "reinstall" ]; then
    echo "ETSAM: removing the old installation in $PREFIX (keeping the model weights)"
    if [ -d "$PREFIX/checkpoints" ]; then
        rm -rf "$KEEP"
        mv "$PREFIX/checkpoints" "$KEEP"
    fi
    rm -rf "$PREFIX"
fi

USER_SKIPS_WEIGHTS="${ETSAM_SKIP_WEIGHTS:-0}"
if [ -d "$KEEP" ]; then
    export ETSAM_SKIP_WEIGHTS=1   # restored and checked below, instead of downloaded
fi

case "$ACTION" in
    upgrade)   echo "ETSAM: upgrading $PREFIX in place (this downloads the new version of ETSAM)"; UPDATE="-u" ;;
    reinstall) echo "ETSAM: reinstalling into $PREFIX (this downloads PyTorch)"; UPDATE="" ;;
    *)         echo "ETSAM: installing into $PREFIX (this downloads PyTorch and the model weights)"; UPDATE="" ;;
esac
# -u: update the installation in place, instead of failing because the folder exists
# shellcheck disable=SC2086
if ! bash "$INSTALLER" -b $UPDATE -p "$PREFIX"; then
    echo "ETSAM: the installation failed." >&2
    [ -d "$KEEP" ] && echo "       The model weights are kept in $KEEP and used by the next installation." >&2
    exit 1
fi

if [ -d "$KEEP" ]; then
    rm -rf "$PREFIX/checkpoints"
    mv "$KEEP" "$PREFIX/checkpoints"
    if [ "$USER_SKIPS_WEIGHTS" != "1" ]; then
        # keeps the weights unless their download URL changed
        ETSAM_CHECKPOINT_DIR="$PREFIX/checkpoints" "$PREFIX/bin/python" -c "import etsam; etsam.ensure_checkpoints()"
    fi
fi
