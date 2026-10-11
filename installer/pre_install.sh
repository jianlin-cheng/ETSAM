#!/bin/sh
# ETSAM installer: pre-install check (Linux).
#
# The installer currently supports NVIDIA GPUs only, so it stops early, with a
# clear message, on computers without a working NVIDIA driver. Support for other
# accelerators (AMD ROCm, Intel GPUs, Apple silicon) can be added here and in
# post_install.sh; it needs testing on such hardware, and contributions are
# welcome: https://github.com/jianlin-cheng/ETSAM
#
# Set ETSAM_SKIP_GPU_CHECK=1 to skip this check (e.g. for testing in CI).

if [ "${ETSAM_SKIP_GPU_CHECK:-0}" = "1" ]; then
    echo "ETSAM: skipping the GPU check (ETSAM_SKIP_GPU_CHECK=1)."
    exit 0
fi

echo "ETSAM: checking for an NVIDIA GPU driver..."

# 1. The NVIDIA kernel driver is loaded (most reliable).
if [ -f /proc/driver/nvidia/version ]; then
    echo "ETSAM: NVIDIA driver found: $(head -n 1 /proc/driver/nvidia/version)"
    exit 0
fi

# 2. The NVIDIA kernel module is loaded, but /proc is not populated.
if command -v lsmod >/dev/null 2>&1 && lsmod | grep -q "^nvidia"; then
    echo "ETSAM: NVIDIA driver found (kernel module loaded)."
    exit 0
fi

# 3. NVIDIA hardware without an active driver: probably needs a driver or a reboot.
if command -v lspci >/dev/null 2>&1 && lspci | grep -qi "nvidia"; then
    echo "=========================================================="
    echo "ETSAM: an NVIDIA GPU was found, but its driver is not active."
    echo "Install the NVIDIA driver (or reboot after installing it),"
    echo "then run this installer again."
    echo "=========================================================="
    exit 1
fi

# Placeholders for other accelerators (not supported yet), for example:
#   AMD ROCm:  command -v rocminfo >/dev/null 2>&1 || [ -d /opt/rocm ]
#   Intel GPU: grep -q 0x8086 /sys/class/drm/card*/device/vendor 2>/dev/null

echo "=========================================================="
echo "ETSAM: this installer requires an NVIDIA GPU with a working driver,"
echo "and none was found. Installation stopped."
echo "Other GPUs are not supported by the installer yet; Contributions are welcome."
echo "=========================================================="
exit 1
