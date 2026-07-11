#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors
#
# Install LeRobot + the Feetech servo SDK on a Raspberry Pi 5 (or any Debian/
# aarch64 host) so it can talk to a LeKiwi's STS3215 servo bus. This is the
# software the SO-101 / LeKiwi tutorial installs, plus the low-level serial SDK
# the standalone auto-setup (`lekiwi_autosetup.py`) drives directly.
#
# It is idempotent: re-running skips work already done. It installs into a
# Miniforge conda env named `lerobot` (Python 3.10), matching the SeeedStudio
# wiki. Nothing here moves the robot — that is `lekiwi_autosetup.py`.
#
# Usage (ON THE PI):
#   bash lekiwi_pi_install.sh
# then:
#   conda activate lerobot
#   lerobot-find-port          # discover the servo bus device
#   python lekiwi_autosetup.py --port /dev/ttyACM0          # read-only check
#   python lekiwi_autosetup.py --port /dev/ttyACM0 --execute  # calibrate (moves)

set -euo pipefail

ENV_NAME="${ENV_NAME:-lerobot}"
PY_VERSION="3.10"

log() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# --------------------------------------------------------------------------- #
log "System check"
uname -a
if ! command -v sudo >/dev/null 2>&1; then
  echo "note: 'sudo' not found; serial-permission and apt steps may need root."
fi

# --------------------------------------------------------------------------- #
log "Base build tools (ffmpeg, libffi, git)"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y git ffmpeg libffi-dev build-essential
else
  echo "skip: no apt-get (not Debian/Ubuntu) — ensure git + ffmpeg are present."
fi

# --------------------------------------------------------------------------- #
log "Miniforge / conda"
if ! command -v conda >/dev/null 2>&1; then
  if [ ! -d "$HOME/miniforge3" ]; then
    arch="$(uname -m)"   # aarch64 on a Pi 5
    url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh"
    echo "installing Miniforge for ${arch} from ${url}"
    curl -fsSL "$url" -o /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  fi
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# --------------------------------------------------------------------------- #
log "Conda env '${ENV_NAME}' (Python ${PY_VERSION})"
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  conda create -y -n "$ENV_NAME" "python=${PY_VERSION}"
fi
conda activate "$ENV_NAME"
python -V

# --------------------------------------------------------------------------- #
log "LeRobot + Feetech driver + serial SDK"
# ffmpeg via conda-forge (LeRobot dataset/video tooling expects it on PATH).
conda install -y -c conda-forge ffmpeg || echo "ffmpeg via conda skipped (already present?)"
python -m pip install --upgrade pip
# LeRobot's Feetech extra pulls the motor-bus driver; we also install the
# low-level SDK + pyserial explicitly because the auto-setup speaks to the bus
# directly (so it does not depend on LeRobot's higher-level API surface).
python -m pip install "lerobot[feetech]" || python -m pip install lerobot
python -m pip install feetech-servo-sdk pyserial

# --------------------------------------------------------------------------- #
log "Serial port permissions"
# Let the user open /dev/ttyACM* and /dev/ttyUSB* without sudo.
if getent group dialout >/dev/null 2>&1; then
  sudo usermod -aG dialout "$USER" || true
  echo "added $USER to 'dialout' (log out/in for it to take effect)."
fi

# --------------------------------------------------------------------------- #
log "Verify imports"
python - <<'PY'
import importlib
ok = True
for mod in ("lerobot", "scservo_sdk", "serial"):
    try:
        importlib.import_module(mod)
        print(f"  ok: {mod}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  MISSING: {mod} ({e})")
print("imports ok" if ok else "imports INCOMPLETE — see above")
PY

log "Done"
cat <<'EOF'
Next steps (with the LeKiwi powered and its servo bus plugged into the Pi):

  conda activate lerobot
  lerobot-find-port                                   # find the bus device
  python lekiwi_autosetup.py --port /dev/ttyACM0      # READ-ONLY: discover + health
  # only after the read-only check looks right, and with a hand on the power:
  python lekiwi_autosetup.py --port /dev/ttyACM0 --execute   # calibrate (MOVES the arm)
EOF
