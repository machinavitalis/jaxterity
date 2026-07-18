# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Fetch the EMPS benchmark dataset for the external DC-motor validation.

EMPS (Electro-Mechanical Positioning System) is a real lab rig — a DC motor
driving a low-friction ball-screw stage — logged at 1 kHz. It is the closest
publicly available *real-hardware* match to the ``sysid.actuator`` DC-motor
recipe, whose model ``V = Ks*sgn(v) + Kv*v + Ka*a`` is exactly the rig's
``F = M*a + Fv*v + Fc*sgn(v) (+ offset)`` friction+inertia relation.

This script downloads the dataset **locally, on demand**. The data is NOT
vendored into the repo: its license is unclear (the nonlinearbenchmark.org page
carries only a liability disclaimer and a cite-the-paper request — no
redistribution grant), so jaxterity uses it transiently for validation, like a
human downloading it to check, and never redistributes it. The paired test
(``test_emps_dc_motor.py``) skips unless these files are already present.

Source:   https://www.nonlinearbenchmark.org/benchmarks/emps
Citation: A. Janot, M. Gautier, M. Brunot, "Data Set and Reference Models of
          EMPS", 2019 (Nonlinear System Identification Benchmarks).

Run:  python test/sysid/external/fetch_emps.py
"""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

# Google-Drive file id for the EMPS .zip (see the benchmark page above).
_GDRIVE_ID = "1zwoXYa9-3f8NQ0ohzmjpF7UxbNgRTHkS"
_URL = f"https://drive.google.com/uc?export=download&id={_GDRIVE_ID}"

DATA_DIR = Path(__file__).resolve().parent / "_emps_data"

# Integrity pins for the two .mat members we use (sha256).
_MEMBERS = {
    "DATA_EMPS.mat": "6cf6814ae14618795da0101aae2bb97ff0c6351b0501b5492751b02b07facf7e",
    "DATA_EMPS_PULSES.mat": "0eab75bb1f6b16228ee78791343a88bf9bfd9a51036ac9807d7b5fc11c0e1704",  # noqa: E501
}


def _present_and_valid() -> bool:
    for name, want in _MEMBERS.items():
        f = DATA_DIR / name
        if not f.exists():
            return False
        if hashlib.sha256(f.read_bytes()).hexdigest() != want:
            return False
    return True


def fetch(force: bool = False) -> Path:
    """Download + extract the two EMPS .mat files into ``DATA_DIR``; return it."""
    if not force and _present_and_valid():
        print(f"EMPS data already present in {DATA_DIR}")
        return DATA_DIR
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading EMPS dataset from {_URL}")
    with urllib.request.urlopen(_URL) as resp:  # noqa: S310 (fixed, trusted URL)
        blob = resp.read()
    if not blob[:2] == b"PK":
        raise RuntimeError(
            "download did not return a zip (Google Drive may have served a "
            "confirm page). Download manually from the benchmark page and unzip "
            f"DATA_EMPS.mat / DATA_EMPS_PULSES.mat into {DATA_DIR}."
        )
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name, want in _MEMBERS.items():
            data = zf.read(name)
            got = hashlib.sha256(data).hexdigest()
            if got != want:
                raise RuntimeError(f"{name}: sha256 {got} != expected {want}")
            (DATA_DIR / name).write_bytes(data)
            print(f"  wrote {name} ({len(data)} bytes, sha256 ok)")
    return DATA_DIR


if __name__ == "__main__":
    fetch(force="--force" in sys.argv)
