# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Fetch the NAD SO-101 subset for the external model-completeness validation.

NAD (Neural Actuation Dataset) accompanies NeuralActuator (Dou et al., RSS 2026).
Its SO-101 subset is telemetry from **the same arm this repo calibrates**, logged
by someone else, on someone else's unit, with a payload deliberately varied from
empty to 500 g. That last part is what makes it valuable here: the payload is
*labelled unmodelled physics*, so it is ground truth for the claim
:mod:`jaxterity.sysid.hybrid` makes — that a growing parametric residual means
something the model does not represent is present.

Only the SO-101 CSVs are fetched — 100 files, ~24 MB. The full repository is
~169 MB and the Hugging Face mirror additionally carries camera recordings;
neither is needed to score a dynamics model, so neither is downloaded.

The data is **not vendored**. The upstream repository declares no license (the
GitHub API reports ``license: null``), so there is no redistribution grant, and
jaxterity uses it transiently — like a human downloading it to check — and never
republishes it. The paired test skips unless these files are present.

Source:   https://github.com/Frank-ZY-Dou/Dynamics-Modeling (NeuralActuator/data/so101)
Mirror:   https://huggingface.co/datasets/frankzydou/NAD (adds camera recordings)
Paper:    https://arxiv.org/abs/2607.11734
Citation: Z. Dou et al., "NeuralActuator: Neural Actuation Modeling for Robot
          Dynamics and External Force Perception", RSS 2026.

Run:  python test/sysid/external/fetch_nad.py
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

_REPO = "Frank-ZY-Dou/Dynamics-Modeling"
_TREE = f"https://api.github.com/repos/{_REPO}/git/trees/main?recursive=1"
_RAW = f"https://raw.githubusercontent.com/{_REPO}/main/"
_PREFIX = "NeuralActuator/data/so101/"

DATA_DIR = Path(__file__).resolve().parent / "_nad_data"

#: Expected file count and an aggregate integrity digest over the whole subset:
#: sha256 of each file's relative path followed by its own sha256, in sorted path
#: order. One pin instead of a hundred, and it still fails on any changed byte,
#: any renamed file, and any missing or extra one — which a per-file list would
#: not catch as cleanly (a dropped file simply would not be checked).
EXPECTED_FILES = 100
MANIFEST_SHA256 = "b85dd9e9723462aa7238eda8bd81001e6b419339f830ed34ff932017bd65fe54"


def manifest_digest(root: Path) -> tuple[int, str]:
    """``(file count, aggregate sha256)`` over ``root``'s CSVs."""
    files = sorted(root.rglob("*.csv"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return len(files), digest.hexdigest()


def present_and_valid(root: Path = DATA_DIR) -> bool:
    """Whether the subset is already downloaded and matches its pins."""
    target = root / "so101"
    if not target.is_dir():
        return False
    count, digest = manifest_digest(target)
    return count == EXPECTED_FILES and digest == MANIFEST_SHA256


def _paths() -> list[str]:
    with urllib.request.urlopen(_TREE, timeout=60) as response:
        tree = json.load(response)
    if tree.get("truncated"):
        raise RuntimeError(
            "fetch_nad: GitHub truncated the tree listing, so the file list is "
            "incomplete. Clone the repository instead and copy "
            f"{_PREFIX} into {DATA_DIR}."
        )
    return sorted(
        e["path"]
        for e in tree["tree"]
        if e["type"] == "blob" and e["path"].startswith(_PREFIX)
    )


def main() -> int:
    if present_and_valid():
        print(f"NAD SO-101 subset already present and verified in {DATA_DIR}")
        return 0

    print(f"listing {_PREFIX} in {_REPO} ...")
    try:
        paths = _paths()
    except Exception as exc:  # network, rate limit, layout change
        print(f"ABORT: could not list the dataset: {type(exc).__name__}: {exc}")
        return 2
    print(f"downloading {len(paths)} CSVs (~24 MB) into {DATA_DIR} ...")

    def fetch(path: str) -> None:
        destination = DATA_DIR / Path(path).relative_to("NeuralActuator/data")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            return
        with urllib.request.urlopen(
            _RAW + urllib.parse.quote(path), timeout=120
        ) as response:
            destination.write_bytes(response.read())

    try:
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            list(pool.map(fetch, paths))
    except Exception as exc:
        print(f"ABORT: download failed: {type(exc).__name__}: {exc}")
        return 2

    count, digest = manifest_digest(DATA_DIR / "so101")
    if count != EXPECTED_FILES or digest != MANIFEST_SHA256:
        print(
            "ABORT: integrity check failed.\n"
            f"  files    expected {EXPECTED_FILES}, got {count}\n"
            f"  manifest expected {MANIFEST_SHA256}\n"
            f"           got      {digest}\n"
            "The upstream data changed, or the download was incomplete. Delete "
            f"{DATA_DIR} and retry; if it still differs, the pins above need "
            "updating deliberately — do not relax them to make a run pass."
        )
        return 3
    print(f"verified {count} files against the pinned manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
