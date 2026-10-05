"""Load record embedding matrices.

Matrices larger than GitHub's 100 MB file limit (the WDC Products tables) are stored as
``<name>.part1.npy``, ``<name>.part2.npy``, ... next to the configured ``<name>.npy`` path.
``load_embeddings`` reads the single file if it exists and otherwise concatenates the parts
row-wise in part order, which gives the identical matrix.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_embeddings(path) -> np.ndarray:
    path = Path(path)
    if path.exists():
        return np.load(path)
    parts = sorted(path.parent.glob(f"{path.stem}.part*.npy"), key=lambda p: int(p.stem.rsplit("part", 1)[1]))
    if not parts:
        raise FileNotFoundError(f"No embedding matrix at {path} and no parts {path.stem}.part*.npy")
    return np.concatenate([np.load(part) for part in parts], axis=0)
