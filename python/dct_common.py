"""Shared utilities: DCT basis matrix, timing helper, data loading."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np


def dct_basis(n: int = 8, dtype=np.float32) -> np.ndarray:
    """
    Orthonormal DCT-II basis matrix C such that  Y = C @ X @ C.T
    gives the 2-D orthonormal DCT-II of an n x n block X.

        C[k,i] = alpha(k) * cos( pi * (2i+1) * k / (2n) )
        alpha(0) = sqrt(1/n),  alpha(k>0) = sqrt(2/n)
    """
    i = np.arange(n)
    k = np.arange(n).reshape(-1, 1)
    C = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    alpha = np.full(n, np.sqrt(2.0 / n))
    alpha[0] = np.sqrt(1.0 / n)
    C = C * alpha.reshape(-1, 1)
    return C.astype(dtype)


def load_blocks(data_dir: Path, limit: int | None = None) -> tuple[np.ndarray, dict]:
    """Read data/blocks.bin + meta.json -> (N,8,8) float32 array."""
    meta = json.loads((data_dir / "meta.json").read_text())
    n = meta["n_blocks"]
    b = meta["block_size"]
    if limit is not None:
        n = min(n, limit)
    arr = np.fromfile(
        data_dir / "blocks.bin", dtype=np.float32, count=n * b * b
    ).reshape(n, b, b)
    return arr, meta


def time_it(
    fn: Callable[[], object],
    warmup: int = 3,
    trials: int = 10,
    sync: Callable[[], None] | None = None,
) -> dict:
    """
    Run `fn` `warmup` times (results discarded) then `trials` times, timing each.
    `sync` is called before stopping each timer (for async GPU backends).
    Returns dict with mean_ms, std_ms, min_ms, max_ms, per_trial_ms.
    """
    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()

    samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        samples.append((time.perf_counter() - t0) * 1000.0)

    a = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(a.mean()),
        "std_ms": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "per_trial_ms": a.tolist(),
        "trials": trials,
        "warmup": warmup,
    }


def flops_for_block_dct(n: int = 8) -> int:
    """
    Two 8x8 matmuls per block: each = 2 * n^3 flops (mul + add).
    For n=8: 2 * 2 * 512 = 2048 flops per block.
    (Treating one FMA as 2 flops, standard convention.)
    """
    return 2 * 2 * n * n * n
