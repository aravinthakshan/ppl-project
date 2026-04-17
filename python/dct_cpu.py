"""
Python CPU DCT-II (8x8 block, orthonormal) on CIFAR-10 blocks.

Produces JSON metrics to stdout / file.

Two variants reported:
  - numpy_matmul:  Y = C @ X @ C.T  (same algorithm as C and CUDA => fair compare)
  - scipy_fft:     scipy.fft.dctn(..., type=2, norm='ortho')  (realistic baseline)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.fft

from dct_common import dct_basis, flops_for_block_dct, load_blocks, time_it


def run_matmul(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    # X: (N,8,8), C: (8,8).  Batched matmul.
    return C @ X @ C.T


def run_scipy(X: np.ndarray) -> np.ndarray:
    return scipy.fft.dctn(X, type=2, norm="ortho", axes=(-2, -1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--limit", type=int, default=None, help="cap # blocks for debug")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default=None, help="write json result here")
    args = ap.parse_args()

    X, meta = load_blocks(Path(args.data_dir), limit=args.limit)
    n_blocks = X.shape[0]
    C = dct_basis(8, dtype=np.float32)

    # correctness: matmul vs scipy on a small sample
    sample = X[: min(1024, n_blocks)]
    y_mm = run_matmul(sample, C)
    y_sp = run_scipy(sample).astype(np.float32)
    max_abs_err = float(np.max(np.abs(y_mm - y_sp)))
    mse = float(np.mean((y_mm - y_sp) ** 2))

    results = {
        "impl": "python_cpu",
        "n_blocks": int(n_blocks),
        "block_size": 8,
        "dtype": "float32",
        "flops_per_block": flops_for_block_dct(8),
        "total_flops": flops_for_block_dct(8) * n_blocks,
        "correctness_vs_scipy": {"max_abs_err": max_abs_err, "mse": mse},
    }

    print(f"[python_cpu] blocks={n_blocks}  max|err vs scipy|={max_abs_err:.2e}")

    # NumPy matmul (fair comparison path)
    t = time_it(lambda: run_matmul(X, C), warmup=args.warmup, trials=args.trials)
    t["gflops"] = results["total_flops"] / (t["mean_ms"] * 1e-3) / 1e9
    t["blocks_per_sec"] = n_blocks / (t["mean_ms"] * 1e-3)
    results["numpy_matmul"] = t
    print(f"  numpy matmul:  {t['mean_ms']:.2f} ± {t['std_ms']:.2f} ms  "
          f"[{t['gflops']:.2f} GFLOP/s, {t['blocks_per_sec']/1e6:.2f} M blocks/s]")

    # scipy.fft.dctn (realistic Python baseline)
    t = time_it(lambda: run_scipy(X), warmup=args.warmup, trials=args.trials)
    t["gflops"] = results["total_flops"] / (t["mean_ms"] * 1e-3) / 1e9
    t["blocks_per_sec"] = n_blocks / (t["mean_ms"] * 1e-3)
    results["scipy_fft"] = t
    print(f"  scipy.dctn:    {t['mean_ms']:.2f} ± {t['std_ms']:.2f} ms  "
          f"[{t['gflops']:.2f} GFLOP/s, {t['blocks_per_sec']/1e6:.2f} M blocks/s]")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
