"""
Python GPU DCT-II (8x8 block, orthonormal) on CIFAR-10 blocks.

Same matrix form as dct_cpu.py: Y = C @ X @ C.T, using torch.matmul on CUDA
(or MPS on macOS when --backend mps).

Reports two timings:
  - compute_only: data already on device; measures kernel-only.
  - with_transfer: includes host->device upload and device->host download
                   (the cost you actually pay end-to-end).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dct_common import dct_basis, flops_for_block_dct, load_blocks, time_it


def resolve_device(pref: str) -> torch.device:
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA not available on this machine.")
        return torch.device("cuda")
    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS not available on this machine.")
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_sync(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.synchronize
    if device.type == "mps":
        return torch.mps.synchronize
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--backend", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.backend)
    sync = make_sync(device)

    X_np, meta = load_blocks(Path(args.data_dir), limit=args.limit)
    n_blocks = X_np.shape[0]
    C_np = dct_basis(8, dtype=np.float32)

    X_host = torch.from_numpy(X_np)  # CPU pinned-by-default-no (we'll pin manually)
    if device.type == "cuda":
        X_host = X_host.pin_memory()
    X_dev = X_host.to(device, non_blocking=True)
    C_dev = torch.from_numpy(C_np).to(device)
    if sync: sync()

    results = {
        "impl": "python_gpu",
        "backend": device.type,
        "n_blocks": int(n_blocks),
        "block_size": 8,
        "dtype": "float32",
        "flops_per_block": flops_for_block_dct(8),
        "total_flops": flops_for_block_dct(8) * n_blocks,
    }

    # compute-only: everything already on device
    def compute():
        # Y = C @ X @ C.T, batched matmul
        out = torch.matmul(torch.matmul(C_dev, X_dev), C_dev.T)
        return out

    t = time_it(compute, warmup=args.warmup, trials=args.trials, sync=sync)
    t["gflops"] = results["total_flops"] / (t["mean_ms"] * 1e-3) / 1e9
    t["blocks_per_sec"] = n_blocks / (t["mean_ms"] * 1e-3)
    results["compute_only"] = t
    print(f"[python_gpu {device.type}] compute only:    "
          f"{t['mean_ms']:.2f} ± {t['std_ms']:.2f} ms  "
          f"[{t['gflops']:.2f} GFLOP/s, {t['blocks_per_sec']/1e6:.2f} M blocks/s]")

    # with transfer: upload + compute + download, per trial
    out_host = torch.empty_like(X_host)

    def compute_with_transfer():
        x_dev = X_host.to(device, non_blocking=True)
        y_dev = torch.matmul(torch.matmul(C_dev, x_dev), C_dev.T)
        out_host.copy_(y_dev, non_blocking=True)

    t = time_it(compute_with_transfer, warmup=args.warmup, trials=args.trials, sync=sync)
    t["gflops"] = results["total_flops"] / (t["mean_ms"] * 1e-3) / 1e9
    t["blocks_per_sec"] = n_blocks / (t["mean_ms"] * 1e-3)
    results["with_transfer"] = t
    print(f"[python_gpu {device.type}] with transfer:   "
          f"{t['mean_ms']:.2f} ± {t['std_ms']:.2f} ms  "
          f"[{t['gflops']:.2f} GFLOP/s, {t['blocks_per_sec']/1e6:.2f} M blocks/s]")

    # correctness vs numpy matmul reference
    ref = (C_np @ X_np[: min(1024, n_blocks)] @ C_np.T)
    got = compute().detach().to("cpu").numpy()[: min(1024, n_blocks)]
    results["correctness_vs_numpy"] = {
        "max_abs_err": float(np.max(np.abs(got - ref))),
        "mse": float(np.mean((got - ref) ** 2)),
    }
    print(f"  max|err vs numpy|={results['correctness_vs_numpy']['max_abs_err']:.2e}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
