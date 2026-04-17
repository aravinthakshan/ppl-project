"""
Run all four DCT implementations and produce:
  results/results.json        — merged metrics
  results/summary.md          — markdown table
  results/plots/*.png         — bar charts

Skips any implementation that isn't available (e.g. CUDA on macOS).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
PY_DIR = ROOT / "python"
C_DIR = ROOT / "c"
DATA_DIR = ROOT / "data"


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"$ {' '.join(str(c) for c in cmd)}")
    return subprocess.call([str(c) for c in cmd], cwd=str(cwd) if cwd else None)


def load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"  warning: couldn't parse {p}: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap # blocks (for quick smoke tests)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip-python-gpu", action="store_true")
    ap.add_argument("--skip-c-cpu", action="store_true")
    ap.add_argument("--skip-cuda", action="store_true")
    ap.add_argument("--gpu-backend", default="auto",
                    choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)

    meta_path = DATA_DIR / "meta.json"
    if not meta_path.exists():
        print("ERROR: data/meta.json not found. Run `python prepare_data.py` first.")
        sys.exit(1)
    meta = json.loads(meta_path.read_text())
    n_blocks = meta["n_blocks"] if args.limit is None else min(meta["n_blocks"], args.limit)

    merged: dict = {"dataset_meta": meta, "n_blocks_used": n_blocks, "runs": {}}

    # ---- Python CPU ----
    out = RESULTS / "python_cpu.json"
    rc = run(
        [
            args.python, str(PY_DIR / "dct_cpu.py"),
            "--data-dir", str(DATA_DIR),
            "--trials", str(args.trials),
            "--warmup", str(args.warmup),
            *(["--limit", str(args.limit)] if args.limit else []),
            "--out", str(out),
        ]
    )
    if rc == 0:
        merged["runs"]["python_cpu"] = load_json(out)

    # ---- Python GPU ----
    if not args.skip_python_gpu:
        out = RESULTS / "python_gpu.json"
        rc = run(
            [
                args.python, str(PY_DIR / "dct_gpu.py"),
                "--data-dir", str(DATA_DIR),
                "--trials", str(args.trials),
                "--warmup", str(args.warmup),
                "--backend", args.gpu_backend,
                *(["--limit", str(args.limit)] if args.limit else []),
                "--out", str(out),
            ]
        )
        if rc == 0:
            merged["runs"]["python_gpu"] = load_json(out)

    # ---- C CPU ----
    if not args.skip_c_cpu:
        bin_exe = C_DIR / "dct_cpu"
        if not bin_exe.exists():
            print("Building C CPU...")
            run(["make", "cpu"], cwd=C_DIR)
        if bin_exe.exists():
            out = RESULTS / "c_cpu.json"
            run([str(bin_exe), str(DATA_DIR / "blocks.bin"), str(n_blocks),
                 "--trials", str(args.trials), "--warmup", str(args.warmup),
                 "--json", str(out)])
            merged["runs"]["c_cpu"] = load_json(out)

    # ---- CUDA ----
    if not args.skip_cuda:
        bin_exe = C_DIR / "dct_gpu"
        if not bin_exe.exists() and shutil.which("nvcc"):
            print("Building CUDA...")
            run(["make", "gpu"], cwd=C_DIR)
        if bin_exe.exists():
            out = RESULTS / "cuda.json"
            run([str(bin_exe), str(DATA_DIR / "blocks.bin"), str(n_blocks),
                 "--trials", str(args.trials), "--warmup", str(args.warmup),
                 "--json", str(out)])
            merged["runs"]["cuda"] = load_json(out)
        else:
            print("CUDA binary not available (needs nvcc + NVIDIA GPU). Skipping.")

    (RESULTS / "results.json").write_text(json.dumps(merged, indent=2))
    print(f"\nMerged results -> {RESULTS / 'results.json'}")

    write_summary(merged)
    try:
        make_plots(merged)
    except Exception as e:
        print(f"plot step failed: {e}")


def pick_mean(run: dict, key_path: list[str]) -> float | None:
    cur = run
    for k in key_path:
        if cur is None or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, (int, float)) else None


def write_summary(merged: dict) -> None:
    """Build a markdown table of the core metrics + speedups vs python_cpu."""
    runs = merged["runs"]
    n_blocks = merged["n_blocks_used"]
    rows = []

    def row(name: str, mean_ms, std_ms, gflops, bps):
        return (name, mean_ms, std_ms, gflops, bps)

    if (r := runs.get("python_cpu")):
        rows.append(row("Python CPU (numpy matmul)",
                        r["numpy_matmul"]["mean_ms"], r["numpy_matmul"]["std_ms"],
                        r["numpy_matmul"]["gflops"], r["numpy_matmul"]["blocks_per_sec"]))
        rows.append(row("Python CPU (scipy.dctn)",
                        r["scipy_fft"]["mean_ms"], r["scipy_fft"]["std_ms"],
                        r["scipy_fft"]["gflops"], r["scipy_fft"]["blocks_per_sec"]))
    if (r := runs.get("python_gpu")):
        backend = r.get("backend", "gpu")
        rows.append(row(f"Python GPU ({backend}, compute only)",
                        r["compute_only"]["mean_ms"], r["compute_only"]["std_ms"],
                        r["compute_only"]["gflops"], r["compute_only"]["blocks_per_sec"]))
        rows.append(row(f"Python GPU ({backend}, +transfer)",
                        r["with_transfer"]["mean_ms"], r["with_transfer"]["std_ms"],
                        r["with_transfer"]["gflops"], r["with_transfer"]["blocks_per_sec"]))
    if (r := runs.get("c_cpu")):
        thr = r.get("threads", 1)
        rows.append(row(f"C CPU (threads={thr})",
                        r["mean_ms"], r["std_ms"], r["gflops"], r["blocks_per_sec"]))
    if (r := runs.get("cuda")):
        gpu = r.get("gpu_name", "GPU")
        rows.append(row(f"CUDA ({gpu}, compute only)",
                        r["compute_only"]["mean_ms"], r["compute_only"]["std_ms"],
                        r["compute_only"]["gflops"], r["compute_only"]["blocks_per_sec"]))
        rows.append(row(f"CUDA ({gpu}, +transfer)",
                        r["with_transfer"]["mean_ms"], r["with_transfer"]["std_ms"],
                        r["with_transfer"]["gflops"], r["with_transfer"]["blocks_per_sec"]))

    # baseline = numpy matmul if present, else scipy, else first row
    baseline = None
    for name, ms, *_ in rows:
        if "numpy matmul" in name:
            baseline = ms; break
    if baseline is None and rows:
        baseline = rows[0][1]

    lines = []
    lines.append(f"# DCT-II 8×8 benchmark summary")
    lines.append("")
    lines.append(f"- N blocks: **{n_blocks:,}** (CIFAR-10 Y channel, 8×8 tiles, float32)")
    lines.append(f"- FLOPs per block: **{2*2*8*8*8} = 2 × 2N³** (two 8×8 matmuls)")
    lines.append(f"- Total FLOPs: **{2*2*8*8*8 * n_blocks / 1e9:.3f} GFLOP**")
    lines.append("")
    lines.append("| Implementation | mean ms | ± std | GFLOP/s | M blocks/s | speedup |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, ms, sd, gf, bps in rows:
        sp = baseline / ms if baseline and ms else float("nan")
        lines.append(
            f"| {name} | {ms:.3f} | {sd:.3f} | {gf:.2f} | {bps/1e6:.2f} | {sp:.2f}× |"
        )
    lines.append("")
    lines.append("Speedup is relative to **Python CPU (numpy matmul)** (or first row if absent).")
    lines.append("")
    lines.append("## Correctness")
    if (r := runs.get("python_cpu")):
        c = r.get("correctness_vs_scipy", {})
        lines.append(f"- NumPy matmul vs scipy.dctn: max|Δ| = {c.get('max_abs_err', 'n/a'):.2e}, "
                     f"MSE = {c.get('mse', 'n/a'):.2e}")
    if (r := runs.get("python_gpu")):
        c = r.get("correctness_vs_numpy", {})
        lines.append(f"- PyTorch GPU vs numpy:       max|Δ| = {c.get('max_abs_err', 'n/a'):.2e}, "
                     f"MSE = {c.get('mse', 'n/a'):.2e}")
    (RESULTS / "summary.md").write_text("\n".join(lines))
    print(f"summary  -> {RESULTS / 'summary.md'}")


def make_plots(merged: dict) -> None:
    import matplotlib.pyplot as plt

    runs = merged["runs"]
    labels, ms, gflops = [], [], []

    def add(name, mean_ms, gfl):
        labels.append(name); ms.append(mean_ms); gflops.append(gfl)

    if (r := runs.get("python_cpu")):
        add("Py CPU\n(numpy)", r["numpy_matmul"]["mean_ms"], r["numpy_matmul"]["gflops"])
        add("Py CPU\n(scipy)", r["scipy_fft"]["mean_ms"], r["scipy_fft"]["gflops"])
    if (r := runs.get("python_gpu")):
        b = r.get("backend", "gpu")
        add(f"Py {b}\ncompute", r["compute_only"]["mean_ms"], r["compute_only"]["gflops"])
        add(f"Py {b}\n+xfer", r["with_transfer"]["mean_ms"], r["with_transfer"]["gflops"])
    if (r := runs.get("c_cpu")):
        add(f"C CPU\n(thr={r.get('threads',1)})", r["mean_ms"], r["gflops"])
    if (r := runs.get("cuda")):
        add("CUDA\ncompute", r["compute_only"]["mean_ms"], r["compute_only"]["gflops"])
        add("CUDA\n+xfer",   r["with_transfer"]["mean_ms"], r["with_transfer"]["gflops"])

    if not labels:
        print("no data to plot")
        return

    # time plot (log scale — times span orders of magnitude)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, ms, color="steelblue")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylabel("Wall time (ms, log scale)")
    ax.set_title(f"DCT-II 8×8 on {merged['n_blocks_used']:,} blocks — wall time")
    for b, v in zip(bars, ms):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "wall_time.png", dpi=140)
    plt.close(fig)

    # throughput plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, gflops, color="seagreen")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylabel("GFLOP/s")
    ax.set_title(f"DCT-II 8×8 on {merged['n_blocks_used']:,} blocks — throughput")
    for b, v in zip(bars, gflops):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "throughput.png", dpi=140)
    plt.close(fig)

    print(f"plots    -> {PLOTS}/")


if __name__ == "__main__":
    main()
