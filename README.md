# DCT CPU vs GPU Benchmark — Python & C/CUDA

A fair, controlled comparison of the 2-D Discrete Cosine Transform (Type-II, orthonormal, 8×8 blocks — the exact variant used by JPEG) across four implementations:

|  | Python | C |
|---|---|---|
| **CPU** | NumPy matmul | Plain C, single-thread (OpenMP optional) |
| **GPU** | PyTorch CUDA matmul | Hand-written CUDA kernel |

All four read the **same binary file** of 8×8 blocks prepared from the CIFAR-10 Y (luminance) channel, use the **same algorithm** (matrix form `Y = C · X · Cᵀ`), the **same precision** (`float32`), and the **same normalization** (orthonormal DCT-II).

## Why CIFAR-10 and not random arrays?
DCT-II was designed for natural images. CIFAR-10 is 32×32 → exactly 16 non-overlapping 8×8 blocks per image (no padding). Random data gives raw throughput numbers but misses the point: this is a realistic JPEG-style workload. See `REPORT.md` for the full rationale.

## Layout
```
type_test/
├── prepare_data.py        # CIFAR-10 → Y channel → 8×8 blocks → data/blocks.bin
├── benchmark.py           # Orchestrates all 4 implementations, writes results/
├── python/
│   ├── dct_cpu.py         # NumPy matmul + scipy reference for correctness
│   └── dct_gpu.py         # PyTorch CUDA
├── c/
│   ├── dct_cpu.c          # Single-thread (OpenMP optional) C
│   ├── dct_gpu.cu         # CUDA kernel
│   ├── common.h
│   └── Makefile
├── results/               # JSON metrics, plots, tables
└── REPORT.md              # Write-up
```

## How to run

### 1. Install
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare data (once)
```bash
python prepare_data.py                  # ~60k CIFAR-10 images → 960k 8x8 blocks
```
Produces `data/blocks.bin` (float32, little-endian, row-major) and `data/meta.json`.

### 3. Build C/CUDA
```bash
cd c && make                            # builds dct_cpu and dct_gpu
```
The Makefile auto-skips `dct_gpu` if `nvcc` isn't on PATH.

### 4. Run full benchmark
```bash
python benchmark.py --trials 10 --warmup 3
```
Outputs `results/results.json`, `results/summary.md`, `results/plots/*.png`.

## Running on macOS (no CUDA)
macOS has no CUDA. You can still run:
- Python CPU ✓
- C CPU ✓
- Python GPU via Metal (MPS): `--backend mps`
- CUDA: **must** be run on Linux/Windows with an NVIDIA GPU (Colab free tier works — see `REPORT.md`).

## Metrics reported
For each implementation:
- Wall-clock time: mean ± std (ms) over N trials after W warmups
- Throughput: blocks/sec, MB/sec
- GFLOP/s (each 8×8 DCT-II = 1024 flops: two 8×8 matmuls)
- Speedup vs Python-CPU baseline
- GPU compute-only vs compute+transfer (both reported)
- Correctness: max|Δ| and MSE vs scipy.fft.dctn reference
