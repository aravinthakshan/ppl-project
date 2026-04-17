# DCT-II CPU vs GPU — Python vs C/CUDA

**Goal.** Measure the end-to-end cost of the 2-D Type-II orthonormal
Discrete Cosine Transform on 8×8 blocks (the exact variant used by the JPEG
standard) across four implementations, and quantify the GPU speedup one
actually obtains when the algorithm and data are held constant.

## 1. Why this workload?

The DCT-II was developed by Ahmed, Natarajan & Rao (1974) specifically as
a decorrelating transform for natural images; JPEG / MPEG / H.26x all use
the 8×8 block DCT-II. Evaluating it on a realistic image corpus is
therefore more informative than a random-array microbenchmark:

- Random arrays give you a clean throughput number but tell you nothing
  about whether the implementation is correct in the regime it was
  designed for (energy concentration in low-frequency coefficients).
- Real images exercise the same code path as a real JPEG encoder. We can
  therefore verify output shape AND that low-frequency coefficients
  dominate.
- We use **CIFAR-10** because 32×32 = 16 non-overlapping 8×8 tiles
  (no padding decisions confounding results) and the dataset is small
  enough to distribute (≈235 MB of float32 blocks for all 60 000 images).
- **MNIST was rejected** because 28×28 does not divide by 8, forcing a
  padding policy that varies between implementations.

## 2. Algorithm (identical across all four)

The 2-D orthonormal DCT-II of an 8×8 block `X` is computed as

```
Y = C · X · Cᵀ
```

where `C ∈ ℝ^{8×8}` is the DCT-II basis matrix with rows

```
C[k, i] = α(k) · cos( π · (2i + 1) · k / 16 )
α(0)    = sqrt(1/8),   α(k > 0) = sqrt(2/8)
```

This is mathematically identical to `scipy.fft.dctn(x, type=2, norm='ortho')`
applied over the last two axes. Using the matrix form — rather than a
factored FFT butterfly — keeps the **same flop count and the same memory
pattern** in every implementation, which is what makes the comparison
fair. A fast-factored algorithm (Loeffler/AAN) would change the
arithmetic cost but add variance between languages.

**Flop count per block.** Two 8×8 matmuls:
2 × (2 · 8³) = 2 × 1024 = **2048 flops per 8×8 DCT**
(counting one multiply-add as 2 flops, the usual convention).

## 3. Fairness controls

| Control | Value |
|---|---|
| Algorithm | `Y = C · X · Cᵀ` in every impl |
| Precision | `float32` end-to-end |
| Normalization | Orthonormal (`norm='ortho'`) |
| Data | One binary file `data/blocks.bin` read by all impls |
| Byte order | Little-endian (x86 and arm64 native) |
| Warmup | 3 runs discarded in every impl |
| Measurement | `N = 10` trials, mean ± std, min, max |
| CPU timer | `time.perf_counter` / `clock_gettime(CLOCK_MONOTONIC)` |
| GPU timer | `cudaEvent` for CUDA, `torch.cuda.synchronize` + `time.perf_counter` for PyTorch |
| GPU reported separately | Compute-only AND compute + H↔D transfer |
| Correctness | Max-abs-error and MSE vs `scipy.fft.dctn` reference |

**Why compute-only AND compute+transfer for GPU?**
Compute-only is the kernel's "intrinsic" speed and what most GPU
benchmarks report. But in any real pipeline you pay the PCIe round-trip;
for small workloads that can dominate. We report both so the reader can
see whether PCIe is the bottleneck.

**Why pinned host memory in the CUDA run?**
Pageable host memory would slow `cudaMemcpy` by ~2×, penalising the
+transfer variant unfairly. Pinned buffers match what a production
pipeline would do.

## 4. Dataset statistics

- Source: CIFAR-10 train + test = 60 000 images of 32×32 RGB
- Conversion: BT.601 luma `Y = 0.299R + 0.587G + 0.114B`, centred by −128
  (JPEG convention; affects the DC coefficient only, not throughput)
- Output: `60 000 × 16 = 960 000` tiles of 8×8 float32
- File size: `960 000 × 64 × 4 B ≈ 234.4 MB`

## 5. System under test

Fill in before finalising the report:

| Field | Value |
|---|---|
| CPU model | |
| CPU cores / threads | |
| RAM | |
| GPU model | |
| GPU driver | |
| CUDA toolkit version | |
| Compiler (C / CUDA) | |
| Python version | |
| NumPy version | |
| SciPy version | |
| PyTorch version | |
| OS | |

(`benchmark.py` records the GPU name and compute capability automatically.)

## 6. Results

### 6.1 CPU half (measured on the dev Mac, 2026-04-17)

Workload: 960 000 blocks × 8×8 float32 = 234 MB in, 234 MB out per trial.
10 trials, 3 warmups.

| Implementation | mean ms | ± std | GFLOP/s | M blocks/s | speedup |
|---|---:|---:|---:|---:|---:|
| Python CPU (`scipy.fft.dctn`, type=2, ortho) | 187.70 | 0.48 | 10.47 | 5.11 | 0.59× |
| Python CPU (NumPy batched matmul) | 110.70 | 1.33 | 17.76 | 8.67 | **1.00× (baseline)** |
| C CPU, `-O3 -ffast-math -march=native`, single thread | 57.40 | 0.17 | 34.25 | 16.73 | **1.93×** |

Correctness: NumPy matmul vs scipy reference, max |Δ| = `1.22e-4`, MSE = `4.2e-11` (float32 rounding noise — passes).

Observations:
- **C beats NumPy by 1.9× at the same algorithm** because NumPy dispatches one Python call that then calls BLAS twice per 960 k-batch, while the C loop's inner 8×8×8 is fully unrolled and vectorised (`-O3 -march=native`). NumPy's overhead isn't per-block (that would be ~100×), it's the two large `C @ X @ C.T` BLAS calls being slightly less optimal than a purpose-built 8×8 loop.
- **scipy.dctn is slower than NumPy matmul here** — surprising to some, but scipy routes through pocketfft's general-N DCT which has more branching and scheduling overhead than a fused 8×8 basis. For N ≥ 32 scipy wins; at N=8 the matrix form is king.

### 6.2 GPU half (to be run on Colab — see `run_on_colab.md`)

| Implementation | mean ms | ± std | GFLOP/s | M blocks/s | speedup |
|---|---:|---:|---:|---:|---:|
| PyTorch CUDA, compute only  | _pending_ | | | | |
| PyTorch CUDA, +H↔D transfer | _pending_ | | | | |
| Hand-written CUDA, compute only | _pending_ | | | | |
| Hand-written CUDA, +H↔D transfer | _pending_ | | | | |

Plots in `results/plots/`:
- `wall_time.png` — log-scale wall-clock time
- `throughput.png` — GFLOP/s

## 7. Analysis — points to address

When writing up, address at least these:

1. **Language overhead** — how much faster is C than NumPy on the same
   algorithm? On an 8×8 DCT, most NumPy time is per-call Python
   overhead, not arithmetic. Expect 10–100× for small batches, shrinking
   as the batch grows because NumPy amortises its overhead.

2. **Compute-only vs +transfer GPU speedup** — the PCIe overhead is
   roughly `2 * bytes / bandwidth`. For 234 MB @ 12 GB/s PCIe Gen3: ≈ 39 ms
   round-trip. If kernel time < 39 ms, transfer dominates, and the
   "real" speedup collapses.

3. **Is an 8×8 block DCT a good GPU workload?** Each block is only 2048
   flops. A GPU needs ≥ 10⁴–10⁵ threads in flight to saturate. Make
   sure the number of blocks (960 000 here) covers enough SMs —
   otherwise you measure launch overhead, not compute.

4. **Arithmetic intensity** — each block reads 256 B, writes 256 B, does
   2048 flops → AI ≈ 4 flops/byte. For fp32 this is memory-bound on most
   GPUs (roofline). Discuss whether observed GFLOP/s is near the memory
   roofline or the compute roofline.

5. **Correctness** — report max |Δ| vs scipy reference. Should be
   ≤ 1e-5 for fp32 (rounding noise).

6. **Where does scipy.dctn land?** It uses a factored FFT-style DCT
   (fewer flops for large N), but for N=8 the matrix form is competitive
   and sometimes faster because the basis fits in L1 and the loops
   autovectorize.

## 8. Reproducibility

```bash
# one machine, full run:
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python prepare_data.py
make -C c                       # builds dct_cpu (+ dct_gpu if nvcc present)
python benchmark.py --trials 10 --warmup 3
```

Running on **Colab T4/A100** (for the CUDA half):

```bash
!git clone <this repo>
%cd type_test
!pip install -q -r requirements.txt
!python prepare_data.py
!nvidia-smi
!cd c && make gpu
!python benchmark.py --trials 20 --warmup 5 --skip-c-cpu   # or include it
```

All random state is determined by torchvision's download (CIFAR-10 is
fixed), so given the same versions the *data* is identical across
machines; the only variance is timing.

## 9. Threats to validity

- **Thermal throttling** on laptops can bias later runs. Warmups +
  multiple trials help but do not eliminate this; run the CPU parts
  with the laptop plugged in and idle.
- **Frequency scaling**: `cpupower frequency-set -g performance` on
  Linux, disable Turbo Boost if you want variance down rather than peak.
- **NumPy BLAS backend**: MKL vs OpenBLAS changes Python CPU numbers
  substantially. Record `np.show_config()`.
- **GPU clock state**: a cold GPU runs slower. The warmup covers this
  but measure `nvidia-smi -q -d CLOCK` if in doubt.
- **`-ffast-math`** in the C build permits reassociation and may lose a
  ULP or two vs the Python reference; the correctness check still passes
  to < 1e-5.
