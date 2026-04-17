# Running the CUDA half on Google Colab (free T4)

macOS has no CUDA, so the two CUDA columns of the experiment (PyTorch CUDA and hand-written `.cu`) need a Linux+NVIDIA machine. A free Colab T4 is enough for this workload.

## Recipe — paste into a Colab notebook

```python
# 1. Verify GPU
!nvidia-smi

# 2. Get the project (replace with your repo URL, or upload a zip)
!git clone <YOUR_REPO_URL> type_test
%cd type_test

# 3. Install deps (Colab already has torch+cuda)
!pip install -q scipy matplotlib

# 4. Prepare data (~1 min: downloads CIFAR-10)
!python prepare_data.py

# 5. Build CUDA kernel
!cd c && make gpu
!ls -la c/dct_gpu

# 6. Run the full benchmark (CPU runs here are Colab's Xeon, not your Mac;
#    compare CPU numbers apples-to-apples either here OR on your Mac, not across)
!python benchmark.py --trials 20 --warmup 5

# 7. See the results
!cat results/summary.md
from IPython.display import Image
Image("results/plots/wall_time.png")
```

```python
Image("results/plots/throughput.png")
```

## Notes
- Colab's CPU is a shared Xeon (2 vCPUs) and its ambient noise is higher than a laptop. Ideally run **all four** implementations on the same box for a clean comparison. If you keep the CPU numbers from your Mac and only take the GPU numbers from Colab, record that split in `REPORT.md` § 5.
- T4 compute capability is 7.5 → `-arch=sm_75`. A100 is sm_80, L4 is sm_89. The Makefile defaults to `sm_70` which is compatible with everything ≥ Volta; bump it for newer cards for marginal gains:
  ```bash
  make gpu NVCCFLAGS="-O3 -std=c++14 -arch=sm_75"
  ```
- The CUDA kernel launches one CUDA block per 8×8 image block. 960 000 CUDA blocks × 64 threads = 61.4 M threads in flight — more than enough to saturate any consumer GPU.
