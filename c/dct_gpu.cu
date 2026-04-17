/*
 * dct_gpu.cu - CUDA DCT-II (8x8, orthonormal) on CIFAR-10 blocks.
 *
 * Same algorithm as dct_cpu.c: Y = C * X * C^T.
 *
 * Kernel design:
 *   - 1 CUDA block = 1 image block (8x8 = 64 threads, one per output element)
 *   - DCT basis C lives in __constant__ memory (64 floats)
 *   - Each thread: loads one X[i][j] into shared mem, two __syncthreads()
 *     between the two matmul passes, writes one Y[i][j].
 *
 * Reports two timings via cudaEvents:
 *   compute_only   - kernel only (data already on device)
 *   with_transfer  - cudaMemcpy H->D + kernel + cudaMemcpy D->H
 */

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, \
            cudaGetErrorString(_e)); exit(1); } } while (0)

__constant__ float d_C[DCT_NN];

/*  1 thread per output element.  blockDim=(8,8,1), gridDim=(n_blocks,1,1). */
__global__ void dct_kernel(const float * __restrict__ X,
                           float * __restrict__ Y,
                           size_t n_blocks) {
    size_t b = blockIdx.x;
    if (b >= n_blocks) return;

    int i = threadIdx.y;  /* row 0..7 */
    int j = threadIdx.x;  /* col 0..7 */

    __shared__ float s_x[DCT_N][DCT_N];
    __shared__ float s_tmp[DCT_N][DCT_N];

    const float *xb = X + b * DCT_NN;
    float *yb       = Y + b * DCT_NN;

    /* Load block into shared memory. */
    s_x[i][j] = xb[i * DCT_N + j];
    __syncthreads();

    /* First matmul: tmp = C * X  =>  tmp[i][j] = sum_k C[i][k] * X[k][j] */
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < DCT_N; k++) {
        s += d_C[i * DCT_N + k] * s_x[k][j];
    }
    s_tmp[i][j] = s;
    __syncthreads();

    /* Second matmul: Y = tmp * C^T => Y[i][j] = sum_k tmp[i][k] * C[j][k] */
    float t = 0.0f;
    #pragma unroll
    for (int k = 0; k < DCT_N; k++) {
        t += s_tmp[i][k] * d_C[j * DCT_N + k];
    }
    yb[i * DCT_N + j] = t;
}

static void build_basis_host(float C[DCT_NN]) {
    const double pi = 3.14159265358979323846;
    const double a0 = sqrt(1.0 / (double)DCT_N);
    const double ak = sqrt(2.0 / (double)DCT_N);
    for (int k = 0; k < DCT_N; k++) {
        double a = (k == 0) ? a0 : ak;
        for (int i = 0; i < DCT_N; i++) {
            C[k * DCT_N + i] = (float)(a * cos(pi * (2 * i + 1) * k
                                               / (double)(2 * DCT_N)));
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage: %s <blocks.bin> <n_blocks> [--trials N] [--warmup W] "
            "[--json PATH]\n", argv[0]);
        return 2;
    }
    const char *bin_path = argv[1];
    size_t n_blocks = strtoull(argv[2], NULL, 10);
    int trials = 10, warmup = 3;
    const char *json_path = NULL;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--trials") && i + 1 < argc) trials = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
    }

    size_t n_floats = n_blocks * DCT_NN;
    size_t bytes    = n_floats * sizeof(float);

    /* Host (pinned) buffers. */
    float *hX = NULL, *hY = NULL;
    CUDA_CHECK(cudaMallocHost(&hX, bytes));
    CUDA_CHECK(cudaMallocHost(&hY, bytes));

    FILE *f = fopen(bin_path, "rb");
    if (!f) { perror(bin_path); return 1; }
    if (fread(hX, sizeof(float), n_floats, f) != n_floats) {
        fprintf(stderr, "short read from %s\n", bin_path); return 1;
    }
    fclose(f);

    float *dX = NULL, *dY = NULL;
    CUDA_CHECK(cudaMalloc(&dX, bytes));
    CUDA_CHECK(cudaMalloc(&dY, bytes));
    CUDA_CHECK(cudaMemcpy(dX, hX, bytes, cudaMemcpyHostToDevice));

    float hC[DCT_NN];
    build_basis_host(hC);
    CUDA_CHECK(cudaMemcpyToSymbol(d_C, hC, sizeof(hC)));

    dim3 block(DCT_N, DCT_N, 1);
    dim3 grid((unsigned)n_blocks, 1, 1);

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    /* ---------- compute-only ---------- */
    for (int w = 0; w < warmup; w++) {
        dct_kernel<<<grid, block>>>(dX, dY, n_blocks);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    double *compute_ms = (double*)malloc(trials * sizeof(double));
    for (int t = 0; t < trials; t++) {
        CUDA_CHECK(cudaEventRecord(e0));
        dct_kernel<<<grid, block>>>(dX, dY, n_blocks);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        compute_ms[t] = ms;
    }

    /* ---------- with transfer ---------- */
    for (int w = 0; w < warmup; w++) {
        CUDA_CHECK(cudaMemcpy(dX, hX, bytes, cudaMemcpyHostToDevice));
        dct_kernel<<<grid, block>>>(dX, dY, n_blocks);
        CUDA_CHECK(cudaMemcpy(hY, dY, bytes, cudaMemcpyDeviceToHost));
    }

    double *xfer_ms = (double*)malloc(trials * sizeof(double));
    for (int t = 0; t < trials; t++) {
        CUDA_CHECK(cudaEventRecord(e0));
        CUDA_CHECK(cudaMemcpyAsync(dX, hX, bytes, cudaMemcpyHostToDevice));
        dct_kernel<<<grid, block>>>(dX, dY, n_blocks);
        CUDA_CHECK(cudaMemcpyAsync(hY, dY, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        xfer_ms[t] = ms;
    }

    /* summarize */
    #define SUMM(arr, nm, sm, sd, mn, mx) do { \
        double _s = 0, _mn = arr[0], _mx = arr[0]; \
        for (int i = 0; i < trials; i++) { _s += arr[i]; \
            if (arr[i] < _mn) _mn = arr[i]; if (arr[i] > _mx) _mx = arr[i]; } \
        sm = _s / trials; \
        double _v = 0; for (int i = 0; i < trials; i++) \
            _v += (arr[i]-sm)*(arr[i]-sm); \
        sd = (trials > 1) ? sqrt(_v/(trials-1)) : 0.0; \
        mn = _mn; mx = _mx; } while(0)

    double c_mean, c_std, c_min, c_max;
    double x_mean, x_std, x_min, x_max;
    SUMM(compute_ms, _, c_mean, c_std, c_min, c_max);
    SUMM(xfer_ms,    _, x_mean, x_std, x_min, x_max);

    double total_flops = (double)FLOPS_PER_BLOCK * (double)n_blocks;
    double c_gflops = total_flops / (c_mean * 1e-3) / 1e9;
    double x_gflops = total_flops / (x_mean * 1e-3) / 1e9;
    double c_bps    = (double)n_blocks / (c_mean * 1e-3);
    double x_bps    = (double)n_blocks / (x_mean * 1e-3);

    cudaDeviceProp props; int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&props, dev));

    FILE *out = json_path ? fopen(json_path, "w") : stdout;
    if (!out) { perror(json_path); return 1; }
    fprintf(out, "{\n");
    fprintf(out, "  \"impl\": \"cuda\",\n");
    fprintf(out, "  \"gpu_name\": \"%s\",\n", props.name);
    fprintf(out, "  \"compute_capability\": \"%d.%d\",\n", props.major, props.minor);
    fprintf(out, "  \"n_blocks\": %zu,\n", n_blocks);
    fprintf(out, "  \"block_size\": %d,\n", DCT_N);
    fprintf(out, "  \"dtype\": \"float32\",\n");
    fprintf(out, "  \"flops_per_block\": %d,\n", FLOPS_PER_BLOCK);
    fprintf(out, "  \"total_flops\": %.0f,\n", total_flops);
    fprintf(out, "  \"trials\": %d,\n", trials);
    fprintf(out, "  \"warmup\": %d,\n", warmup);
    fprintf(out, "  \"compute_only\": {\n");
    fprintf(out, "    \"mean_ms\": %.6f, \"std_ms\": %.6f, \"min_ms\": %.6f, \"max_ms\": %.6f,\n",
            c_mean, c_std, c_min, c_max);
    fprintf(out, "    \"gflops\": %.6f, \"blocks_per_sec\": %.6f,\n", c_gflops, c_bps);
    fprintf(out, "    \"per_trial_ms\": [");
    for (int t = 0; t < trials; t++) fprintf(out, "%s%.6f", t ? ", " : "", compute_ms[t]);
    fprintf(out, "]\n  },\n");
    fprintf(out, "  \"with_transfer\": {\n");
    fprintf(out, "    \"mean_ms\": %.6f, \"std_ms\": %.6f, \"min_ms\": %.6f, \"max_ms\": %.6f,\n",
            x_mean, x_std, x_min, x_max);
    fprintf(out, "    \"gflops\": %.6f, \"blocks_per_sec\": %.6f,\n", x_gflops, x_bps);
    fprintf(out, "    \"per_trial_ms\": [");
    for (int t = 0; t < trials; t++) fprintf(out, "%s%.6f", t ? ", " : "", xfer_ms[t]);
    fprintf(out, "]\n  }\n");
    fprintf(out, "}\n");
    if (out != stdout) fclose(out);

    fprintf(stderr,
        "[cuda %s] compute: %.3f ± %.3f ms [%.2f GFLOP/s, %.2f M blocks/s]\n",
        props.name, c_mean, c_std, c_gflops, c_bps / 1e6);
    fprintf(stderr,
        "[cuda %s] +xfer:   %.3f ± %.3f ms [%.2f GFLOP/s, %.2f M blocks/s]\n",
        props.name, x_mean, x_std, x_gflops, x_bps / 1e6);

    free(compute_ms); free(xfer_ms);
    cudaFree(dX); cudaFree(dY);
    cudaFreeHost(hX); cudaFreeHost(hY);
    return 0;
}
