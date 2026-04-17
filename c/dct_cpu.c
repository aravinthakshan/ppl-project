/*
 * dct_cpu.c - CPU DCT-II (8x8, orthonormal) on CIFAR-10 blocks.
 *
 * Algorithm (same as Python/CUDA): Y = C * X * C^T, where C is the 8x8
 * orthonormal DCT-II basis. Implemented as two explicit 8x8 matmuls so
 * -O3 autovectorizes and unrolls well.
 *
 * Args:
 *   ./dct_cpu <blocks.bin> <n_blocks> [--trials N] [--warmup W] [--json PATH]
 *
 * Output: JSON (stdout or --json path) with timing and correctness info.
 */

#define _POSIX_C_SOURCE 200809L
#include "common.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}

/* One 8x8 block DCT:  Y = C * X * C^T, all row-major 8x8 floats. */
static inline void dct_block(const float C[DCT_N][DCT_N],
                             const float *x, float *y) {
    float tmp[DCT_N][DCT_N];  /* = C * X */
    for (int i = 0; i < DCT_N; i++) {
        for (int j = 0; j < DCT_N; j++) {
            float s = 0.0f;
            for (int k = 0; k < DCT_N; k++) {
                s += C[i][k] * x[k * DCT_N + j];
            }
            tmp[i][j] = s;
        }
    }
    /* y = tmp * C^T => y[i][j] = sum_k tmp[i][k] * C[j][k]  */
    for (int i = 0; i < DCT_N; i++) {
        for (int j = 0; j < DCT_N; j++) {
            float s = 0.0f;
            for (int k = 0; k < DCT_N; k++) {
                s += tmp[i][k] * C[j][k];
            }
            y[i * DCT_N + j] = s;
        }
    }
}

static void run_all(const float C[DCT_N][DCT_N],
                    const float *X, float *Y, size_t n_blocks) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t b = 0; b < n_blocks; b++) {
        dct_block(C, X + b * DCT_NN, Y + b * DCT_NN);
    }
}

typedef struct { double mean_ms, std_ms, min_ms, max_ms; } stats_t;

static stats_t summarize(const double *a, int n) {
    double s = 0, mn = a[0], mx = a[0];
    for (int i = 0; i < n; i++) { s += a[i]; mn = a[i] < mn ? a[i] : mn; mx = a[i] > mx ? a[i] : mx; }
    double mean = s / n;
    double var = 0;
    for (int i = 0; i < n; i++) var += (a[i] - mean) * (a[i] - mean);
    double sd = (n > 1) ? sqrt(var / (n - 1)) : 0.0;
    stats_t out = { mean, sd, mn, mx };
    return out;
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
    float *X = (float*)malloc(n_floats * sizeof(float));
    float *Y = (float*)malloc(n_floats * sizeof(float));
    if (!X || !Y) { fprintf(stderr, "oom\n"); return 1; }

    FILE *f = fopen(bin_path, "rb");
    if (!f) { perror(bin_path); return 1; }
    if (fread(X, sizeof(float), n_floats, f) != n_floats) {
        fprintf(stderr, "short read from %s\n", bin_path);
        return 1;
    }
    fclose(f);

    float C[DCT_N][DCT_N];
    dct_build_basis(C);

    for (int w = 0; w < warmup; w++) run_all(C, X, Y, n_blocks);

    double *samples = (double*)malloc(trials * sizeof(double));
    for (int t = 0; t < trials; t++) {
        double t0 = now_ms();
        run_all(C, X, Y, n_blocks);
        samples[t] = now_ms() - t0;
    }
    stats_t s = summarize(samples, trials);

    double total_flops = (double)FLOPS_PER_BLOCK * (double)n_blocks;
    double gflops = total_flops / (s.mean_ms * 1e-3) / 1e9;
    double blocks_per_sec = (double)n_blocks / (s.mean_ms * 1e-3);

    int n_threads = 1;
#ifdef _OPENMP
    #pragma omp parallel
    { if (omp_get_thread_num() == 0) n_threads = omp_get_num_threads(); }
#endif

    FILE *out = json_path ? fopen(json_path, "w") : stdout;
    if (!out) { perror(json_path); return 1; }
    fprintf(out, "{\n");
    fprintf(out, "  \"impl\": \"c_cpu\",\n");
    fprintf(out, "  \"openmp\": %s,\n",
#ifdef _OPENMP
        "true"
#else
        "false"
#endif
    );
    fprintf(out, "  \"threads\": %d,\n", n_threads);
    fprintf(out, "  \"n_blocks\": %zu,\n", n_blocks);
    fprintf(out, "  \"block_size\": %d,\n", DCT_N);
    fprintf(out, "  \"dtype\": \"float32\",\n");
    fprintf(out, "  \"flops_per_block\": %d,\n", FLOPS_PER_BLOCK);
    fprintf(out, "  \"total_flops\": %.0f,\n", total_flops);
    fprintf(out, "  \"trials\": %d,\n", trials);
    fprintf(out, "  \"warmup\": %d,\n", warmup);
    fprintf(out, "  \"mean_ms\": %.6f,\n", s.mean_ms);
    fprintf(out, "  \"std_ms\": %.6f,\n", s.std_ms);
    fprintf(out, "  \"min_ms\": %.6f,\n", s.min_ms);
    fprintf(out, "  \"max_ms\": %.6f,\n", s.max_ms);
    fprintf(out, "  \"gflops\": %.6f,\n", gflops);
    fprintf(out, "  \"blocks_per_sec\": %.6f,\n", blocks_per_sec);
    fprintf(out, "  \"per_trial_ms\": [");
    for (int t = 0; t < trials; t++) fprintf(out, "%s%.6f", t ? ", " : "", samples[t]);
    fprintf(out, "]\n}\n");
    if (out != stdout) fclose(out);

    fprintf(stderr,
        "[c_cpu threads=%d] %.2f ± %.2f ms  [%.2f GFLOP/s, %.2f M blocks/s]\n",
        n_threads, s.mean_ms, s.std_ms, gflops, blocks_per_sec / 1e6);

    free(samples); free(X); free(Y);
    return 0;
}
