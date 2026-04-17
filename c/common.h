#ifndef DCT_COMMON_H
#define DCT_COMMON_H

#include <stddef.h>
#include <stdint.h>

#define DCT_N 8
#define DCT_NN (DCT_N * DCT_N)
/* Two 8x8 matmuls per block, each 2*N^3 flops (mul+add). */
#define FLOPS_PER_BLOCK (2 * 2 * DCT_N * DCT_N * DCT_N)

/* Fill C[8][8] with the orthonormal DCT-II basis:
 *   C[k][i] = alpha(k) * cos(pi * (2i+1) * k / (2N))
 *   alpha(0) = sqrt(1/N),  alpha(k>0) = sqrt(2/N)
 */
void dct_build_basis(float C[DCT_N][DCT_N]);

#endif
