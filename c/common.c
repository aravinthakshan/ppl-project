#include "common.h"
#include <math.h>

void dct_build_basis(float C[DCT_N][DCT_N]) {
    const float pi = 3.14159265358979323846f;
    const float a0 = (float)sqrt(1.0 / (double)DCT_N);
    const float ak = (float)sqrt(2.0 / (double)DCT_N);
    for (int k = 0; k < DCT_N; k++) {
        float a = (k == 0) ? a0 : ak;
        for (int i = 0; i < DCT_N; i++) {
            C[k][i] = a * (float)cos((double)pi * (2 * i + 1) * k
                                     / (double)(2 * DCT_N));
        }
    }
}
