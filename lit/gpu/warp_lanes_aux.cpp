#include <algorithm>
#include <cmath>
#include <cstdio>
extern "C" int warp_rows(float (*a)[4], float (*b)[2], float (*c)[4], float (*d)[4], float (*r)[2]);
int main() {
    float A[16][8], B[8][8], C[16][8];
    for (int i = 0; i < 16; ++i)
        for (int k = 0; k < 8; ++k) A[i][k] = float((3 * i + 5 * k + 1) % 7) - 3, C[i][k] = float((i + 2 * k) % 5);
    for (int k = 0; k < 8; ++k)
        for (int j = 0; j < 8; ++j) B[k][j] = float((7 * k + 2 * j + 3) % 5) - 2;
    static float a[32][4], b[32][2], c[32][4], d[32][4], r[32][2];
    for (int l = 0; l < 32; ++l) {
        for (int j = 0; j < 4; ++j) {
            a[l][j] = A[l / 4 + 8 * (j % 2)][l % 4 + 4 * (j / 2)];
            c[l][j] = C[l / 4 + 8 * (j / 2)][2 * (l % 4) + j % 2];
        }
        for (int j = 0; j < 2; ++j) b[l][j] = B[l % 4 + 4 * j][l / 4];
    }
    warp_rows(a, b, c, d, r);
    float D[16][8], M[16];
    for (int i = 0; i < 16; ++i) {
        M[i] = -INFINITY;
        for (int j = 0; j < 8; ++j) {
            D[i][j] = C[i][j];
            for (int k = 0; k < 8; ++k) D[i][j] += A[i][k] * B[k][j];
            M[i] = std::max(M[i], D[i][j]);
        }
    }
    int bad = 0;
    for (int l = 0; l < 32; ++l) {
        for (int j = 0; j < 4; ++j)
            if (d[l][j] != D[l / 4 + 8 * (j / 2)][2 * (l % 4) + j % 2]) ++bad;
        for (int h = 0; h < 2; ++h)
            if (r[l][h] != M[l / 4 + 8 * h]) {
                if (bad < 3) std::printf("r[%d][%d]=%g want %g\n", l, h, r[l][h], M[l / 4 + 8 * h]);
                ++bad;
            }
    }
    std::printf("bad=%d\n", bad);
    return bad != 0;
}
