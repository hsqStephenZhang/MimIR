#include <cstdio>
extern "C" int wgmma_tile(_Float16* a, _Float16* bt, _Float16* b, float (*ck)[8][4], float (*cm)[8][4]);
int main() {
    static _Float16 a[64 * 64], bt[64 * 64], b[64 * 64];
    float A[64][64], B[64][64];
    for (int i = 0; i < 64; ++i)
        for (int k = 0; k < 64; ++k) A[i][k] = float((3 * i + 5 * k + 1) % 7) - 3, a[i * 64 + k] = _Float16(A[i][k]);
    for (int k = 0; k < 64; ++k)
        for (int n = 0; n < 64; ++n) {
            B[k][n]          = float((7 * k + 2 * n + 3) % 5) - 2;
            b[k * 64 + n]    = _Float16(B[k][n]);
            bt[n * 64 + k]   = _Float16(B[k][n]);
        }
    static float ck[128][8][4], cm[128][8][4];
    wgmma_tile(a, bt, b, ck, cm);
    int bad = 0;
    for (int tid = 0; tid < 128; ++tid)
        for (int j = 0; j < 8; ++j)
            for (int e = 0; e < 4; ++e) {
                int l = tid % 32, row = 16 * (tid / 32) + l / 4 + 8 * (e / 2), col = 8 * j + 2 * (l % 4) + e % 2;
                float want = 0;
                for (int k = 0; k < 64; ++k) want += A[row][k] * B[k][col];
                if (ck[tid][j][e] != want || cm[tid][j][e] != want) {
                    if (bad < 4) std::printf("C[%d][%d] = %g (K-major), %g (MN-major), want %g\n", row, col, ck[tid][j][e], cm[tid][j][e], want);
                    ++bad;
                }
            }
    std::printf("bad=%d\n", bad);
    return bad != 0;
}
