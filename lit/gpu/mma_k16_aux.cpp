#include <cstdio>
extern "C" int mma_k16(float (*a)[4][2], float (*b)[2][2], float (*c)[4], float (*dh)[4], float (*db)[4]);
int main() {
    float A[16][16], B[16][8], C[16][8];
    for (int i = 0; i < 16; ++i)
        for (int k = 0; k < 16; ++k) A[i][k] = float((3 * i + 5 * k + 1) % 7) - 3;
    for (int k = 0; k < 16; ++k)
        for (int j = 0; j < 8; ++j) B[k][j] = float((7 * k + 2 * j + 3) % 5) - 2;
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 8; ++j) C[i][j] = float((i + 2 * j) % 5);
    static float a[32][4][2], b[32][2][2], c[32][4], dh[32][4], db[32][4];
    for (int l = 0; l < 32; ++l) {
        int g = l / 4, t = l % 4;
        for (int r = 0; r < 4; ++r)
            for (int h = 0; h < 2; ++h) a[l][r][h] = A[g + 8 * (r % 2)][2 * t + h + 8 * (r / 2)];
        for (int r = 0; r < 2; ++r)
            for (int h = 0; h < 2; ++h) b[l][r][h] = B[2 * t + h + 8 * r][g];
        for (int j = 0; j < 4; ++j) c[l][j] = C[g + 8 * (j / 2)][2 * t + j % 2];
    }
    mma_k16(a, b, c, dh, db);
    int bad = 0;
    for (int l = 0; l < 32; ++l)
        for (int j = 0; j < 4; ++j) {
            int i = l / 4 + 8 * (j / 2), n = 2 * (l % 4) + j % 2;
            float want = C[i][n];
            for (int k = 0; k < 16; ++k) want += A[i][k] * B[k][n];
            if (dh[l][j] != want || db[l][j] != want) {
                if (bad < 3) std::printf("d[%d][%d] = %g, %g want %g\n", l, j, dh[l][j], db[l][j], want);
                ++bad;
            }
        }
    std::printf("bad=%d\n", bad);
    return bad != 0;
}
