// Driver for compute_at_exec.mim: calls the kernels whose producers are computed one tile at a time and checks
// every output against a scalar reference.
// Exit code: 0 if all outputs match, 1 otherwise.

#include <cstdint>
#include <cstdio>

extern "C" {
// `«s; I32»` tensors lower to plain pointers at the C ABI.
int32_t* mm_at0(int32_t* a, int32_t* b);
int32_t* mm_at1(int32_t* a, int32_t* b);
int32_t* conv_at(int32_t* x, int32_t* w);
int32_t* red_at(int32_t* a, int32_t* b);
int32_t* rep_at(int32_t* x);
int32_t* grp_at(int32_t* x, int32_t* g);
int32_t* edge_at(int32_t* x, int32_t* g);
int32_t* rev_at(int32_t* a, int32_t* b);
int32_t* chain_at(int32_t* a, int32_t* b, int32_t* e);
int32_t* back_at(int32_t* a, int32_t* b);
int32_t* fused_at(int32_t* x, int32_t* y, int32_t* a);
int32_t* fold_at(int32_t* x);
int32_t* fold_plain(int32_t* x);
int32_t* epi_at(int32_t* a, int32_t* b);
}

static int compare(const char* name, const int32_t* got, const int32_t* want, int n) {
    int bad = 0;
    for (int i = 0; i < n; ++i)
        if (got[i] != want[i] && ++bad <= 5) std::printf("%s[%d]: got %d want %d\n", name, i, got[i], want[i]);
    if (bad) std::printf("%s: %d mismatches\n", name, bad);
    return bad;
}

int main() {
    int32_t a[4 * 6], b[6 * 5], b2[4 * 5], x[2 * 2 * 6 * 6], w[3 * 2 * 3 * 3];
    for (int i = 0; i < 4 * 6; ++i)
        a[i] = (i * 7) % 11 - 5;
    for (int i = 0; i < 6 * 5; ++i)
        b[i] = (i * 5) % 13 - 6;
    for (int i = 0; i < 4 * 5; ++i)
        b2[i] = (i * 3) % 7 - 3;
    for (int i = 0; i < 2 * 2 * 6 * 6; ++i)
        x[i] = (i * 11) % 9 - 4;
    for (int i = 0; i < 3 * 2 * 3 * 3; ++i)
        w[i] = (i * 13) % 5 - 2;

    int32_t mm[4 * 5];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 5; ++j) {
            int32_t s = 0;
            for (int k = 0; k < 6; ++k)
                s += (a[i * 6 + k] + 1) * b[k * 5 + j];
            mm[i * 5 + j] = s;
        }

    int32_t cv[2 * 3 * 4 * 4];
    for (int n = 0; n < 2; ++n)
        for (int c = 0; c < 3; ++c)
            for (int oh = 0; oh < 4; ++oh)
                for (int ow = 0; ow < 4; ++ow) {
                    int32_t s = 0;
                    for (int ci = 0; ci < 2; ++ci)
                        for (int kh = 0; kh < 3; ++kh)
                            for (int kw = 0; kw < 3; ++kw)
                                s += (x[((n * 2 + ci) * 6 + oh + kh) * 6 + ow + kw] + 1)
                                   * w[((c * 2 + ci) * 3 + kh) * 3 + kw];
                    cv[((n * 3 + c) * 4 + oh) * 4 + ow] = s;
                }

    int32_t rd[4 * 5];
    for (int i = 0; i < 4; ++i) {
        int32_t s = 0;
        for (int k = 0; k < 6; ++k)
            s += a[i * 6 + k];
        for (int j = 0; j < 5; ++j)
            rd[i * 5 + j] = s * b2[i * 5 + j];
    }

    int32_t rp[64], gp[48], x48[48], g3[3], x4[4];
    for (int i = 0; i < 48; ++i)
        x48[i] = (i * 5) % 7 - 3;
    for (int i = 0; i < 3; ++i)
        g3[i] = 10 * i - 7;
    for (int i = 0; i < 4; ++i)
        x4[i] = 3 * i - 4;
    for (int i = 0; i < 64; ++i)
        rp[i] = x4[i / 16] + 2;
    for (int c = 0; c < 48; ++c)
        gp[c] = x48[c] + g3[c / 16] + 1;
    int32_t g5[5], ed[24];
    for (int i = 0; i < 5; ++i)
        g5[i] = 4 * i - 9;
    for (int c = 0; c < 24; ++c)
        ed[c] = x48[c] + g5[c / 5] + 1;

    // P[i, j] = a3[i, j, 0] + a3[i, j, 1], read reversed by rev_at and in order by chain_at.
    int32_t a3[3 * 6 * 2], p[3 * 6], rv[3 * 5], pq[3 * 5], ch[3 * 3], e[5 * 3];
    for (int i = 0; i < 3 * 6 * 2; ++i)
        a3[i] = (i * 7) % 10 - 4;
    for (int i = 0; i < 3 * 6; ++i)
        p[i] = a3[2 * i] + a3[2 * i + 1];
    for (int i = 0; i < 5 * 3; ++i)
        e[i] = (i * 4) % 9 - 4;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j) {
            int32_t r = 0, q = 0;
            for (int k = 0; k < 6; ++k) {
                r += p[i * 6 + 5 - k] * b[k * 5 + j];
                q += p[i * 6 + k] * b[k * 5 + j];
            }
            rv[i * 5 + j] = r;
            pq[i * 5 + j] = q;
        }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            int32_t s = 0;
            for (int k = 0; k < 5; ++k)
                s += pq[i * 5 + k] * e[k * 3 + j];
            ch[i * 3 + j] = s;
        }

    // back_at: P = Σ over the last axis of a4 «4, 6, 2», times b; fused_at: Σ_k (x + y)[i, j, k] · P7[i, k].
    int32_t a4[4 * 6 * 2], bk[4 * 5], x7[4 * 5 * 7], y7[4 * 5 * 7], a7[4 * 7 * 2], fu[4 * 5];
    for (int i = 0; i < 4 * 6 * 2; ++i)
        a4[i] = (i * 5) % 9 - 4;
    for (int i = 0; i < 4 * 5 * 7; ++i)
        x7[i] = (i * 3) % 7 - 3, y7[i] = (i * 11) % 5 - 2;
    for (int i = 0; i < 4 * 7 * 2; ++i)
        a7[i] = (i * 7) % 6 - 2;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 5; ++j) {
            int32_t r = 0, f = 0;
            for (int k = 0; k < 6; ++k)
                r += (a4[(i * 6 + k) * 2] + a4[(i * 6 + k) * 2 + 1]) * b[k * 5 + j];
            for (int k = 0; k < 7; ++k)
                f += (x7[(i * 5 + j) * 7 + k] + y7[(i * 5 + j) * 7 + k])
                   * (a7[(i * 7 + k) * 2] + a7[(i * 7 + k) * 2 + 1]);
            bk[i * 5 + j] = r;
            fu[i * 5 + j] = f;
        }

    // epi_at: (a9 summed over its last axis) + b9 + 1.
    int32_t a9[2 * 9 * 2], b9[2 * 9], ep[2 * 9], x15[5] = {10, 20, 30, 40, 50};
    for (int i = 0; i < 2 * 9 * 2; ++i)
        a9[i] = (i * 5) % 8 - 3;
    for (int i = 0; i < 2 * 9; ++i) {
        b9[i] = (i * 7) % 5 - 2;
        ep[i] = a9[2 * i] + a9[2 * i + 1] + b9[i] + 1;
    }

    int bad = 0;
    bad += compare("mm_at0", mm_at0(a, b), mm, 4 * 5);
    bad += compare("mm_at1", mm_at1(a, b), mm, 4 * 5);
    bad += compare("conv_at", conv_at(x, w), cv, 2 * 3 * 4 * 4);
    bad += compare("red_at", red_at(a, b2), rd, 4 * 5);
    bad += compare("rep_at", rep_at(x4), rp, 64);
    bad += compare("grp_at", grp_at(x48, g3), gp, 48);
    bad += compare("edge_at", edge_at(x48, g5), ed, 24);
    bad += compare("rev_at", rev_at(a3, b), rv, 3 * 5);
    bad += compare("chain_at", chain_at(a3, b, e), ch, 3 * 3);
    bad += compare("back_at", back_at(a4, b), bk, 4 * 5);
    bad += compare("fused_at", fused_at(x7, y7, a7), fu, 4 * 5);
    bad += compare("fold_at", fold_at(x15), fold_plain(x15), 5);
    bad += compare("epi_at", epi_at(a9, b9), ep, 2 * 9);
    return bad ? 1 : 0;
}
