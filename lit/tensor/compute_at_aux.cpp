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
int64_t* edge_at(int64_t* x, int64_t* g);
int32_t* rev_at(int32_t* a, int32_t* b);
int32_t* chain_at(int32_t* a, int32_t* b, int32_t* e);
int32_t* back_at(int32_t* a, int32_t* b);
int32_t* fused_at(int32_t* x, int32_t* y, int32_t* a);
int32_t* fold_at(int32_t* x);
int32_t* fold_plain(int32_t* x);
int32_t* epi_at(int32_t* a, int32_t* b);
int32_t* inl_at(int32_t* a, int32_t* b);
int32_t* two_at(int32_t* a, int32_t* b);
int32_t* fuse_at(int32_t* a, int32_t* b, int32_t* e);
int32_t* fuse_lim(int32_t* a, int32_t* b, int32_t* e);
int32_t* fuse_tile(int32_t* a, int32_t* b, int32_t* e);
int32_t* split_mm(int32_t* a, int32_t* b);
int32_t* split_at(int32_t* a, int32_t* b);
int32_t* tail_mm(int32_t* a, int32_t* b);
int32_t* tail_at(int32_t* a, int32_t* b);
int32_t* tail_perm(int32_t* a, int32_t* b);
int32_t* fuse_tail(int32_t* a, int32_t* b, int32_t* e);
int32_t* shared_at(int32_t* a, int32_t* b, int32_t* e);
int32_t* rev_rowsum(int32_t* a, int32_t* b);
int32_t* rev_ew(int32_t* a);
int32_t* rev_halo(int32_t* a);
}

// q[i] = a[i, 0] + a[i, 1] over the n pairs of a.
static void pair_sum(const int32_t* a, int32_t* q, int n) {
    for (int i = 0; i < n; ++i)
        q[i] = a[2 * i] + a[2 * i + 1];
}

// c = x · y for x «m, k» and y «k, n».
static void matmul(const int32_t* x, const int32_t* y, int32_t* c, int m, int k, int n) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            int32_t s = 0;
            for (int l = 0; l < k; ++l)
                s += x[i * k + l] * y[l * n + j];
            c[i * n + j] = s;
        }
}

static void fill(int32_t* p, int n, int s) {
    for (int i = 0; i < n; ++i)
        p[i] = (i * s) % 9 - 4;
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
    int64_t x24[24], g5[5], ed[24];
    for (int i = 0; i < 5; ++i)
        g5[i] = 4 * i - 9;
    for (int c = 0; c < 24; ++c)
        x24[c] = x48[c], ed[c] = x48[c] + g5[c / 5] + 1;

    // P[i, j] = a3[i, j, 0] + a3[i, j, 1], read reversed by rev_at.
    int32_t a3[3 * 6 * 2], p[3 * 6], rv[3 * 5];
    for (int i = 0; i < 3 * 6 * 2; ++i)
        a3[i] = (i * 7) % 10 - 4;
    for (int i = 0; i < 3 * 6; ++i)
        p[i] = a3[2 * i] + a3[2 * i + 1];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j) {
            int32_t r = 0;
            for (int k = 0; k < 6; ++k)
                r += p[i * 6 + 5 - k] * b[k * 5 + j];
            rv[i * 5 + j] = r;
        }

    // The chains of chain_at, two_at, fuse_at and fuse_lim: pair sums Q, then products.
    int32_t ca[3 * 11 * 2], cb[11 * 5], ce[5 * 3], cq[3 * 11], cp[3 * 5], ch[3 * 3];
    fill(ca, 3 * 11 * 2, 7), fill(cb, 11 * 5, 5), fill(ce, 5 * 3, 4);
    pair_sum(ca, cq, 3 * 11), matmul(cq, cb, cp, 3, 11, 5), matmul(cp, ce, ch, 3, 5, 3);

    int32_t ta[3 * 7 * 2], tb[7 * 4 * 2], tp[3 * 7], tq[7 * 4], tw[3 * 4];
    fill(ta, 3 * 7 * 2, 5), fill(tb, 7 * 4 * 2, 7);
    pair_sum(ta, tp, 3 * 7), pair_sum(tb, tq, 7 * 4), matmul(tp, tq, tw, 3, 7, 4);

    int32_t fa[3 * 13 * 2], fb[13 * 10], fe[10 * 2], fq[3 * 13], fp[3 * 10], fc[3 * 2];
    fill(fa, 3 * 13 * 2, 5), fill(fb, 13 * 10, 7), fill(fe, 10 * 2, 4);
    pair_sum(fa, fq, 3 * 13), matmul(fq, fb, fp, 3, 13, 10), matmul(fp, fe, fc, 3, 10, 2);

    // fuse_tile writes C «4, 2» strip-mined to «2, 2, 2», which keeps its row-major order.
    int32_t ua[4 * 13 * 2], uq[4 * 13], up[4 * 10], uc[4 * 2];
    fill(ua, 4 * 13 * 2, 7);
    pair_sum(ua, uq, 4 * 13), matmul(uq, fb, up, 4, 13, 10), matmul(up, fe, uc, 4, 10, 2);

    // split_mm and split_at: (sa + 1) · sb.
    int32_t sa[8 * 7], sb[7 * 5], sp[8 * 7], sc[8 * 5];
    fill(sa, 8 * 7, 5), fill(sb, 7 * 5, 7);
    for (int i = 0; i < 8 * 7; ++i)
        sp[i] = sa[i] + 1;
    matmul(sp, sb, sc, 8, 7, 5);

    // tail_mm, tail_at and tail_perm: (ka + 1) · kb over 7 rows; fuse_tail: the chain over 7 rows.
    int32_t ka[7 * 9], kb[9 * 3], kp[7 * 9], kc[7 * 3];
    fill(ka, 7 * 9, 4), fill(kb, 9 * 3, 5);
    for (int i = 0; i < 7 * 9; ++i)
        kp[i] = ka[i] + 1;
    matmul(kp, kb, kc, 7, 9, 3);
    int32_t va[7 * 12 * 2], vb[12 * 9], ve[9 * 2], vq[7 * 12], vp[7 * 9], vc[7 * 2];
    fill(va, 7 * 12 * 2, 7), fill(vb, 12 * 9, 4), fill(ve, 9 * 2, 5);
    pair_sum(va, vq, 7 * 12), matmul(vq, vb, vp, 7, 12, 9), matmul(vp, ve, vc, 7, 9, 2);

    // shared_at: (P · B) · (P · E)ᵀ with P the pair sums of ha.
    int32_t ha[5 * 14 * 2], hb[14 * 3], he[14 * 3], hp[5 * 14], hc[5 * 3], hd[5 * 3], hf[5 * 5];
    fill(ha, 5 * 14 * 2, 5), fill(hb, 14 * 3, 7), fill(he, 14 * 3, 4);
    pair_sum(ha, hp, 5 * 14), matmul(hp, hb, hc, 5, 14, 3), matmul(hp, he, hd, 5, 14, 3);
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j) {
            hf[i * 5 + j] = 0;
            for (int k = 0; k < 3; ++k)
                hf[i * 5 + j] += hc[i * 3 + k] * hd[j * 3 + k];
        }

    // rev_rowsum: row sums of ra · rb; rev_ew and rev_halo over the pair sums wp of wa.
    int32_t ra[6 * 5], rb[5 * 7], rq[6 * 7], rs[6], wa[6 * 7 * 2], wp[6 * 7], we[6 * 7];
    int32_t za[6 * 8 * 2], zp[6 * 8], wh[4 * 8];
    fill(ra, 6 * 5, 7), fill(rb, 5 * 7, 4), fill(wa, 6 * 7 * 2, 5);
    matmul(ra, rb, rq, 6, 5, 7), pair_sum(wa, wp, 6 * 7);
    for (int i = 0; i < 6; ++i) {
        rs[i] = 0;
        for (int j = 0; j < 7; ++j)
            rs[i] += rq[i * 7 + j];
    }
    for (int i = 0; i < 6 * 7; ++i)
        we[i] = wp[i] + 1;
    fill(za, 6 * 8 * 2, 7), pair_sum(za, zp, 6 * 8);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j)
            wh[i * 8 + j] = zp[i * 8 + j] + zp[(i + 1) * 8 + j] + zp[(i + 2) * 8 + j];

    int32_t la[2 * 5 * 2], lb[5 * 7], le[7 * 3], lq[2 * 5], lp[2 * 7], lc[2 * 3];
    fill(la, 2 * 5 * 2, 4), fill(lb, 5 * 7, 5), fill(le, 7 * 3, 7);
    pair_sum(la, lq, 2 * 5), matmul(lq, lb, lp, 2, 5, 7), matmul(lp, le, lc, 2, 7, 3);

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

    // inl_at: R[i] = Σ_j (a8[i, j, 0] + a8[i, j, 1] + b8[i, j]).
    int32_t a8[5 * 8 * 2], b8[5 * 8], il[5] = {};
    for (int i = 0; i < 5 * 8 * 2; ++i)
        a8[i] = (i * 3) % 7 - 3;
    for (int i = 0; i < 5 * 8; ++i) {
        b8[i] = (i * 5) % 9 - 4;
        il[i / 8] += a8[2 * i] + a8[2 * i + 1] + b8[i];
    }

    int bad = 0;
    bad += compare("mm_at0", mm_at0(a, b), mm, 4 * 5);
    bad += compare("mm_at1", mm_at1(a, b), mm, 4 * 5);
    bad += compare("conv_at", conv_at(x, w), cv, 2 * 3 * 4 * 4);
    bad += compare("red_at", red_at(a, b2), rd, 4 * 5);
    bad += compare("rep_at", rep_at(x4), rp, 64);
    bad += compare("grp_at", grp_at(x48, g3), gp, 48);
    {
        auto got = edge_at(x24, g5);
        for (int c = 0; c < 24; ++c)
            if (got[c] != ed[c] && ++bad <= 5)
                std::printf("edge_at[%d]: got %lld want %lld\n", c, (long long)got[c], (long long)ed[c]);
    }
    bad += compare("rev_at", rev_at(a3, b), rv, 3 * 5);
    bad += compare("chain_at", chain_at(ca, cb, ce), ch, 3 * 3);
    bad += compare("two_at", two_at(ta, tb), tw, 3 * 4);
    bad += compare("fuse_at", fuse_at(fa, fb, fe), fc, 3 * 2);
    bad += compare("fuse_lim", fuse_lim(la, lb, le), lc, 2 * 3);
    bad += compare("fuse_tile", fuse_tile(ua, fb, fe), uc, 4 * 2);
    bad += compare("split_mm", split_mm(sa, sb), sc, 8 * 5);
    bad += compare("split_at", split_at(sa, sb), sc, 8 * 5);
    bad += compare("tail_mm", tail_mm(ka, kb), kc, 7 * 3);
    bad += compare("tail_at", tail_at(ka, kb), kc, 7 * 3);
    bad += compare("tail_perm", tail_perm(ka, kb), kc, 7 * 3);
    bad += compare("fuse_tail", fuse_tail(va, vb, ve), vc, 7 * 2);
    bad += compare("shared_at", shared_at(ha, hb, he), hf, 5 * 5);
    bad += compare("rev_rowsum", rev_rowsum(ra, rb), rs, 6);
    bad += compare("rev_ew", rev_ew(wa), we, 6 * 7);
    bad += compare("rev_halo", rev_halo(za), wh, 4 * 8);
    bad += compare("back_at", back_at(a4, b), bk, 4 * 5);
    bad += compare("fused_at", fused_at(x7, y7, a7), fu, 4 * 5);
    bad += compare("fold_at", fold_at(x15), fold_plain(x15), 5);
    bad += compare("epi_at", epi_at(a9, b9), ep, 2 * 9);
    bad += compare("inl_at", inl_at(a8, b8), il, 5);
    return bad ? 1 : 0;
}
