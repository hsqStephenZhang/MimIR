#include "mim/plug/math/phase/lower_exp.h"

#include <mim/driver.h>

#include <mim/plug/core/core.h>

#include "mim/plug/math/math.h"

namespace mim::plug::math::phase {

void LowerExp::start() {
    // NVPTX lowers `math.exp` to a vectorizable device intrinsic itself.
    if (driver().is_loaded("gpu") || !driver().is_loaded("core") || !driver().arg_nat("math", "exp-poly").value_or(1))
        return;
    RWPhase::start();
}

const Def* LowerExp::rewrite_imm_App(const App* app) {
    auto exp = Axm::isa<math::exp>(app);
    if (!exp || exp.id() != math::exp::exp || isa_f(app->type()) != 32) return RWPhase::rewrite_imm_App(app);

    auto& w   = new_world();
    auto mode = rewrite(app->callee()->as<App>()->decurry()->arg());
    auto x    = rewrite(exp->arg());
    auto f    = [&](float c) { return lit_f(w, c); };
    auto add  = [&](const Def* a, const Def* b) { return w.call(arith::add, mode, Defs{a, b}); };
    auto sub  = [&](const Def* a, const Def* b) { return w.call(arith::sub, mode, Defs{a, b}); };
    auto mul  = [&](const Def* a, const Def* b) { return w.call(arith::mul, mode, Defs{a, b}); };
    auto sel  = [&](const Def* cond, const Def* t, const Def* e) { return w.extract(w.tuple({e, t}), cond); };

    // Beyond these bounds the final 2^n products overflow to inf or underflow to zero by themselves.
    constexpr float hi = 89.0f;
    constexpr float lo = -104.0f;

    // Ordered compares keep NaN, which then propagates through the polynomial.
    auto xc = sel(w.call(cmp::g, mode, Defs{x, f(hi)}), f(hi), x);
    xc      = sel(w.call(cmp::l, mode, Defs{xc, f(lo)}), f(lo), xc);

    // Range reduction and the split 2^n scaling rely on exact rounding steps, so they must not be reassociated.
    auto exact = w.lit_nat(nat_t(Mode::contract));
    auto xadd  = [&](const Def* a, const Def* b) { return w.call(arith::add, exact, Defs{a, b}); };
    auto xsub  = [&](const Def* a, const Def* b) { return w.call(arith::sub, exact, Defs{a, b}); };
    auto xmul  = [&](const Def* a, const Def* b) { return w.call(arith::mul, exact, Defs{a, b}); };

    // Adding 1.5 * 2^23 rounds to the nearest integer and leaves n in the low mantissa bits.
    constexpr float magic = 12582912.0f;
    auto t                = xadd(xmul(xc, f(1.44269504088896341f)), f(magic));
    auto nf               = xsub(t, f(magic));
    auto r                = xsub(xsub(xc, xmul(nf, f(0.693359375f))), xmul(nf, f(-2.12194440e-4f)));

    const Def* q = f(1.9875691500e-4f);
    for (float c : {1.3981999507e-3f, 8.3334519073e-3f, 4.1665795894e-2f, 1.6666665459e-1f, 5.0000001201e-1f})
        q = add(mul(q, r), f(c));
    auto y = add(add(mul(mul(q, r), r), r), f(1.0f));

    // 2^n as two normal factors, so n down to -150 still yields the subnormal range.
    auto I32  = w.type_i32();
    auto n    = w.call(core::wrap::sub, w.lit_nat_0(),
                       Defs{w.call<core::bitcast>(I32, t), w.call<core::bitcast>(I32, f(magic))});
    auto half = w.call(core::shr::a, Defs{n, w.lit_i32(1)});
    auto rest = w.call(core::wrap::sub, w.lit_nat_0(), Defs{n, half});
    auto pow2 = [&](const Def* e) {
        auto biased = w.call(core::wrap::add, w.lit_nat_0(), Defs{e, w.lit_i32(127)});
        auto bits   = w.call(core::wrap::shl, w.lit_nat_0(), Defs{biased, w.lit_i32(23)});
        return w.call<core::bitcast>(rewrite(app->type()), bits);
    };
    return xmul(xmul(y, pow2(half)), pow2(rest));
}

} // namespace mim::plug::math::phase
