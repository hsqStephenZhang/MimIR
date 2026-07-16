#include "mim/plug/gpu/phase/split_off_kernels.h"

#include <mim/driver.h>

namespace mim::plug::gpu::phase {

void SplitOffKernels::start() {
    analyze();

    for (const auto& [f, entry] : old_world().annexes())
        rewrite_annex(f, entry.sym, entry.def);

    for (auto kernel : kernels_)
        rewrite(kernel);
}

bool SplitOffKernels::analyze() {
    for (auto def : old_world().annexes().defs())
        analyze(def);
    for (auto def : old_world().externals().muts())
        analyze(def);

    return false; // no fixed-point necessary
}

void SplitOffKernels::analyze(const Def* def) {
    if (auto [_, ins] = analyzed_.emplace(def); !ins) return;

    auto find_kernel = [&](auto launch) {
        auto kernel = launch->decurry()->decurry()->arg();
        if (auto lam = kernel->template isa_mut<Lam>()) kernels_.emplace(lam);
    };

    if (auto launch = Axm::isa<gpu::launch>(def))
        find_kernel(launch);
    else if (auto launch = Axm::isa<gpu::launch_3d>(def))
        find_kernel(launch);

    for (auto d : def->deps())
        analyze(d);
}

const Def* SplitOffKernels::rewrite_mut_Lam(Lam* old_lam) {
    auto new_def = RWPhase::rewrite_mut_Lam(old_lam);

    if (kernels_.contains(old_lam)) {
        auto new_lam = new_def->as_mut<Lam>();
        if (new_lam->sym().empty()) {
            assert(!old_lam->sym().empty());
            new_lam->set(old_lam->sym());
        }
        if (!new_lam->is_external()) {
            auto sym = new_lam->sym();
            if (auto i = emitted_kernel_sym2def_.find(sym); i != emitted_kernel_sym2def_.end()) {
                old_lam->unset();
                return i->second;
            }
            if (auto existing = new_lam->world().externals()[sym]) {
                old_lam->unset();
                return existing;
            }
            new_lam->externalize();
            emitted_kernel_sym2def_.emplace(sym, new_lam);
        }
        old_lam->unset();
    }

    return new_def;
}

} // namespace mim::plug::gpu::phase
