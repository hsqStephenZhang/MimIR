#include "mim/plug/tensor/phase/op_pattern.h"
#include "mim/plug/tensor/tensor.h"
#include "mim/plug/affine/affine.h"

#include <format>

namespace mim::plug::tensor::phase {

static const char* kind_name(OpPatternKind kind) {
    switch (kind) {
        case OpPatternKind::kElemWise: return "ElemWise";
        case OpPatternKind::kBroadcast: return "Broadcast";
        case OpPatternKind::kInjective: return "Injective";
        case OpPatternKind::kCommReduce: return "CommReduce";
        case OpPatternKind::kOutEWiseFusable: return "OutEWiseFusable";
        case OpPatternKind::kTuple: return "Tuple";
        case OpPatternKind::kOpaque: return "Opaque";
    }
    fe::unreachable();
}

static bool is_id_map(const Def* map) {
    if (auto lam = map->isa<Lam>()) {
        auto mut_lam = const_cast<Lam*>(lam);
        if (mut_lam->body() == mut_lam->var()) return true;
        if (auto tuple = mut_lam->body()->isa<Tuple>()) {
            bool matches = true;
            for (size_t j = 0; j < tuple->num_ops(); ++j) {
                if (tuple->op(j) != mut_lam->var(j)) {
                    matches = false;
                    break;
                }
            }
            return matches;
        }
    }
    return false;
}

static bool is_zero_idx(const Def* idx) {
    if (auto lit = Lit::isa<u64>(idx)) return *lit == 0;
    if (auto app = idx->isa<App>()) {
        if (Axm::isa<affine::constant>(app)) {
            return Lit::isa<u64>(app->arg()) && *Lit::isa<u64>(app->arg()) == 0;
        }
    }
    return false;
}

static bool is_broadcast_map(const Def* map) {
    if (auto lam = map->isa<Lam>()) {
        auto mut_lam = const_cast<Lam*>(lam);
        auto body    = mut_lam->body();
        auto var     = mut_lam->var();

        if (auto ex = body->isa<Extract>()) {
            if (ex->tuple() == var) {
                return true;
            }
        }

        if (auto tuple = body->isa<Tuple>()) {
            size_t last_idx = -1;
            bool first      = true;
            for (size_t j = 0; j < tuple->num_ops(); ++j) {
                auto op = tuple->op(j);
                if (is_zero_idx(op)) {
                    continue;
                }
                if (auto ex = op->isa<Extract>()) {
                    if (ex->tuple() == var) {
                        if (auto idx_lit = Lit::isa<u64>(ex->index())) {
                            if (!first && *idx_lit <= last_idx) {
                                return false;
                            }
                            last_idx = *idx_lit;
                            first    = false;
                            continue;
                        }
                    }
                }

                bool found = false;
                for (size_t k = (first ? 0 : last_idx + 1); k < mut_lam->num_vars(); ++k) {
                    if (op == mut_lam->var(k)) {
                        last_idx = k;
                        first    = false;
                        found    = true;
                        break;
                    }
                }
                if (!found) return false;
            }
            return true;
        }
    }
    return false;
}

static const char* map_tag(const Def* map) {
    if (is_id_map(map)) return "elemwise";
    if (is_broadcast_map(map)) return "broadcast";
    return "injective";
}

const Def* OpPatternAnalysis::rewrite_imm_App(const App* app) {
    auto record_kind = [&](OpPatternKind kind, std::string summary = {}) {
        patterns_[app] = kind;
        if (summary.empty())
            app->world().ILOG("OpPatternAnalysis: kind {} ({}) for {}", (int)kind, kind_name(kind), app);
        else
            app->world().ILOG("OpPatternAnalysis: kind {} ({}) for {} {}", (int)kind, kind_name(kind), app, summary);
    };

    if (auto mr = Axm::isa<tensor::map_reduce>(app)) {
        record_kind(analyze_map_reduce_aff(mr), summarize_map_reduce_aff(mr));
    } else if (Axm::isa<tensor::dot_product>(app) || Axm::isa<tensor::product_2d>(app)) {
        // Simple map_reduce and dot products are reductions.
        // We could be more precise for map_reduce if we check if subs covers all output dims.
        record_kind(OpPatternKind::kCommReduce);
    } else if (Axm::isa<tensor::broadcast>(app) || Axm::isa<tensor::broadcast_in_dim>(app)) {
        record_kind(OpPatternKind::kBroadcast);
    } else if (Axm::isa<tensor::transpose>(app) || Axm::isa<tensor::transpose_2d>(app) || Axm::isa<tensor::reshape>(app) || Axm::isa<tensor::repeat>(app)) {
        record_kind(OpPatternKind::kInjective);
    } else if (Axm::isa<tensor::map>(app) || Axm::isa<tensor::unary>(app) || Axm::isa<tensor::binary>(app) || Axm::isa<tensor::select>(app)) {
        record_kind(OpPatternKind::kElemWise);
    } else if (Axm::isa<tensor::get>(app) || Axm::isa<tensor::set>(app)) {
        record_kind(OpPatternKind::kOpaque); // Treat array access as opaque for now
    }

    return Analysis::rewrite_imm_App(app);
}

std::string OpPatternAnalysis::summarize_map_reduce_aff(const App* mra) {
    auto callee = mra->callee()->as<App>();
    auto [nis, ToRoRr, SoSr, _TisRisSis, comb_init, map_out, maps] = callee->uncurry_args<7>();
    (void)_TisRisSis;
    auto [To, Ro, Rr] = ToRoRr->projs<3>();

    auto nis_lit = Lit::isa<u64>(nis);
    auto Ro_lit  = Lit::isa<u64>(Ro);
    auto Rr_lit  = Lit::isa<u64>(Rr);

    std::string summary = std::format("(nis={}, Ro={}, Rr={}, map_out={}, maps=[", nis_lit ? std::to_string(*nis_lit) : "?", Ro_lit ? std::to_string(*Ro_lit) : "?", Rr_lit ? std::to_string(*Rr_lit) : "?", map_tag(map_out));
    if (nis_lit) {
        for (u64 i = 0; i < *nis_lit; ++i) {
            if (i) summary += ",";
            summary += map_tag(maps->proj(*nis_lit, i));
        }
    } else {
        summary += "?";
    }
    summary += "])";
    return summary;
}

OpPatternKind OpPatternAnalysis::analyze_map_reduce_aff(const App* mra) {
    auto callee = mra->callee()->as<App>();
    auto [nis, ToRoRr, SoSr, TisRisSis, comb_init, map_out, maps] = callee->uncurry_args<7>();

    auto [To, Ro, Rr] = ToRoRr->projs<3>();
    auto Rr_lit = Lit::isa<u64>(Rr);

    if (!Rr_lit) return OpPatternKind::kOpaque;

    bool is_elem_wise = true;

    if (!is_id_map(map_out)) {
        is_elem_wise = false;
    }

    auto nis_lit = Lit::isa<u64>(nis);
    if (!nis_lit) return OpPatternKind::kOpaque;
    auto nis_nat = *nis_lit;

    bool has_elem_wise = false;
    bool has_broadcast = false;
    bool has_injective = false;

    for (u64 i = 0; i < nis_nat; ++i) {
        auto map_i = maps->proj(nis_nat, i);
        if (is_id_map(map_i)) {
            has_elem_wise = true;
        } else if (is_broadcast_map(map_i)) {
            has_broadcast = true;
        } else {
            has_injective = true;
        }
    }

    if (*Rr_lit > 0) {
        if (has_injective) return OpPatternKind::kOutEWiseFusable;
        return OpPatternKind::kCommReduce;
    }

    if (has_injective || !is_elem_wise) {
        return OpPatternKind::kInjective;
    }

    // ElemWise Assimilation Rule from TVM:
    // If at least one input is kElemWise and the rest are kBroadcast, overall is kElemWise.
    if (has_elem_wise) {
        return OpPatternKind::kElemWise;
    }

    if (has_broadcast) {
        return OpPatternKind::kBroadcast;
    }

    // Default if no inputs (nis=0)
    return OpPatternKind::kElemWise;
}

} // namespace mim::plug::tensor::phase
