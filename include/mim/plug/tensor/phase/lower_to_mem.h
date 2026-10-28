#pragma once

#include <mim/def.h>
#include <mim/lam.h>
#include <mim/phase.h>

namespace mim::plug::tensor::phase {

class LowerToMem : public RWPhase {
public:
    using RWPhase::RWPhase;

private:
    void start() override;
    void validate();
    bool is_tensor_type(const Def*) const;
    bool mentions_tensor(const Def*) const;
    bool is_tensor_fn(Lam*) const;
    bool has_memory_effects(const Def*, bool allocations_only = false) const;

    const Def* rewrite(const Def*) override;
    const Def* rewrite_mut_Lam(Lam*) override;
    const Def* rewrite_imm_Nom(const Nom*) override;
    const Def* rewrite_imm_Name(const Name*) override;
    const Def* rewrite_imm_Struc(const Struc*) override;
    const Def* rewrite_imm_App(const App*) override;
    const Def* uncurry(Lam*);
    const Def* lower_call(const App*, Lam*);
    const Def* buf_of(const Def*);

    const Def* lower_from_array(const App*);
    const Def* lower_get(const App*);
    const Def* lower_set(const App*);
    const Def* lower_splat(const App*);
    const Def* lower_generate(const App*);
    const Def* lower_broadcast(const App*);
    const Def* lower_map_reduce(const App*);
    const Def* lower_pad(const App*);
    const Def* lower_concat(const App*);
    const Def* lower_gather(const App*);
    const Def* lower_scatter(const App*);
    const Def* splat_buffer(const Def*, const Def*);
    const Def* buffer_list(const Def*, const Def*);

    void collect_leaves(const Def*, DefVec&);
    const DefVec& leaves(Lam*);
    const Def* conv_leaf(const Def*);
    const Def* rebuild_param(const Def*);
    void explode_arg(const Def*, const Def*, DefVec&);

    const Def* bot_mem();
    const Def* fresh_mem();
    const Def* wrap_fresh_mem(const Def*);

    flags_t tensor_key_ = 0;
    bool tensors_seen_  = false;
    fe::Vector<Lam*> pending_;
    DefMap<const Def*> fresh_memo_;
    LamSet tensor_fns_;
    LamSet shape_poly_;
    LamSet inline_callbacks_;
    LamMap<DefVec> leaves_;
    DefSet leaf_;
};

} // namespace mim::plug::tensor::phase
