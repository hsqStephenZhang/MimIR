#pragma once

#include <functional>
#include <unordered_map>

#include "mim/pattern/pattern.h"

namespace mim::pattern {

struct OpInput {
    const Def* value;
    size_t group;
    std::vector<size_t> path;
};

struct OpView {
    std::vector<OpInput> inputs;
    bool complete = true;
};

struct InputGroup {
    size_t index;
    /// Zero means a data value; one expands an argument tuple, two expands two wrapper levels, etc.
    /// This is explicit so tensor-valued Packs are never mistaken for argument wrappers.
    size_t tuple_depth = 0;
};

/// Plugins register data operands and purity separately from the physical App/tuple representation.
/// Decoders are read-only. An incomplete view must say so instead of inventing projections.
class OpRegistry {
public:
    using Decoder = std::function<std::optional<OpView>(const Def*, size_t max_inputs)>;
    /// A read-only proof over the complete curried argument list, also valid in a rebuilt World.
    using Purity = std::function<bool(std::span<const Def* const> argument_groups)>;

    void add(flags_t axm, size_t num_groups, std::vector<InputGroup>, bool pure = false);
    void add(flags_t axm, Decoder, bool pure);
    void add(flags_t axm, Decoder, Purity);
    std::optional<OpView> decode(const Def*, size_t max_inputs) const;
    bool is_pure(flags_t) const;
    bool has_purity(flags_t) const;
    bool is_pure_call(flags_t, std::span<const Def* const> argument_groups) const;
    bool is_pure(const Def*) const;

private:
    struct Spec {
        Decoder decode;
        bool pure;
        Purity purity = {};
    };
    std::unordered_map<flags_t, Spec> specs_;
};

} // namespace mim::pattern
