#include "mim/pattern/pattern.h"

#include <stdexcept>

#include "mim/world.h"

namespace mim::pattern {

void PatternProgram::remember(const Def* def) {
    auto& world = def->world();
    for (const auto& ref : references_)
        if (ref.world == &world && ref.generation == world.data_.storage_token) return;
    references_.push_back({&world, world.data_.storage_token});
}

bool PatternProgram::is_current() const {
    return std::ranges::all_of(references_,
                               [](const auto& ref) { return ref.generation == ref.world->data_.storage_token; });
}

void PatternProgram::check(Pattern p) const {
    if (p.program_ != this || p.index_ >= entries_.size())
        throw std::invalid_argument("pattern belongs to another program");
}

Pattern PatternProgram::append(Entry entry) {
    for (auto p : entry.operands)
        check(p);
    auto result = Pattern(this, entries_.size());
    entries_.push_back(std::move(entry));
    return result;
}

Pattern PatternProgram::any() { return append({.kind = Kind::Any}); }

Pattern PatternProgram::exact(const Def* def) {
    if (!def) throw std::invalid_argument("exact pattern requires a Def");
    remember(def);
    return append({.kind = Kind::Exact, .def = def});
}

Pattern PatternProgram::literal(const Def* type, flags_t bits) {
    if (!type) throw std::invalid_argument("literal pattern requires a type");
    remember(type);
    return append({.kind = Kind::Literal, .def = type, .flags = bits});
}

Pattern PatternProgram::node(Node node, std::vector<Pattern> operands, std::optional<flags_t> flags) {
    return append({.kind = Kind::Node, .operands = std::move(operands), .node = node, .flags = flags});
}

Pattern PatternProgram::capture(Pattern p) { return append({.kind = Kind::Capture, .operands = {p}}); }
Pattern PatternProgram::all(Pattern p, Pattern q) {
    return append({
        .kind     = Kind::All,
        .operands = {p, q}
    });
}
Pattern PatternProgram::either(Pattern p, Pattern q) {
    return append({
        .kind     = Kind::Either,
        .operands = {p, q}
    });
}
Pattern PatternProgram::negate(Pattern p) { return append({.kind = Kind::Not, .operands = {p}}); }
Pattern PatternProgram::type(Pattern p) { return append({.kind = Kind::Type, .operands = {p}}); }

Pattern PatternProgram::where(Pattern p, std::vector<Pattern> captures, Predicate predicate) {
    check(p);
    if (!predicate) throw std::invalid_argument("constraint requires a predicate");
    for (auto c : captures) {
        check(c);
        if (entries_[c.index_].kind != Kind::Capture) throw std::invalid_argument("constraint inputs must be captures");
    }
    return append(
        {.kind = Kind::Where, .operands = {p}, .inputs = std::move(captures), .predicate = std::move(predicate)});
}

} // namespace mim::pattern
