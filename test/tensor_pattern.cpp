#include <array>
#include <limits>
#include <sstream>

#include <doctest/doctest.h>

#include <mim/phase.h>

#include <mim/ast/parser.h>
#include <mim/pattern/rewrite.h>

#include <mim/plug/core/core.h>
#include <mim/plug/tensor/tensor.h>

using namespace mim;
using namespace mim::pattern;
using namespace mim::plug;
namespace tg = tensor::graph;

TEST_CASE("Tensor witnesses: transpose box follows source-axis permutation in three dimensions") {
    Driver driver;
    auto& w = driver.world();
    ast::load_plugin(w, "tensor");
    auto nat         = w.type_nat();
    auto rank        = w.lit_nat(3);
    auto dims        = w.tuple({w.lit_nat(5), w.lit_nat(7), w.lit_nat(11)});
    auto input       = w.axm(w.arr(Defs{w.lit_nat(5), w.lit_nat(7), w.lit_nat(11)}, nat));
    auto permutation = w.tuple({w.lit_idx(3, 1), w.lit_idx(3, 2), w.lit_idx(3, 0)});
    auto op          = w.app(w.app(w.app(w.annex<tensor::transpose>(), {nat, rank}), permutation), dims);
    auto source      = w.app(op, input);
    OpRegistry ops;
    ops.add(Annex::base<tensor::transpose>(), 4, {{3}}, true);
    GraphView graph({source}, {.operators = &ops});
    auto offset = w.tuple({w.lit_nat(2), w.lit_nat(1), w.lit_nat(3)});
    auto extent = w.tuple({w.lit_nat(4), w.lit_nat(2), w.lit_nat(3)});
    auto native = driver.witness(Annex::base<tensor::transpose_box>());
    REQUIRE(native);
    CHECK(driver.normalizer(Annex::base<tensor::transpose_box>()) == nullptr);
    auto gid    = w.curr_gid();
    auto target = w.inherit();
    std::array inputs{source, offset, extent};
    auto result = native->evaluate(graph, inputs, *target);
    REQUIRE(result.status == CheckResult::True);
    auto [starts, sizes] = result.value->projs<2>();
    for (size_t i = 0; i != 3; ++i) {
        CHECK(Lit::as(starts->proj(3, i)) == std::array<u64, 3>{1, 3, 2}[i]);
        CHECK(Lit::as(sizes->proj(3, i)) == std::array<u64, 3>{2, 3, 4}[i]);
    }
    CHECK(w.curr_gid() == gid);
    CHECK(graph.is_current());
}

namespace {

struct TensorIR {
    Driver driver;
    World& w = driver.world();
    OpRegistry ops;
    explicit TensorIR(bool load = true) {
        if (load) {
            ast::load_plugin(w, "tensor");
            driver.move_to_library(w);
            w.annex<tensor::combine_products>();
        }
        tg::register_operations(ops);
    }
    const Def* ring() {
        return w.tuple({w.type_nat(), w.lit_nat(0), w.annex(core::nat::add), w.annex(core::nat::mul)});
    }
    const Def* shape(u64 m, u64 n) { return w.tuple({w.lit_nat(m), w.lit_nat(n)}); }
    const Def* product(const Def* R, const Def* dims, const Def* a, const Def* b) {
        return w.app(w.app(w.app(w.annex<tensor::product_2d>(), R), dims), {a, b});
    }
    const Def* matrix(const Def* m, const Def* n, const Def* type = nullptr) {
        return w.arr(Defs{m, n}, type ? type : w.type_nat());
    }
    Lam* function(std::array<const Def*, 3> widths,
                  const Def* R           = nullptr,
                  bool same_input        = true,
                  const Def* rows        = nullptr,
                  const Def* contraction = nullptr) {
        if (!rows) rows = w.lit_nat(2);
        if (!contraction) contraction = w.lit_nat(3);
        if (!R) R = ring();
        auto type = tg::slot(R, 4, 0);
        DefVec dom{matrix(rows, contraction, type)}, codom;
        for (auto width : widths) {
            dom.push_back(matrix(contraction, width, type));
            codom.push_back(matrix(rows, width, type));
        }
        if (!same_input) dom.push_back(matrix(rows, contraction, type));
        auto fn = w.mut_lam(w.pi(w.sigma(dom), w.sigma(codom)))->set("f");
        DefVec outputs;
        for (size_t i = 0; i != 3; ++i)
            outputs.push_back(product(R, w.tuple({rows, contraction, widths[i]}),
                                      fn->var(dom.size(), !same_input && i == 2 ? 4 : 0), fn->var(dom.size(), i + 1)));
        fn->set(false, w.tuple(outputs))->externalize();
        return fn;
    }
    Lam* f() { return w.externals()[driver.sym("f")]->as_mut<Lam>(); }
    bool run(flags_t flags) {
        auto phase = Phase::create(driver.phases(), w.annex(flags));
        phase->run();
        return phase->todo();
    }
    size_t count(flags_t flags, std::string_view name = "f") {
        auto fn = w.externals()[driver.sym(name)]->as<Lam>();
        GraphView view({fn->body()}, {.operators = &ops});
        return std::ranges::count_if(view.operations(),
                                     [&](auto def) { return std::get<0>(Axm::get(def))->flags() == flags; });
    }
    void parse(std::string_view text) {
        ast::AST ast(w);
        ast::Parser parser(ast);
        std::istringstream input{std::string(text)};
        parser.import(input, "tensor-pattern-test.mim")->compile(ast);
        driver.move_to_library(w);
    }
};

constexpr auto Combine = Annex::base<tensor::combine_products>();
constexpr auto Product = Annex::base<tensor::product_2d>();
constexpr auto Concat  = Annex::base<tensor::concat>();
constexpr auto Slice   = Annex::base<tensor::slice>();

} // namespace

TEST_CASE("Tensor witnesses: reverse geometry is frozen and preserves shared-grid slot order") {
    TensorIR ir(false);
    ir.parse(R"(
        plugin tensor;
        plugin core;
        plugin affine;
        use tensor;
        fun inc (_ x: Nat)@tt: Nat = return (core.nat.add (x, 1));
        extern lam f (x: «6, 5; Nat», dim size: Nat): [«6, 5; Nat», «5, 6; Nat»] =
            let p = map_reduce ((6, 5), (6, 5)) (inc, 0) (affine.id @2) (affine.id @2,) x;
            let c = map_reduce ((5, 6), (5, 6)) (inc, 0) (affine.id @2)
                ((λ (j i: affine.Index): «2; affine.Index» = (i, j)),) p;
            reverse_at (p, c, dim, size);
    )");
    auto& w                = ir.w;
    auto [p, c, dim, size] = ir.f()->body()->as<App>()->args<4>();
    auto request           = [&](const Def* axis, const Def* step) { return w.tuple({p, c, axis, step}); };
    auto legal             = request(w.lit_nat_1(), w.lit_nat(2));
    std::array invalid{request(w.lit_nat(2), w.lit_nat_1()), request(w.lit_nat_0(), w.lit_nat_0()),
                       request(dim, w.lit_nat_1()), request(w.lit_nat_0(), size)};
    GraphView graph({ir.f()->body()}, {.operators = &ir.ops});
    auto schema = graph.isa_operation(ir.f()->body());
    REQUIRE(schema);
    REQUIRE(schema->inputs.size() == 2);
    CHECK(schema->inputs[0].value == p);
    CHECK(schema->inputs[1].value == c);
    CHECK_FALSE(ir.ops.decode(ir.f()->body(), 1)->complete);
    auto native = ir.driver.witness(Annex::base<tensor::reverse_legal>());
    REQUIRE(native);
    CHECK(native->num_inputs == 1);
    CHECK(ir.driver.normalizer(Annex::base<tensor::reverse_legal>()) == nullptr);
    auto target = w.inherit();
    auto gid    = w.curr_gid();
    auto result = native->evaluate(graph, std::array{legal}, *target);
    REQUIRE(result.status == CheckResult::True);
    CHECK(result.value == target->lit_tt());
    for (auto fields : invalid) {
        auto rejected = native->evaluate(graph, std::array{fields}, *target);
        CHECK(rejected.status == CheckResult::False);
        CHECK_FALSE(rejected.reason.empty());
    }
    CHECK(w.curr_gid() == gid);
    CHECK(graph.is_current());

    Rewriter importer(*target);
    auto fields = importer.rewrite(legal);
    auto build
        = target->app(target->annex<tensor::reverse_build>(), {fields->proj(4, 0)->type(), fields->proj(4, 1)->type()});
    auto plan = target->app(build, fields);
    auto pn   = plan->proj(2, 0)->as<Extract>()->tuple();
    auto cn   = plan->proj(2, 1)->as<Extract>()->tuple();
    CHECK(pn == cn);
    REQUIRE(Axm::isa<tensor::tiled_n>(pn));
    auto [meta, geometry, boxes, body] = pn->as<App>()->uncurry_args<4>();
    CHECK(Lit::as(meta->proj(4, 0)) == 2);
    CHECK(Lit::as(geometry->proj(2, 1)) == 18);
    auto [pbox, cbox] = target->app(boxes, target->lit_nat(17))->projs<2>();
    auto [poff, pext] = pbox->projs<2>();
    auto [coff, cext] = cbox->projs<2>();
    CHECK(poff == target->tuple({target->lit_nat(5), target->lit_nat(4)}));
    CHECK(pext == target->tuple({target->lit_nat(1), target->lit_nat(1)}));
    CHECK(coff == target->tuple({target->lit_nat(4), target->lit_nat(5)}));
    CHECK(cext == pext);
    auto [ptile, ctile] = target->app(body, target->lit_nat(17))->projs<2>();
    REQUIRE(tg::map_reduce(ptile));
    REQUIRE(tg::map_reduce(ctile));
    GraphView tiles({ctile}, {.operators = &ir.ops});
    CHECK(tiles.data_used_by(ptile, ctile) == CheckResult::True);
    CHECK(w.curr_gid() == gid);
}

TEST_CASE("Tensor pattern: product schema excludes ring and shape metadata") {
    TensorIR ir;
    auto f   = ir.function({ir.w.lit_nat(4), ir.w.lit_nat(5), ir.w.lit_nat(6)});
    auto gid = ir.w.curr_gid();
    GraphView graph({f->body()}, {.operators = &ir.ops});
    REQUIRE(graph.operations().size() == 3);
    for (auto op : graph.operations()) {
        auto view = graph.isa_operation(op);
        REQUIRE(view->inputs.size() == 2);
        CHECK(view->complete);
        CHECK(view->inputs[0].group == 2);
        CHECK(view->inputs[0].path == std::vector<size_t>{0});
        CHECK(view->inputs[1].path == std::vector<size_t>{1});
        CHECK(graph.is_pure(op));
        CHECK(tg::product(op) == op);
        CHECK_FALSE(tg::product(op->as<App>()->callee()));
    }
    CHECK_FALSE(graph.is_pure(Product)); // Purity is conditional on the ring.
    CHECK(ir.w.curr_gid() == gid);
}

TEST_CASE("Tensor pattern: concat expands input slots but preserves tensor-valued Packs") {
    TensorIR ir;
    auto& w     = ir.w;
    auto a      = w.pack(Defs{w.lit_nat(3), w.lit_nat(4)}, w.lit_nat(7));
    auto b      = w.pack(Defs{w.lit_nat(3), w.lit_nat(4)}, w.lit_nat(8));
    auto c      = w.pack(Defs{w.lit_nat(3), w.lit_nat(4)}, w.lit_nat(9));
    auto concat = w.app(
        w.app(w.app(w.app(w.annex<tensor::concat>(), {w.type_nat(), w.lit_nat(3), w.lit_nat(2)}), w.lit_idx(2, 1)),
              w.pack(3, ir.shape(3, 4))),
        w.tuple({a, b, c}));
    auto gid  = w.curr_gid();
    auto view = ir.ops.decode(concat, 10);
    REQUIRE(view);
    CHECK(view->complete);
    REQUIRE(view->inputs.size() == 3);
    for (size_t i = 0; i != 3; ++i) {
        CHECK(view->inputs[i].value == std::array{a, b, c}[i]);
        CHECK(view->inputs[i].path == std::vector<size_t>{i});
    }
    CHECK_FALSE(ir.ops.decode(concat, 2)->complete);
    CHECK(w.curr_gid() == gid);
}

TEST_CASE("Tensor pattern: map-reduce schemas handle singleton and three-input packs") {
    TensorIR ir(false);
    ir.parse(R"(
        plugin tensor;
        plugin core;
        plugin affine;
        use tensor;
        fun fold (acc: Nat, xs: «3; Nat»)@tt: Nat = return (core.nat.add (xs#0_3, core.nat.add (xs#1_3, xs#2_3)));
        extern lam f (a b c: «2, 3, 4; Nat», unused: Bool): «2, 3, 4; Nat» =
            map_reduce ((2, 3, 4), (2, 3, 4)) (fold, 0) (affine.id @3)
                ‹3; affine.id @3› (a, b, c);
    )");
    auto gid = ir.w.curr_gid();
    GraphView graph({ir.f()->body()}, {.operators = &ir.ops});
    REQUIRE(graph.operations().size() == 1);
    auto op = graph.operations()[0];
    CHECK(tg::map_reduce(op) == op);
    auto view = graph.isa_operation(op);
    REQUIRE(view->inputs.size() == 3);
    CHECK(view->complete);
    for (size_t i = 0; i != 3; ++i) {
        CHECK(view->inputs[i].group == 8);
        CHECK(view->inputs[i].path == std::vector<size_t>{0, i});
    }

    CHECK_FALSE(graph.is_pure(op));
    CHECK_FALSE(ir.ops.decode(op, 2)->complete);
    CHECK(ir.w.curr_gid() == gid);
}

TEST_CASE("Tensor pattern: tensor accumulators are declared data inputs") {
    TensorIR ir(false);
    ir.parse(R"(
            plugin tensor;
            plugin core;
            plugin affine;
            use tensor;
            fun add (acc: Nat, x: Nat)@tt: Nat = return (core.nat.add (acc, x));
            extern lam f (x: «3; Nat», values: «3, 5; Nat»): «3; Nat» =
                map_reduce_post @(1, 0) @(Nat, Nat, 1, 2, SchedT)
                    (3, (3, 5), mk_sched (2, 0)) @(Nat, 2, (3, 5), (), (), ())
                    (add, outs @(Nat, 1, 2) 3 (proj_map @(2, 1) 0) x, id @Nat)
                    (proj_map @(2, 1) 0) (affine.id @2, ()) (values, ());
        )");
    auto gid  = ir.w.curr_gid();
    auto view = ir.ops.decode(ir.f()->body(), 10);
    REQUIRE(view);
    CHECK(view->complete);
    REQUIRE(view->inputs.size() == 2);
    CHECK(view->inputs[1].value == ir.f()->var(2, 0));
    CHECK(view->inputs[1].group == 5);
    CHECK(view->inputs[1].path == std::vector<size_t>{1, 2});
    CHECK(ir.w.curr_gid() == gid);
}

TEST_CASE("Tensor halo: image caps survive runtime clipping and recursive producer tiling") {
    TensorIR ir(false);
    ir.parse(R"(
        plugin tensor;
        plugin core;
        plugin affine;
        use tensor;
        fun scale (_ x: Nat)@tt: Nat = return (core.nat.mul (x, 2));
        fun add (acc x: Nat)@tt: Nat = return (core.nat.add (acc, x));
        extern lam f (a: «7; Nat», off: Nat): «5; Nat» =
            let p = map_reduce (7, 7) (scale, 0) (affine.id @1) (affine.id @1,) a;
            map_reduce (5, (5, 3)) (add, 0) (proj_map @(2, 1) 0)
                ((λ (i k: affine.Index): affine.Index = affine.op.add (i, k)),) p;
    )");
    auto& w   = ir.w;
    auto f    = ir.f();
    auto off  = f->var(2, 1);
    auto two  = w.lit_nat(2);
    auto tile = [&](const Def* source, const Def* shape, const Def* start, const Def* size, const Def* cap) {
        auto call = w.app(w.app(w.annex<tensor::tile_of_box>(), {w.type_nat(), w.lit_nat_1()}), shape);
        return w.app(call, {source, start, size, cap});
    };
    auto outer = tg::map_reduce(tile(f->body(), w.lit_nat(5), off, two, two));
    REQUIRE(outer);
    auto view = Axm::isa<tensor::slice_box>(outer->arg(2, 0));
    REQUIRE(view);
    auto [meta, shape, params, value] = view->uncurry_args<4>();
    auto [start, extent, cap]         = params->projs<3>();
    CHECK(Lit::isa(cap) == 4);
    CHECK_FALSE(Lit::isa(extent));
    auto gid    = w.curr_gid();
    auto schema = ir.ops.decode(view, 10);
    REQUIRE(schema);
    REQUIRE(schema->inputs.size() == 1);
    CHECK(schema->inputs[0].value == view->arg());
    CHECK(w.curr_gid() == gid);
    auto inner = tg::map_reduce(tile(value, shape, start, extent, cap));
    REQUIRE(inner);
    auto producer = btensor::MapRedPost::of(inner->callee()->as<App>());
    auto clipped  = Axm::isa<btensor::clip_sched>(producer.sched);
    REQUIRE(clipped);
    auto [valid, output_cap, loop_cap, schedule] = clipped->args<4>();
    CHECK_FALSE(Lit::isa(valid));
    CHECK(Lit::isa(output_cap) == 4);
    CHECK(Lit::isa(loop_cap) == 4);
    auto retiled = tg::map_reduce(tile(inner, extent, w.lit_nat_0(), w.lit_nat_1(), w.lit_nat_1()));
    REQUIRE(retiled);
    auto nested = btensor::MapRedPost::of(retiled->callee()->as<App>());
    auto bounds = Axm::isa<btensor::clip_sched>(nested.sched);
    REQUIRE(bounds);
    CHECK(Lit::isa(bounds->arg(4, 1)) == 1);
    CHECK(Lit::isa(bounds->arg(4, 2)) == 1);
    auto exact_call = w.app(w.app(w.annex<tensor::tile_of>(), {w.type_nat(), w.lit_nat_1()}), w.lit_nat(5));
    auto exact      = tg::map_reduce(w.app(exact_call, {f->body(), off, two}));
    REQUIRE(exact);
    CHECK(Axm::isa<tensor::slice>(exact->arg(2, 0)));
    CHECK(Axm::isa<tensor::tile_of_box>(tile(f->body(), w.lit_nat(5), off, two, w.lit_nat_1())));
}

TEST_CASE("Tensor graph rewrite: three real products share a concat and a single product") {
    TensorIR ir;
    ir.function({ir.w.lit_nat(4), ir.w.lit_nat(5), ir.w.lit_nat(6)});
    CHECK(ir.run(Combine));
    CHECK(ir.count(Product) == 1);
    CHECK(ir.count(Concat) == 1);
    CHECK(ir.count(Slice) == 3);
    auto output       = ir.f()->body();
    const Def* merged = nullptr;
    u64 offset        = 0;
    for (size_t i = 0; i != 3; ++i) {
        auto slice                        = Axm::as<tensor::slice>(output->proj(3, i));
        auto [meta, shape, params, input] = slice->uncurry_args<4>();
        CHECK(Lit::isa(shape->proj(2, 1)) == 15);
        CHECK(Lit::isa(params->proj(3, 0)->proj(2, 1)) == offset);
        offset += i + 4;
        if (!merged) merged = input;
        CHECK(input == merged);
    }
    CHECK_FALSE(ir.run(Combine));
    CHECK(ir.count(Product) == 1);
}

TEST_CASE("Tensor graph rewrite: mismatched shared input and unproved rings stay unchanged") {
    TensorIR ir;
    SUBCASE("different left input") {
        ir.function({ir.w.lit_nat(4), ir.w.lit_nat(5), ir.w.lit_nat(6)}, nullptr, false);
    }
    SUBCASE("opaque ring operators") {
        auto binary = ir.w.pi(ir.w.sigma({ir.w.type_nat(), ir.w.type_nat()}), ir.w.type_nat());
        auto R      = ir.w.tuple({ir.w.type_nat(), ir.w.lit_nat(0), ir.w.axm(binary), ir.w.axm(binary)});
        ir.function({ir.w.lit_nat(4), ir.w.lit_nat(5), ir.w.lit_nat(6)}, R);
    }
    auto before = ir.f();
    auto gid    = ir.w.curr_gid();
    CHECK_FALSE(ir.run(Combine));
    CHECK(ir.f() == before);
    CHECK(ir.w.curr_gid() == gid);
    CHECK(ir.count(Product) == 3);
}

TEST_CASE("Tensor graph rewrite: symbolic width and overflowing total width are rejected") {
    TensorIR ir;
    SUBCASE("symbolic width") { ir.function({ir.w.lit_nat(4), ir.w.axm(ir.w.type_nat()), ir.w.lit_nat(6)}); }
    SUBCASE("overflow") {
        ir.function({ir.w.lit_nat(std::numeric_limits<u64>::max() - 2), ir.w.lit_nat(4), ir.w.lit_nat(6)});
    }
    SUBCASE("zero width") { ir.function({ir.w.lit_nat(0), ir.w.lit_nat(4), ir.w.lit_nat(6)}); }
    auto before = ir.count(Product);
    CHECK_FALSE(ir.run(Combine));
    CHECK(ir.count(Product) == before);
}

TEST_CASE("Tensor graph rewrite: singleton axes and symbolic rows preserve declared shapes") {
    TensorIR ir;
    auto rows = ir.w.lit_nat(1);
    SUBCASE("rank collapses to scalar for the first output") {
        ir.function({ir.w.lit_nat(1), ir.w.lit_nat(2), ir.w.lit_nat(3)}, nullptr, true, rows, ir.w.lit_nat(1));
    }
    SUBCASE("symbolic rows") {
        ir.function({ir.w.lit_nat(4), ir.w.lit_nat(5), ir.w.lit_nat(6)}, nullptr, true, ir.w.axm(ir.w.type_nat()));
    }
    CHECK(ir.run(Combine));
    CHECK(ir.count(Product) == 1);
    CHECK(ir.count(Slice) == 3);
}

TEST_CASE("Tensor graph rewrite: conditional purity is checked for newly constructed calls") {
    TensorIR ir;
    auto fn = ir.function({ir.w.lit_nat(4), ir.w.lit_nat(5), ir.w.lit_nat(6)});
    GraphView graph({fn->body()}, {.operators = &ir.ops, .scope = fn});
    PatternProgram p;
    GraphPattern pattern(p);
    auto anchor = pattern.anchor(p.capture());
    auto found  = find_matches(pattern, graph);
    REQUIRE(found.matches.size() == 3);
    const auto& match    = found.matches[0];
    auto old             = match.at(anchor)->as<App>();
    auto [R, dims, args] = old->uncurry_args<3>();
    RewritePlan plan(match);
    // Swapping add and mul keeps the type but violates the recognized ring contract.
    auto bad_ring = plan.tuple({plan.import(tg::slot(R, 4, 0)), plan.import(tg::slot(R, 4, 1)),
                                plan.import(tg::slot(R, 4, 3)), plan.import(tg::slot(R, 4, 2))});
    plan.replace(
        old, plan.call(ir.w.annex<tensor::product_2d>()->as<Axm>(), {bad_ring, plan.import(dims), plan.import(args)}));
    std::array plans{&plan};
    CHECK_THROWS_AS(rewrite_graph(graph, plans), std::invalid_argument);
    CHECK(graph.is_current());
    CHECK(ir.f() == fn);
}

TEST_CASE("Tensor pattern: broadcast exposes only its tensor operand") {
    TensorIR ir;
    auto& w        = ir.w;
    auto input     = w.axm(w.arr(3, w.type_nat()));
    auto broadcast = w.app(w.app(w.annex<tensor::broadcast>(), {w.type_nat(), w.lit_nat(2)}),
                           {ir.shape(1, 3), ir.shape(2, 3), input});
    auto gid       = w.curr_gid();
    auto decoded   = ir.ops.decode(broadcast, 10);
    REQUIRE(decoded);
    CHECK(decoded->complete);
    REQUIRE(decoded->inputs.size() == 1);
    CHECK(decoded->inputs[0].value == input);
    CHECK(decoded->inputs[0].path == std::vector<size_t>{2});
    CHECK(tg::broadcast(broadcast) == broadcast);
    CHECK(w.curr_gid() == gid);
}

TEST_CASE("Tensor pattern: opaque and fused input wrappers remain explicitly incomplete") {
    TensorIR ir;
    auto& w     = ir.w;
    auto meta   = w.tuple({w.type_nat(), w.lit_nat(3), w.lit_nat(2)});
    auto callee = w.app(w.app(w.app(w.annex<tensor::concat>(), meta), w.lit_idx(2, 1)), w.pack(3, ir.shape(3, 4)));
    const Def* values = nullptr;
    SUBCASE("opaque aggregate") { values = w.axm(callee->type()->as<Pi>()->dom()); }
    SUBCASE("fused tensor-valued Pack") {
        values = w.pack(Defs{w.lit_nat(3), w.lit_nat(3), w.lit_nat(4)}, w.lit_nat(7));
    }
    auto concat  = w.app(callee, values);
    auto gid     = w.curr_gid();
    auto decoded = ir.ops.decode(concat, 10);
    REQUIRE(decoded);
    CHECK_FALSE(decoded->complete);
    CHECK(decoded->inputs.empty());
    CHECK(w.curr_gid() == gid);
}

TEST_CASE("Tensor pattern: combiner and multiple epilogue slots are distinct data uses") {
    TensorIR ir(false);
    ir.parse(R"(
        plugin tensor;
        plugin core;
        use tensor;
        fun fold (acc x: Nat)@tt: Nat = return (core.nat.add (acc, x));
        fun post (x: Nat, extras: «2; Nat»)@tt: Nat =
            return (core.nat.add (x, core.nat.add (extras#0_2, extras#1_2)));
        extern lam f (a: «2; Nat», unused: Bool): «2; Nat» =
            map_reduce_post (2, 2, mk_sched (1, 0)) (fold, 0, post)
                (proj_map 0) (proj_map 0, ‹2; proj_map 0›) (a, (a, a));
    )");
    auto gid = ir.w.curr_gid();
    GraphView graph({ir.f()->body()}, {.coverage = UseCoverage::Closed, .operators = &ir.ops});
    REQUIRE(graph.operations().size() == 1);
    auto op      = graph.operations()[0];
    auto decoded = graph.isa_operation(op);
    REQUIRE(decoded->inputs.size() == 3);
    CHECK(decoded->complete);
    auto input = decoded->inputs[0].value;
    CHECK(decoded->inputs[0].path == std::vector<size_t>{0, 0});
    CHECK(decoded->inputs[1].path == std::vector<size_t>{1, 0});
    CHECK(decoded->inputs[2].path == std::vector<size_t>{1, 1});
    CHECK(decoded->inputs[1].value == input);
    CHECK(decoded->inputs[2].value == input);
    CHECK(graph.data_use_count(input) == 3);
    CHECK(graph.data_consumer_count(input) == 1);
    CHECK(ir.w.curr_gid() == gid);
}

TEST_CASE("Tensor graph rewrite: wrapping integer rings are accepted and checked by mode") {
    TensorIR ir;
    auto& w   = ir.w;
    auto mode = 0_u64;
    SUBCASE("plain wrapping arithmetic") {}
    SUBCASE("overflow assumptions are not covered by the proof") { mode = 3; }
    auto op   = [&](core::wrap f) { return w.app(w.app(w.annex(f), w.lit_nat(mode)), w.lit_nat(0)); };
    auto ring = w.tuple({w.type_i64(), w.lit_i64(0), op(core::wrap::add), op(core::wrap::mul)});
    ir.function({w.lit_nat(4), w.lit_nat(5), w.lit_nat(6)}, ring);
    CHECK(ir.run(Combine) == (mode == 0));
    CHECK(ir.count(Product) == (mode == 0 ? 1 : 3));
}

TEST_CASE("Tensor graph rewrite: identical tensor types do not prove logical shape equality") {
    TensorIR ir;
    auto& w     = ir.w;
    auto R      = ir.ring();
    auto x      = w.axm(w.arr(3, w.type_nat()));
    auto wide   = w.axm(ir.matrix(w.lit_nat(3), w.lit_nat(4)));
    auto wide2  = w.axm(wide->type());
    auto narrow = w.axm(w.arr(4, w.type_nat()));
    auto a      = ir.product(R, w.tuple({w.lit_nat(1), w.lit_nat(3), w.lit_nat(4)}), x, wide);
    auto b      = ir.product(R, w.tuple({w.lit_nat(1), w.lit_nat(3), w.lit_nat(4)}), x, wide2);
    auto c      = ir.product(R, w.tuple({w.lit_nat(3), w.lit_nat(1), w.lit_nat(4)}), x, narrow);
    auto body   = w.tuple({a, b, c});
    auto fn     = w.mut_lam(w.pi(w.sigma({}), body->type()))->set("f");
    fn->set(false, body)->externalize();
    CHECK_FALSE(ir.run(Combine));
    CHECK(ir.count(Product) == 3);
}

TEST_CASE("Tensor graph rewrite: a product used in another function retains its implementation") {
    TensorIR ir;
    auto& w   = ir.w;
    auto R    = ir.ring();
    auto dims = w.tuple({w.lit_nat(2), w.lit_nat(3), w.lit_nat(4)});
    auto x    = w.axm(ir.matrix(w.lit_nat(2), w.lit_nat(3)));
    DefVec products;
    for (size_t i = 0; i != 3; ++i)
        products.push_back(ir.product(R, dims, x, w.axm(ir.matrix(w.lit_nat(3), w.lit_nat(4)))));
    auto body = w.tuple(products);
    auto fn   = w.mut_lam(w.pi(w.sigma({}), body->type()))->set("f");
    fn->set(false, body)->externalize();
    auto other = w.mut_lam(w.pi(w.sigma({}), products[0]->type()))->set("g");
    other->set(false, products[0])->externalize();
    CHECK(ir.run(Combine));
    CHECK(ir.count(Product) == 1);
    CHECK(ir.count(Product, "g") == 1);
    CHECK(ir.count(Concat, "g") == 0);
    CHECK(ir.count(Slice, "g") == 0);
}

TEST_CASE("Tensor fusion: repeating a reduction in two slots never duplicates the reduction") {
    TensorIR ir(false);
    ir.parse(R"(
        plugin tensor;
        plugin core;
        use tensor;
        let Ring = (Nat, 0, core.nat.add, core.nat.mul);
        extern lam f (a: «2, 3; Nat», b: «3, 4; Nat»): «2, 4; Nat» =
            let product = product_2d Ring (a, b);
            binary core.nat.add (product, product);
    )");
    ir.run(Annex::base<tensor::lower_tensor>());
    auto phase = Phase::create(ir.driver.phases(), ir.w.annex<tensor::fuse_tensor>());
    for (size_t round = 0; round != 2; ++round) {
        phase->run();
        GraphView view({ir.f()->body()}, {.operators = &ir.ops});
        size_t reductions = 0, maps = 0;
        for (auto def : view.operations()) {
            if (auto mr = tg::map_reduce(def)) {
                auto meta = mr->uncurry_args<9>()[1];
                auto ro = Lit::isa(tg::slot(meta, 5, 2)), rn = Lit::isa(tg::slot(meta, 5, 3));
                REQUIRE(ro);
                REQUIRE(rn);
                if (*rn > *ro)
                    ++reductions;
                else
                    ++maps;
            }
        }
        CHECK(reductions == 1);
        CHECK(maps == 1);
        phase = phase->recreate(); // PhaseMan recreates an RWPhase between fixed-point rounds.
    }
}

TEST_CASE("Tensor graph rewrite: selecting one triple does not require complete combinatorial enumeration") {
    TensorIR ir;
    auto& w                = ir.w;
    constexpr size_t count = 24; // 2024 possible triples exceed the default 1000-result limit.
    auto R                 = ir.ring();
    auto input_type        = ir.matrix(w.lit_nat(2), w.lit_nat(3));
    auto weight_type       = ir.matrix(w.lit_nat(3), w.lit_nat(4));
    auto output_type       = ir.matrix(w.lit_nat(2), w.lit_nat(4));
    DefVec dom{input_type};
    for (size_t i = 0; i != count; ++i)
        dom.push_back(weight_type);
    auto fn = w.mut_lam(w.pi(w.sigma(dom), w.arr(count, output_type)))->set("f");
    DefVec values;
    auto dims = w.tuple({w.lit_nat(2), w.lit_nat(3), w.lit_nat(4)});
    for (size_t i = 0; i != count; ++i)
        values.push_back(ir.product(R, dims, fn->var(count + 1, 0), fn->var(count + 1, i + 1)));
    fn->set(false, w.tuple(values))->externalize();
    CHECK(ir.run(Combine));
    CHECK(ir.count(Product) == count - 2);
    CHECK(ir.count(Concat) == 1);
    CHECK(ir.count(Slice) == 3);
}
