#include <cstdint>
#include <fe/sym.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mim/def.h>
#include <mim/world.h>
#include <mim/lam.h>
#include <mim/pass/optimize.h>

namespace py = pybind11;

namespace mim {

namespace {

const Def* resolve_callee(World& w, const py::object& callee_obj) {
    if (py::isinstance<py::str>(callee_obj))
        return w.sym2annex(w.sym(callee_obj.cast<std::string_view>()));
    return callee_obj.cast<const Def*>();
}

std::vector<const Def*> stage_to_defs(const py::handle& stage) {
    if (py::isinstance<py::tuple>(stage) || py::isinstance<py::list>(stage))
        return stage.cast<std::vector<const Def*>>();
    return {stage.cast<const Def*>()};
}

} // namespace

void init_world(py::module_& m) {
    py::class_<mim::World, std::unique_ptr<mim::World, py::nodelete>>(m, "World")
        .def("write", py::overload_cast<>(&mim::World::write))
        .def("write", py::overload_cast<const char*>(&mim::World::write))
        .def("write_to", [](mim::World& w, const std::string& path) {
            w.write(path.c_str());
        })
        .def("dump", py::overload_cast<>(&mim::World::dump))
        .def("annex", &mim::World::sym2annex, py::return_value_policy::reference_internal)
        .def("annex_name",
            [](mim::World& w, const std::string& name) {
                return w.sym2annex(w.sym(name));
            },
            py::return_value_policy::reference_internal)
        .def("top_nat", &mim::World::top_nat, py::return_value_policy::reference_internal)
        .def("type_bool", &mim::World::type_bool, py::return_value_policy::reference_internal)
        .def("type_i8", &mim::World::type_i8, py::return_value_policy::reference_internal)
        .def("type_i32", &mim::World::type_i32, py::return_value_policy::reference_internal)
        .def("lit_nat_0", &mim::World::lit_nat_0, py::return_value_policy::reference_internal)
        .def("lit_nat", &mim::World::lit_nat, py::return_value_policy::reference_internal)
        .def("lit", &mim::World::lit, py::return_value_policy::reference_internal)
        .def("lit_i32",
            static_cast<const mim::Lit* (World::*)(u32)>(&mim::World::lit_i32),
            py::return_value_policy::reference_internal)
        .def("type_idx", 
            static_cast<const mim::Def* (World::*)(const mim::Def*)>(&mim::World::type_idx), py::return_value_policy::reference_internal
        )
        .def("type_idx", 
            static_cast<const mim::Def* (World::*)(nat_t)>(&mim::World::type_idx), py::return_value_policy::reference_internal
        )
        .def("lit_idx",
            static_cast<const mim::Lit* (World::*)(nat_t, u64)>(&mim::World::lit_idx),
            py::return_value_policy::reference_internal)
        .def("cn",
            [](mim::World& w, std::vector<const Def*> dom) {
                return w.cn(Defs(dom));
            }
        )
        .def("lit_i8", 
            static_cast<const mim::Lit* (World::*)(u8)>(&mim::World::lit_i8), 
            py::return_value_policy::reference_internal
            )
        .def("implicit_app",
            [](mim::World& w, const mim::Def* callee, std::vector<const Def*> args) {
                return w.implicit_app(callee, Defs(args));
            },
            py::return_value_policy::reference_internal
            )
        .def("app",
            [](mim::World& w, const mim::Def* callee, std::vector<const Def*> args) {
                return w.app(callee, Defs(args));
            },
            py::return_value_policy::reference_internal)
        .def("call",
            [](mim::World& w, py::object callee_obj, py::args stages, py::kwargs kwargs) {
                bool implicit = false;
                if (kwargs.contains("implicit"))
                    implicit = kwargs["implicit"].cast<bool>();

                const Def* target = resolve_callee(w, callee_obj);
                for (const py::handle& stage : stages) {
                    auto defs = stage_to_defs(stage);
                    target = implicit ? w.implicit_app(target, Defs(defs)) : w.app(target, Defs(defs));
                }
                return target;
            },
            py::return_value_policy::reference_internal)
        .def("tuple",
            [](mim::World& w, std::vector<const Def*> ops) {
                return w.tuple(Defs(ops));
            },
            py::return_value_policy::reference_internal)
        .def("bot", &mim::World::bot, py::return_value_policy::reference_internal)
        .def("sym",
            static_cast<fe::Sym (World::*)(std::string_view)>(&mim::World::sym),
            py::return_value_policy::reference_internal)
        .def(
            "mut_fun2",
            [](mim::World& w, std::vector<mim::Def*> dom, std::vector<mim::Def*> codom) {
                auto d = dom;
                return w.mut_fun(mim::Defs(dom), mim::Defs(codom));
            },
            py::return_value_policy::reference_internal)
        .def(
            "mut_fun",
            [](mim::World& w, const mim::Def* dom, std::vector<mim::Def*> codom) {
                auto d = dom;

                //std::cout << "called mut_fun with domain: " << d << std::endl;
                return w.mut_fun(dom, codom);
            },
            py::return_value_policy::reference_internal)
        .def("arr",
            [](mim::World& w, Def* arity, Def* body){
                return w.arr(arity, body);
            },
            py::return_value_policy::reference_internal)
        .def("call_by_id",
            [](mim::World& w, uint64_t id, std::vector<Def*> args) {
                if (args.size()<1){
                    return w.annex(id);
                }
                // auto defs = args.cast<std::vector<mim::Def*>>();
                // for (auto d : defs)
                // {
                //     std::cout << d << std::endl;
                // }
                return w.call(id, mim::Defs(args));
            },
            pybind11::arg("sym"),
            pybind11::arg("args") = std::vector<mim::Def*>() 
        )
        .def("optimize", [](mim::World& w) { 
            std::cout << "printing world externals: " << std::endl;
            for(auto[sym, _] :  w.externals().sym2mut()){
                std::cout << sym.str() << std::endl;
            }
            std::cout << "----" << std::endl;
            mim::optimize(w); })
        .def("dot", static_cast<void (World::*)(const char*, bool, bool) const>(&mim::World::dot))
        .def("mut_con", [](mim::World& w, std::vector<Def*> domains){
            return w.mut_con(Defs(domains));
        })
        .def("annex_by_id", [](mim::World& w, uint64_t id){
            return w.annex(id);
        });
}
} // namespace mim
