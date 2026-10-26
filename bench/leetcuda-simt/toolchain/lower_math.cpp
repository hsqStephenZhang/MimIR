#include <llvm/ADT/STLExtras.h>
#include <llvm/CodeGen/IntrinsicLowering.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Plugins/PassPlugin.h>

namespace {

class LowerMath : public llvm::PassInfoMixin<LowerMath> {
public:
    llvm::PreservedAnalyses run(llvm::Module& module, llvm::ModuleAnalysisManager&) {
        llvm::IntrinsicLowering lowering(module.getDataLayout());
        bool changed = false;
        for (auto& function : module) {
            for (auto& instruction : llvm::make_early_inc_range(llvm::instructions(function))) {
                if (auto call = llvm::dyn_cast<llvm::IntrinsicInst>(&instruction)) {
                    if (call->getIntrinsicID() == llvm::Intrinsic::pow
                        && (call->getType()->isFloatTy() || call->getType()->isDoubleTy())) {
                        lowering.LowerIntrinsicCall(call);
                        changed = true;
                    }
                }
            }
        }
        return changed ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
    }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "mim-nvptx-math", LLVM_VERSION_STRING, [](llvm::PassBuilder& builder) {
                builder.registerPipelineParsingCallback([](llvm::StringRef name, llvm::ModulePassManager& manager,
                                                           llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                    if (name != "mim-nvptx-math") return false;
                    manager.addPass(LowerMath{});
                    return true;
                });
            }};
}
