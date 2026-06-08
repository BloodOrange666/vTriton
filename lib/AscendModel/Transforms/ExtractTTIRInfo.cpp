//===- ExtractTTIRInfo.cpp - Extract TTIR structural info as JSON ---------===//
//
// This pass walks a TTIR module and extracts structural information needed
// for Tier 1 grid modeling: grid axes, persistent loops, tile shapes, and
// whether the kernel uses the Cube unit (tt.dot).
//
// The output is JSON emitted to stdout, suitable for consumption by Python.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

// Forward declaration for Triton dialect (if available)
#ifdef TRITONSIM_HAS_TRITON
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Traits.h"
#endif

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_EXTRACTTTIRINFOPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Get integer constant value from arith.constant, or nullopt if not a constant.
static std::optional<int64_t> getConstantIntValue(Value value) {
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ExtractTTIRInfoPass
//===----------------------------------------------------------------------===//

struct ExtractTTIRInfoPass
    : public impl::ExtractTTIRInfoPassBase<ExtractTTIRInfoPass> {
  using ExtractTTIRInfoPassBase::ExtractTTIRInfoPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // JSON output structure
    llvm::json::Object output;

    // 1. Extract grid axes from tt.get_program_id
    llvm::json::Array gridAxesArr;
#ifdef TRITONSIM_HAS_TRITON
    module.walk([&](triton::GetProgramIdOp op) {
      int32_t axis = op.getAxisAsInt();
      gridAxesArr.push_back(llvm::json::Value(axis));
    });
#endif
    output["grid_axes"] = std::move(gridAxesArr);

    // 2. Extract persistent loops (scf.for where lb = program_id)
    llvm::json::Array persistentLoopsArr;
    module.walk([&](scf::ForOp forOp) {
      // Check if lower bound is defined by GetProgramIdOp
      bool lbIsPid = false;
#ifdef TRITONSIM_HAS_TRITON
      if (auto pidOp = forOp.getLowerBound()
                            .getDefiningOp<triton::GetProgramIdOp>()) {
        lbIsPid = true;
      }
#endif

      // Get upper bound and step as constants if available
      auto ubVal = getConstantIntValue(forOp.getUpperBound());
      auto stepVal = getConstantIntValue(forOp.getStep());

      // Only emit loops where lower bound is program_id (persistent pattern)
      if (lbIsPid) {
        llvm::json::Object loopObj;
        loopObj["lb_is_pid"] = true;
        loopObj["ub_value"] = ubVal.value_or(-1);
        loopObj["step_value"] = stepVal.value_or(-1);
        persistentLoopsArr.push_back(std::move(loopObj));
      }
    });
    output["persistent_loops"] = std::move(persistentLoopsArr);

    // 3. Extract tensor pointer shapes from tt.make_tensor_ptr result types
    llvm::json::Array tensorPtrShapesArr;
#ifdef TRITONSIM_HAS_TRITON
    module.walk([&](triton::MakeTensorPtrOp op) {
      // Get the result type (should be a pointer to tensor)
      auto resultType = op.getResult().getType();

      // Try to extract tensor shape from the pointer's pointee type
      // The result type is !tt.ptr<tensor<SHAPE...>>
      // We need to get the tensor type and then its shape
      RankedTensorType tensorType;
      if (auto ptrType = dyn_cast<triton::PointerType>(resultType)) {
        tensorType = dyn_cast<RankedTensorType>(ptrType.getPointeeType());
      }

      if (tensorType) {
        llvm::json::Array dims;
        for (int64_t dim : tensorType.getShape()) {
          dims.push_back(llvm::json::Value(dim));
        }
        tensorPtrShapesArr.push_back(llvm::json::Value(std::move(dims)));
      }
    });
#endif
    output["tensor_ptr_shapes"] = std::move(tensorPtrShapesArr);

    // 4. Detect if kernel uses Cube (has tt.dot)
    bool hasDot = false;
#ifdef TRITONSIM_HAS_TRITON
    module.walk([&](triton::DotOp) { hasDot = true; });
#endif
    output["has_dot"] = hasDot;

    // Emit JSON to stdout
    llvm::json::Value outputValue(std::move(output));
    llvm::outs() << llvm::formatv("{0:2}\n", outputValue);
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
