#!/bin/bash
# Build bishengir using AscendNPU-IR's own LLVM (19.1.7)
set -euo pipefail

BISHENGIR_SRC="/mnt/d/work/git/triton-ascend/third_party/ascend/AscendNPU-IR"
NPROC=$(nproc 2>/dev/null || echo 8)

# Ensure upgraded ninja is in PATH
export PATH="$HOME/.local/bin:$PATH"
echo "Ninja: $(which ninja) -> $(ninja --version)"
echo "CMake: $(cmake --version | head -1)"
echo "Jobs: ${NPROC}"
echo ""

cd "${BISHENGIR_SRC}"
echo "Building from: $(pwd)"
echo "LLVM submodule: $(cd third-party/llvm-project && git log --oneline -1)"
echo ""

"${BISHENGIR_SRC}/build-tools/build.sh" \
  --apply-patches \
  --build-type Release \
  --fast-build \
  --add-cmake-options '-DLLVM_ENABLE_RTTI=OFF -DLLVM_ENABLE_EH=OFF' \
  -j "${NPROC}"

echo ""
echo "=== BUILD COMPLETE ==="
echo "Output: ${BISHENGIR_SRC}/build/"
echo "LLVM install: ${BISHENGIR_SRC}/build/install/"
echo ""
echo "Verify artifacts:"
ls -la "${BISHENGIR_SRC}/build/install/lib/cmake/mlir/MLIRConfig.cmake" 2>/dev/null && echo "  MLIR cmake config: OK"
ls -la "${BISHENGIR_SRC}/build/lib/Dialect/HIVM/IR/libBiShengIRHIVMDialect.a" 2>/dev/null && echo "  HIVM dialect lib: OK"
find "${BISHENGIR_SRC}/build/include/bishengir/Dialect/HIVM" -name "*.h.inc" 2>/dev/null | head -3 && echo "  HIVM tablegen output: OK"
