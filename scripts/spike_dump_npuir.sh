#!/bin/bash
# Spike script: dump NPUIR from chunk_kda on remote 910B3
set -euo pipefail

REMOTE="910B3"
REMOTE_PATH="/root/vTriton"
DUMP_DIR="${REMOTE_PATH}/ttdump"

echo "=== Step 1: Clean dump dir ==="
ssh "$REMOTE" "rm -rf $DUMP_DIR && mkdir -p $DUMP_DIR"

echo "=== Step 2: Run chunk_kda with TRITON_DEBUG=1 + TRITON_KERNEL_DUMP=1 ==="
ssh "$REMOTE" bash -s <<'REMOTE_SCRIPT'
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source $(conda info --base)/etc/profile.d/conda.sh
conda activate triton_hxl
cd /root/vTriton

# Run the kernel with debug + dump enabled
TRITON_DEBUG=1 \
TRITON_KERNEL_DUMP=1 \
TRITON_DUMP_DIR=/root/vTriton/ttdump \
TRITON_ALWAYS_COMPILE=1 \
python test/chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2.py 2>&1 || true
REMOTE_SCRIPT

echo ""
echo "=== Step 3: Search for .npuir.mlir files ==="
ssh "$REMOTE" "find $DUMP_DIR -name '*.npuir.mlir' -o -name '*.mlir' 2>/dev/null | head -20"
ssh "$REMOTE" "find /root/.triton -name '*.npuir.mlir' 2>/dev/null | head -10"

echo ""
echo "=== Step 4: List dump dir contents ==="
ssh "$REMOTE" "find $DUMP_DIR -type f 2>/dev/null | head -30"

echo ""
echo "=== Step 5: Check triton cache for npuir ==="
ssh "$REMOTE" "find /root/.triton -name '*npuir*' -newer $DUMP_DIR 2>/dev/null | head -10"
ssh "$REMOTE" "find /root/.triton/cache -name 'kernel.npuir.mlir' 2>/dev/null | head -10"

echo "=== DONE ==="
