"""Launcher-compatible vector-add kernel for multi-kernel validation (US-SB-005).

Unlike test/triton_hivm_launch_smoke.py (which has a bare main() that returns
None and is incompatible with scripts/kernel_launcher.py's output capture), this
exposes the standard build_inputs()/Model contract so remote_bench can profile
it AND dump its output for correctness.

A large (16 M element) fp32 add: an MTE/HBM-bound kernel — a useful soundness
data point distinct from the compute-bound chunk_kda.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


N_ELEMENTS = 16 * 1024 * 1024  # 16 M
BLOCK = 2048


def build_inputs():
    device = "npu"
    torch.manual_seed(0)
    x = torch.randn(N_ELEMENTS, device=device, dtype=torch.float32)
    y = torch.randn(N_ELEMENTS, device=device, dtype=torch.float32)
    out = torch.empty_like(x)
    return {"x": x, "y": y, "out": out}


class Model(nn.Module):
    def forward(self, data):
        x, y, out = data["x"], data["y"], data["out"]
        grid = (triton.cdiv(N_ELEMENTS, BLOCK),)
        add_kernel[grid](x, y, out, N_ELEMENTS, BLOCK=BLOCK)
        return (out,)


def reference(x, y):
    """CPU reference for correctness checks (run_counterfactual reference_fn)."""
    return x + y


if __name__ == "__main__":
    data = build_inputs()
    out = Model().forward(data)[0]
    print("vector_add launch OK", out.shape, out.dtype)
