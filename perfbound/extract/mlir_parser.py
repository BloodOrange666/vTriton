"""
MLIR Parser — Wrapper for C++ ExtractTTIRInfo pass.

This module wraps the C++ ExtractTTIRInfo pass in tritonsim-opt, which walks
the TTIR MLIR AST and emits structured JSON. No regex/text parsing in Python.

Note: The implementation uses a direct pipeline (tritonsim-opt --extract-ttir-info)
instead of the two-stage pipeline (triton-opt | tritonsim-opt) because generic
preprocessing can convert tt.* ops to their generic representation, preventing
the C++ pass from recognizing them via triton::GetProgramIdOp, triton::MakeTensorPtrOp, etc.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


# Paths to binaries (relative to project root)
# mlir_parser.py is at perfbound/extract/mlir_parser.py → parents[2] is project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRITONSIM_OPT = PROJECT_ROOT / "build" / "bin" / "tritonsim-opt"
TRITON_OPT = Path("/mnt/d/work/git/triton-ascend/python/build"
                  "/cmake.linux-x86_64-cpython-3.12/bin/triton-opt")


def parse_ttir(ttir_path: str | Path) -> Dict:
    """Run the C++ ExtractTTIRInfoPass and return parsed JSON.

    Args:
        ttir_path: Path to .ttir or .mlir file containing Triton IR.

    Returns:
        Dict with keys:
        - grid_axes: List[int] - program_id axis indices (0=x, 1=y, 2=z)
        - persistent_loops: List[Dict] - scf.for loops where lb=program_id
        - tensor_ptr_shapes: List[List[int]] - shapes of tt.make_tensor_ptr results
        - has_dot: bool - true if tt.dot op is present

    Raises:
        subprocess.CalledProcessError: If subprocess fails.
        json.JSONDecodeError: If output is not valid JSON.
        FileNotFoundError: If tritonsim-opt binary not found.
    """
    ttir_path = Path(ttir_path)
    if not ttir_path.exists():
        raise FileNotFoundError(f"TTIR file not found: {ttir_path}")

    tritonsim = TRITONSIM_OPT
    if not tritonsim.exists():
        raise FileNotFoundError(
            f"tritonsim-opt not found at {tritonsim}. "
            "Build the project first: cd build && ninja"
        )

    # Run the pass directly on TTIR (no triton-opt preprocessing needed)
    cmd = [str(tritonsim), str(ttir_path.absolute()),
           "--allow-unregistered-dialect",
           "--extract-ttir-info"]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )

    # The output contains JSON followed by the module dump.
    # Extract just the JSON part (everything from the opening { to closing })
    stdout = result.stdout
    json_start = stdout.find("{")
    if json_start == -1:
        raise json.JSONDecodeError(
            "No JSON found in output",
            stdout,
            0
        )

    # Find the matching closing brace by counting braces
    brace_count = 0
    json_end = -1
    for i in range(json_start, len(stdout)):
        if stdout[i] == "{":
            brace_count += 1
        elif stdout[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                json_end = i + 1
                break

    if json_end == -1:
        raise json.JSONDecodeError(
            "Incomplete JSON in output",
            stdout,
            json_start
        )

    json_str = stdout[json_start:json_end]
    return json.loads(json_str)


# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ttir_file = sys.argv[1]
    else:
        ttir_file = Path(__file__).parents[3] / "test" / "flash_attention.ttir"

    try:
        info = parse_ttir(ttir_file)
        print(f"Successfully parsed {ttir_file}")
        print(f"  grid_axes: {info['grid_axes']}")
        print(f"  persistent_loops: {len(info['persistent_loops'])} loops")
        print(f"  tensor_ptr_shapes: {len(info['tensor_ptr_shapes'])} shapes")
        print(f"  has_dot: {info['has_dot']}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
