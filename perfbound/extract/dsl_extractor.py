# M2 — DSL Extractor / Tier 1 input.
#
# Parse the @triton.jit function + launch grid + shape → Tier-1 quantities:
#   G, tile_assignment[p], occupancy, work[p], load_balance, redundancy.
#
# Method: recover the affine map from tl.program_id → tile via TTIR
# (tt.get_program_id, tt.load pointer arithmetic) using C++ MLIR pass.
# Common idioms as templates first (grid_idioms.py), general affine recovery second.
#
# TTIR extraction now uses C++ MLIR pass (mlir_parser.py) instead of regex.
# No fragile text matching — the pass walks the MLIR AST and extracts:
#   - grid axes from tt.get_program_id
#   - persistent loops from scf.for where lb = program_id
#   - tile shapes from tt.make_tensor_ptr result types
#   - Cube kernel detection from tt.dot presence

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .grid_idioms import (
    idiom_1d_row_block, idiom_2d_tile_grid, TileIdiomResult,
)
from .mlir_parser import parse_ttir


# ── Grid info dataclass ────────────────────────────────────────────────────

@dataclass
class GridInfo:
    """Tier 1 grid-level quantities for the bound model."""
    # Launch grid dimensions
    grid_dims: Tuple[int, ...]          # (G_x, G_y, G_z) from launch
    total_programs: int                 # G = product(grid_dims)

    # Per-program tile assignment: program_id → (tile_m, tile_n, ...)
    tile_assignment: Dict[int, Tuple[int, ...]]

    # Per-program work amount (e.g., elements computed)
    work: Dict[int, float]

    # Derived quantities
    occupancy: float                    # min(G, n_cores) / n_cores
    load_balance: float                 # mean(work) / max(work)
    redundancy: float = 1.0            # GM read amplification (default 1)

    # Busiest core (largest work)
    busiest_core_id: int = 0

    # Hardware-legality constraints (from configs/ascend_910b3.json)
    buffer_pressure_ok: bool = True
    divisibility_ok: bool = True

    @property
    def is_valid(self) -> bool:
        return self.buffer_pressure_ok and self.divisibility_ok


# ── Main extractor ─────────────────────────────────────────────────────────

def extract_grid_info(
    kernel_source: str,
    launch_grid: Tuple[int, ...],
    problem_shape: Tuple[int, ...],
    block_sizes: Dict[str, int],
    n_cores: int = 20,
) -> GridInfo:
    """Extract Tier 1 grid information from a Triton kernel.

    Supports two input modes:
    1. TTIR file path (detected by .ttir or .mlir extension or tt.func content)
       → parses TTIR using C++ MLIR pass to recover grid, tile shapes, and affine maps.
    2. Triton Python source or "python" placeholder → uses grid idioms (grid_idioms.py) with
       problem_shape + block_sizes to compute tile assignment.

    Args:
        kernel_source: Triton kernel source code OR path to TTIR dump file OR "python".
        launch_grid: Launch grid dimensions (G_x, G_y, G_z).
        problem_shape: Problem dimensions (M, N, K, ...).
        block_sizes: Block/tile sizes {BLOCK_M: 128, BLOCK_N: 64, ...}.
        n_cores: Number of available cores (20 for Cube, 40 for Vector-only).

    Returns:
        GridInfo with all Tier 1 quantities.

    Raises:
        NotImplementedError: For grid idioms not yet supported.
    """
    # Detect TTIR mode
    is_ttir = False
    source_path = Path(kernel_source) if "\n" not in kernel_source else None

    # Handle "python" source type (explicit marker for idiom path)
    if kernel_source == "python":
        return _extract_from_idioms("", launch_grid, problem_shape,
                                    block_sizes, n_cores)

    # Check if source is a TTIR file path
    if source_path and source_path.exists() and source_path.is_file():
        ttir_text = source_path.read_text()
        is_ttir = bool("tt.func" in ttir_text or "tt.get_program_id" in ttir_text)
    elif "tt.func" in kernel_source or "tt.get_program_id" in kernel_source:
        raise ValueError(
            "Inline TTIR text is not supported. "
            "Write the TTIR to a file and pass the file path instead."
        )

    if is_ttir:
        return _extract_from_ttir(source_path or ttir_text, launch_grid,
                                  problem_shape, block_sizes, n_cores)
    else:
        return _extract_from_idioms(kernel_source, launch_grid, problem_shape,
                                    block_sizes, n_cores)


def _extract_from_ttir(
    ttir_source: str | Path,
    launch_grid: Tuple[int, ...],
    problem_shape: Tuple[int, ...],
    block_sizes: Dict[str, int],
    n_cores: int,
) -> GridInfo:
    """Extract grid info from TTIR using C++ MLIR pass."""
    # Parse TTIR using C++ pass (no regex!)
    info = parse_ttir(ttir_source)

    axes = info["grid_axes"]           # [0] for x-axis, [0,1] for 2D, etc.
    loops = info["persistent_loops"]   # [{"lb_is_pid": true, "ub_value": 4, "step_value": 20}]
    shapes = [tuple(s) for s in info["tensor_ptr_shapes"]]  # [(32, 128), (128, 128), ...]
    has_dot = info["has_dot"]         # true if Cube kernel

    # Detect idiom from TTIR structure
    is_persistent = any(l.get("lb_is_pid", False) for l in loops) and len(axes) == 1
    is_2d_grid = len(axes) >= 2

    if is_persistent:
        # === Bug 3 Fix: Use correct persistent kernel work model ===
        return _persistent_kernel_info(loops, shapes, problem_shape,
                                       launch_grid, n_cores)

    if is_2d_grid or (shapes and len(shapes) >= 2 and len(shapes[0]) >= 2):
        # 2D tile grid
        if not block_sizes and shapes:
            shape0 = shapes[0]
            if len(shape0) >= 2:
                block_sizes = {"BLOCK_M": shape0[0], "BLOCK_N": shape0[1]}
        if "BLOCK_M" in block_sizes and "BLOCK_N" in block_sizes:
            M = problem_shape[0] if len(problem_shape) > 0 else 1
            N = problem_shape[1] if len(problem_shape) > 1 else 1
            result = idiom_2d_tile_grid(M, N, block_sizes["BLOCK_M"],
                                        block_sizes["BLOCK_N"])
            return _idiom_to_grid(result, launch_grid, n_cores)

    if len(axes) == 1 and shapes:
        # 1D row-block
        if not block_sizes and shapes:
            shape0 = shapes[0]
            blk = shape0[0] if shape0 else 128
            block_sizes = {"BLOCK_M": blk}
        if "BLOCK_M" in block_sizes:
            M = problem_shape[0] if len(problem_shape) > 0 else 1
            result = idiom_1d_row_block(M, block_sizes["BLOCK_M"])
            return _idiom_to_grid(result, launch_grid, n_cores)

    # Fallback: uniform work assumption for exotic patterns
    return _uniform_grid(launch_grid, n_cores)


def _persistent_kernel_info(
    loops: List[Dict],
    shapes: List[Tuple[int, ...]],
    problem_shape: Tuple[int, ...],
    launch_grid: Tuple[int, ...],
    n_cores: int,
) -> GridInfo:
    """Build GridInfo for persistent kernel using correct work model.

    Bug 3 Fix:
    - Use round-robin tile assignment: program p gets tiles p, p+stride, p+2*stride, ...
    - Derive block_m from tensor_ptr_shapes[0][0] (e.g., 32 for flash_attention)
    - Derive total_tiles from problem_shape (ceil(M/block_m))
    - n_programs = stride (e.g., 20 for flash_attention)
    - occupancy = active_programs / n_programs (e.g., 4/20 = 0.2)
    - load_balance = mean(work) / max(work) (e.g., 1.0 for flash_attention)
    """
    # Find the persistent loop (where lb_is_pid=true)
    pers_loop = next((l for l in loops if l.get("lb_is_pid", False)), None)
    if not pers_loop:
        # Fallback if no persistent loop detected
        return _uniform_grid(launch_grid, n_cores)

    stride = pers_loop["step_value"]        # = n_programs = n_cores (e.g., 20)
    ub_value = pers_loop["ub_value"]        # upper bound (e.g., 4)

    # Derive block_m from tensor_ptr_shapes; fall back to 128
    block_m = 128
    if shapes and len(shapes[0]) > 0:
        block_m = shapes[0][0]  # e.g., 32 for flash_attention

    # Derive total_tiles from problem_shape (authoritative)
    # ub_value is cross-check (should be >= total_tiles)
    M = problem_shape[0] if len(problem_shape) > 0 else 1
    total_tiles = math.ceil(M / block_m)

    # If ub_value is smaller, use it (safety check)
    if ub_value > 0 and ub_value < total_tiles:
        total_tiles = ub_value

    n_programs = stride if stride > 0 else n_cores

    # Round-robin tile assignment
    tile_assignment = {}
    work = {}
    for p in range(n_programs):
        # Program p gets tiles: p, p+stride, p+2*stride, ... < total_tiles
        my_tiles = [t for t in range(p, total_tiles, n_programs)]
        if my_tiles:
            # Convert tile indices to actual row offsets (tile index * block_m)
            tile_assignment[p] = tuple(t * block_m for t in my_tiles)
            work[p] = len(my_tiles)  # tile count
        else:
            tile_assignment[p] = tuple()
            work[p] = 0

    # Compute occupancy and load balance (Bug 3 fix)
    active = [w for w in work.values() if w > 0]
    n_active = len(active)
    max_w = max(active) if active else 1
    occ = n_active / n_programs if n_programs > 0 else 1.0

    # load_balance = mean(work) / max(work) across active programs only
    if active:
        lb = sum(active) / (max_w * n_active)
    else:
        lb = 1.0

    # Find busiest core
    busiest_core_id = max(work, key=work.get) if work else 0

    return GridInfo(
        grid_dims=(n_programs,),
        total_programs=n_programs,
        tile_assignment=tile_assignment,
        work=work,
        occupancy=occ,
        load_balance=lb,
        redundancy=1.0,
        busiest_core_id=busiest_core_id,
        buffer_pressure_ok=True,
        divisibility_ok=True,
    )


def _extract_from_idioms(
    kernel_source: str,
    launch_grid: Tuple[int, ...],
    problem_shape: Tuple[int, ...],
    block_sizes: Dict[str, int],
    n_cores: int,
) -> GridInfo:
    """Extract grid info from Python Triton source using idiom templates."""
    # Try to match against known idioms
    # Check for 2D: either program_id(0) and program_id(1) in source, OR both BLOCK_M and BLOCK_N provided
    is_2d = ("program_id(0)" in kernel_source and "program_id(1)" in kernel_source)
    is_2d = is_2d or ("BLOCK_M" in block_sizes and "BLOCK_N" in block_sizes)

    if is_2d and "BLOCK_M" in block_sizes and "BLOCK_N" in block_sizes:
        M = problem_shape[0] if len(problem_shape) > 0 else 1
        N = problem_shape[1] if len(problem_shape) > 1 else 1
        result = idiom_2d_tile_grid(M, N, block_sizes["BLOCK_M"],
                                    block_sizes["BLOCK_N"])
        return _idiom_to_grid(result, launch_grid, n_cores)

    if "BLOCK_M" in block_sizes:
        M = problem_shape[0] if len(problem_shape) > 0 else 1
        result = idiom_1d_row_block(M, block_sizes["BLOCK_M"])
        return _idiom_to_grid(result, launch_grid, n_cores)

    # Fallback
    return _uniform_grid(launch_grid, n_cores)


def _idiom_to_grid(
    result: TileIdiomResult,
    launch_grid: Tuple[int, ...],
    n_cores: int,
) -> GridInfo:
    """Convert a TileIdiomResult to GridInfo."""
    G = 1
    for d in launch_grid:
        G *= d

    works = list(result.work.values())
    occupancy = min(G, n_cores) / n_cores if n_cores > 0 else 1.0
    load_balance = sum(works) / (max(works) * len(works)) if works and max(works) > 0 else 1.0
    busiest = max(result.work, key=result.work.get) if result.work else 0

    return GridInfo(
        grid_dims=launch_grid,
        total_programs=G,
        tile_assignment=result.tile_assignment,
        work=result.work,
        occupancy=occupancy,
        load_balance=load_balance,
        redundancy=1.0,
        busiest_core_id=busiest,
        buffer_pressure_ok=result.buffer_pressure_ok,
        divisibility_ok=result.divisibility_ok,
    )


def _uniform_grid(
    launch_grid: Tuple[int, ...],
    n_cores: int,
) -> GridInfo:
    """Create a uniform-work GridInfo (fallback for unknown idioms)."""
    G = 1
    for d in launch_grid:
        G *= d

    work = {p: 1.0 for p in range(G)}
    occupancy = min(G, n_cores) / n_cores if n_cores > 0 else 1.0

    return GridInfo(
        grid_dims=launch_grid,
        total_programs=G,
        tile_assignment={},
        work=work,
        occupancy=occupancy,
        load_balance=1.0,
        redundancy=1.0,
        busiest_core_id=0,
    )
