"""
M2 — Grid idiom templates for common Triton kernel patterns.

Each idiom is a function that, given problem_shape + block_sizes,
computes tile_assignment, work[], and validates hardware-legality constraints.

Supported idioms (Phase 1):
  - 1D row-block: program_id(0) → row tiles (common in reduction, softmax)
  - 2D tile grid: program_id(0,1) → (tile_m, tile_n) (matmul, flash-attn)
  - Persistent/grouped: program_id(0) → dynamic work queue

Phase 2: general affine recovery via symbolic execution of TTIR.

Capacity limits are loaded from hardware config (configs/ascend_910b3.json)
via the calibration system. Default values are provided for compatibility.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


# Default capacity limits (910B3 values from ascend_910b3.json)
# These are used when no config is loaded; calibration system overrides these.
DEFAULT_UB_CAPACITY_BYTES = 256 * 1024   # 256 KB
DEFAULT_L1_CAPACITY_BYTES = 1 * 1024 * 1024  # 1 MB
DEFAULT_L0A_CAPACITY_BYTES = 64 * 1024  # 64 KB
DEFAULT_L0B_CAPACITY_BYTES = 64 * 1024  # 64 KB
DEFAULT_L0C_CAPACITY_BYTES = 256 * 1024  # 256 KB


def _load_capacity_from_config(config_path: Optional[str] = None) -> Dict[str, int]:
    """Load memory capacity limits from hardware config JSON.

    Args:
        config_path: Path to config JSON (e.g., configs/ascend_910b3.json).
                    If None, uses default 910B3 capacities.

    Returns:
        Dict with keys: ub, l1, l0a, l0b, l0c (values in bytes).
    """
    if config_path is None:
        # Try to find ascend_910b3.json relative to project root
        project_root = Path(__file__).parents[2]  # perfbound/extract/ → perfbound/ → vTriton/
        config_path = project_root / "configs" / "ascend_910b3.json"

    config_file = Path(config_path)
    if not config_file.exists():
        return {
            "ub": DEFAULT_UB_CAPACITY_BYTES,
            "l1": DEFAULT_L1_CAPACITY_BYTES,
            "l0a": DEFAULT_L0A_CAPACITY_BYTES,
            "l0b": DEFAULT_L0B_CAPACITY_BYTES,
            "l0c": DEFAULT_L0C_CAPACITY_BYTES,
        }

    with open(config_file) as f:
        config = json.load(f)

    mem_spaces = config.get("memory_spaces", {})
    return {
        "ub": int(mem_spaces.get("ub", {}).get("size_kb", 256)) * 1024,
        "l1": int(mem_spaces.get("l1", {}).get("size_kb", 1024)) * 1024,
        "l0a": int(mem_spaces.get("l0a", {}).get("size_kb", 64)) * 1024,
        "l0b": int(mem_spaces.get("l0b", {}).get("size_kb", 64)) * 1024,
        "l0c": int(mem_spaces.get("l0c", {}).get("size_kb", 256)) * 1024,
    }


# Cached capacities (lazy-loaded on first call)
_cached_capacities: Optional[Dict[str, int]] = None


def get_capacities(config_path: Optional[str] = None, force_reload: bool = False) -> Dict[str, int]:
    """Get memory capacity limits, loading from config if not cached.

    Args:
        config_path: Path to hardware config JSON.
        force_reload: If True, reload even if cached.

    Returns:
        Dict with capacity limits in bytes.
    """
    global _cached_capacities
    if _cached_capacities is None or force_reload:
        _cached_capacities = _load_capacity_from_config(config_path)
    return _cached_capacities


@dataclass
class TileIdiomResult:
    """Output from an idiom template."""
    tile_assignment: Dict[int, Tuple[int, ...]]
    work: Dict[int, float]
    buffer_pressure_ok: bool = True
    divisibility_ok: bool = True


def idiom_1d_row_block(
    M: int,
    BLOCK_M: int,
    ub_limit_bytes: Optional[int] = None,
    l1_limit_bytes: Optional[int] = None,
    elem_bytes: int = 2,                  # fp16
) -> TileIdiomResult:
    """1D row-block idiom: program_id(0) partitions M into BLOCK_M tiles.

    Grid: (ceil(M / BLOCK_M),)
    Each program handles BLOCK_M rows.

    Args:
        M: Total problem dimension (rows).
        BLOCK_M: Block size per program.
        ub_limit_bytes: UB capacity in bytes. If None, loads from config.
        l1_limit_bytes: L1 capacity in bytes. If None, loads from config.
        elem_bytes: Bytes per element (default: 2 for FP16).

    Returns:
        TileIdiomResult with tile assignment and validity checks.
    """
    caps = get_capacities()
    if ub_limit_bytes is None:
        ub_limit_bytes = caps["ub"]
    if l1_limit_bytes is None:
        l1_limit_bytes = caps["l1"]

    G = math.ceil(M / BLOCK_M)
    tile_assignment = {}
    work = {}
    for p in range(G):
        rows = min(BLOCK_M, M - p * BLOCK_M)
        tile_assignment[p] = (p * BLOCK_M,)
        work[p] = rows

    # Hardware-legality: check UB can hold one block
    buffer_ok = BLOCK_M * elem_bytes <= ub_limit_bytes
    div_ok = True  # 1D has no divisibility constraint beyond BLOCK_M > 0

    return TileIdiomResult(
        tile_assignment=tile_assignment,
        work=work,
        buffer_pressure_ok=buffer_ok,
        divisibility_ok=div_ok,
    )


def idiom_2d_tile_grid(
    M: int, N: int,
    BLOCK_M: int, BLOCK_N: int,
    ub_limit_bytes: Optional[int] = None,
    l1_limit_bytes: Optional[int] = None,
    elem_bytes: int = 2,
) -> TileIdiomResult:
    """2D tile grid idiom: program_id(0,1) → (tile_m, tile_n).

    Grid: (ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    Standard for matmul, flash-attention output loop.

    Args:
        M, N: Total problem dimensions.
        BLOCK_M, BLOCK_N: Block sizes per program.
        ub_limit_bytes: UB capacity in bytes. If None, loads from config.
        l1_limit_bytes: L1 capacity in bytes. If None, loads from config.
        elem_bytes: Bytes per element (default: 2 for FP16).

    Returns:
        TileIdiomResult with tile assignment and validity checks.
    """
    caps = get_capacities()
    if ub_limit_bytes is None:
        ub_limit_bytes = caps["ub"]
    if l1_limit_bytes is None:
        l1_limit_bytes = caps["l1"]

    G_m = math.ceil(M / BLOCK_M)
    G_n = math.ceil(N / BLOCK_N)
    G = G_m * G_n

    tile_assignment = {}
    work = {}
    for p in range(G):
        tile_m = p // G_n
        tile_n = p % G_n
        rows = min(BLOCK_M, M - tile_m * BLOCK_M)
        cols = min(BLOCK_N, N - tile_n * BLOCK_N)
        tile_assignment[p] = (tile_m * BLOCK_M, tile_n * BLOCK_N)
        work[p] = rows * cols

    # Hardware-legality
    buffer_ok = (BLOCK_M * BLOCK_N * elem_bytes <= ub_limit_bytes and
                 BLOCK_M * elem_bytes <= l1_limit_bytes)
    div_ok = True

    return TileIdiomResult(
        tile_assignment=tile_assignment,
        work=work,
        buffer_pressure_ok=buffer_ok,
        divisibility_ok=div_ok,
    )


# Registry of idioms for template matching
IDIOM_REGISTRY = {
    "1d_row_block": idiom_1d_row_block,
    "2d_tile_grid": idiom_2d_tile_grid,
    # Phase 1 additions: persistent, grouped, batched
}
