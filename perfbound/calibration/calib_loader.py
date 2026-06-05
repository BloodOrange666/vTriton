# M1 — Load and validate a calibration JSON file.
#
# The calibration file (perfbound/data/calib_910b3_v1.json) is the single
# source of truth for all sustained hardware rates.  Every P0 constant must
# be measured (≥30 runs, <5% CI) before the bound model is valid.
#
# Schema: CalibrationDB → JSON via to_dict() / from_dict().

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .constants import (
    CalibrationDB,
    CalibrationConstant,
    CoreConfig,
    CubeConfig,
    VectorConfig,
    MemHierarchy,
    MemBandwidth,
    DType,
    MemLoc,
)

DEFAULT_CALIB_PATH = Path(__file__).parent.parent / "data" / "calib_910b3_v1.json"


def _parse_constant(d: dict) -> CalibrationConstant:
    """Parse a CalibrationConstant from a JSON dict entry."""
    return CalibrationConstant(
        name=d.get("name", ""),
        value=float(d["value"]),
        unit=d.get("unit", ""),
        ci_95=float(d.get("ci_95", 0.0)),
        source=d.get("source", "unknown"),
        n_runs=int(d.get("n_runs", 0)),
        notes=d.get("notes", ""),
    )


def load_calibration(path: str | Path | None = None) -> CalibrationDB:
    """Load and validate a calibration JSON file.

    Args:
        path: Path to calib JSON file.  Defaults to perfbound/data/calib_910b3_v1.json.

    Returns:
        CalibrationDB with all sustained-rate constants.

    Raises:
        FileNotFoundError: If the calibration file does not exist.
    """
    if path is None:
        path = DEFAULT_CALIB_PATH
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {path}\n"
            "Run M1 CCE microbenchmarks first to produce it."
        )

    with open(path) as f:
        raw = json.load(f)

    # Use the built-in deserializer (handles nesting)
    db = CalibrationDB.from_dict(raw)

    # Load bandwidth CSV if present alongside the JSON
    bw_csv = path.with_suffix("").with_suffix(".csv")
    if bw_csv.exists():
        _load_bandwidth_csv(db, bw_csv)

    return db


def _load_bandwidth_csv(db: CalibrationDB, csv_path: Path) -> None:
    """Load sustained bandwidths from a companion CSV file (tilesim format).

    CSV columns: src_mem, dst_mem, core_num, pkt_size, mode, bandwidth(GB/s)
    """
    import csv

    with open(csv_path) as f:
        # Skip comment lines (starting with #)
        lines = [line for line in f if not line.strip().startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        src = row["src_mem"].strip().lower()
        dst = row["dst_mem"].strip().lower()
        core_num = int(row.get("core_num", -1))
        bw = MemBandwidth.from_csv_row(row)
        db.memory.bw[(src, dst, core_num)] = bw


def validate_calibration(db: CalibrationDB) -> list[str]:
    """Check calibration provenance for soundness-critical issues.

    Returns a list of warning strings.  An empty list means the DB is
    fully calibrated for bound computation.
    """
    warnings = []

    # 1. P0 constants with insufficient provenance
    p0_keys = [
        "P_cube_fp16", "P_cube_int8",
        "P_vector_add_fp16", "P_vector_mul_fp16", "P_vector_exp_fp16",
        "BW_gm_l1", "BW_gm_ub", "BW_l1_l0", "BW_ub_gm", "BW_l0c_gm",
        "BW_hbm_allcore",
        "mandatory_handoff",
    ]
    for key in p0_keys:
        c = db.constants.get(key)
        if c is None:
            warnings.append(f"{key}: not present in calibration DB")
        elif not c.is_valid:
            warnings.append(
                f"{key}: n_runs={c.n_runs} (need ≥30), "
                f"ci_rel={c.ci_rel:.3f} (need <0.05)"
            )

    # 2. Mandatory handoff cost required for T_serial_irreducible
    if db.mandatory_handoff_cycles <= 0:
        warnings.append(
            "mandatory_handoff_cycles not measured — "
            "T_serial_irreducible will be 0 (unsound for Cube↔Vector kernels)"
        )

    # 3. Scalar overhead required for kernel-level cycle estimates
    if db.scalar_overhead_factor <= 1.0:
        warnings.append(
            "scalar_overhead_factor not calibrated — "
            "kernel-level bounds may be optimistic"
        )

    return warnings


def seed_calibration_from_ascend_json(
    ascend_json_path: str | Path,
    output_path: str | Path | None = None,
) -> CalibrationDB:
    """Seed a CalibrationDB from the existing configs/ascend_910b.json.

    Tags every constant with source="datasheet_seed" and n_runs=0.
    These MUST be replaced by measured sustained rates before any
    I_c computation enters the bound.

    Args:
        ascend_json_path: Path to configs/ascend_910b.json.
        output_path: If given, write the seeded DB as JSON.

    Returns:
        CalibrationDB seeded from the ascend hardware config.
    """
    with open(ascend_json_path) as f:
        cfg = json.load(f)

    db = CalibrationDB(
        version="v0-seed",
        hardware_name=cfg.get("name", "Ascend 910B"),
        description=f"Seeded from {ascend_json_path} — all constants are datasheet peaks, NOT measured.",
    )

    # Core config
    clock = cfg.get("clock", {})
    db.core = CoreConfig(
        clock_freq_ghz=clock.get("frequency_ghz", 1.85),
    )
    calib = cfg.get("calibration", {})
    parallelism = calib.get("parallelism", {})
    db.core.aic_core_num = parallelism.get("num_aic_cores", 20)
    db.core.aiv_core_num = parallelism.get("num_aiv_cores", 40)

    # Cube throughput (datasheet seeds)
    cube_cfg = cfg.get("compute_units", {}).get("cube", {})
    for dtype_str, tflops in [
        ("fp16", cube_cfg.get("tflops_fp16", 320)),
        ("bf16", cube_cfg.get("tflops_fp16", 320)),
        ("fp32", cube_cfg.get("tflops_fp32", 160)),
        ("int8", cube_cfg.get("tflops_int8", 640)),
    ]:
        dtype = DType.from_str(dtype_str)
        db.cube.throughput[dtype] = float(tflops)
        db.constants[f"P_cube_{dtype_str}"] = CalibrationConstant(
            name=f"P_cube_{dtype_str}",
            value=float(tflops),
            unit="TFLOPS",
            ci_95=0.0,
            source="datasheet_seed",
            n_runs=0,
            notes="DATASHEET PEAK — replace with sustained microbench measurement",
        )
    # Fractal sizes
    fractal = cube_cfg.get("fractal_sizes", {})
    for dtype_str, sizes in fractal.items():
        db.cube.fractal_sizes[DType.from_str(dtype_str)] = tuple(sizes)

    # Vector throughput
    vec_cfg = cfg.get("compute_units", {}).get("vector", {})
    db.vector.throughput_fp16_tflops = float(vec_cfg.get("tflops_fp16", 20))
    db.vector.throughput_fp32_tflops = float(vec_cfg.get("tflops_fp32", 10))

    # Vector per-op cycles (seed from calibration block — trust these)
    op_cycles = calib.get("vector_op_cycles_per_vec_instruction", {})
    _cycle_map = {
        "simple_ops_add_sub_mul_etc": [
            ("add", 1), ("sub", 1), ("mul", 1), ("max", 1), ("min", 1),
            ("neg", 1), ("abs", 1), ("relu", 1), ("cast", 1),
            ("broadcast", 1), ("select", 1),
        ],
    }
    for key, cycles in op_cycles.items():
        if key in ("notes",):
            continue
        if key == "simple_ops_add_sub_mul_etc":
            for op_name, _ in _cycle_map.get(key, []):
                db.constants[f"P_vector_{op_name}_fp16"] = CalibrationConstant(
                    name=f"P_vector_{op_name}_fp16",
                    value=float(cycles),
                    unit="cycles/128elems",
                    ci_95=0.0,
                    source="profiling_seed",
                    n_runs=1,
                    notes="Seed from flash-attn profiling. Re-validate with sweep.",
                )
        else:
            db.constants[f"P_vector_{key}_fp16"] = CalibrationConstant(
                name=f"P_vector_{key}_fp16",
                value=float(cycles),
                unit="cycles/128elems",
                ci_95=0.0,
                source="profiling_seed",
                n_runs=1,
                notes="Seed from flash-attn profiling.",
            )

    # Memory hierarchy
    mem_cfg = cfg.get("memory_spaces", {})
    hbm = mem_cfg.get("hbm", {})
    l2 = mem_cfg.get("l2", {})
    l1 = mem_cfg.get("l1", {})
    ub = mem_cfg.get("ub", {})
    l0a = mem_cfg.get("l0a", {})
    l0b = mem_cfg.get("l0b", {})
    l0c = mem_cfg.get("l0c", {})
    db.memory = MemHierarchy(
        gm_size_gb=float(hbm.get("size_gb", 32)),
        l2_size_mb=float(l2.get("size_mb", 192)),
        l1_size_kb=float(l1.get("size_kb", 1024)),
        l0a_size_kb=float(l0a.get("size_kb", 64)),
        l0b_size_kb=float(l0b.get("size_kb", 64)),
        l0c_size_kb=float(l0c.get("size_kb", 256)),
        ub_size_kb=float(ub.get("size_kb", 256)),
    )

    # MTE bandwidths (datasheet seeds)
    movers = cfg.get("data_movers", {})
    _mte_paths = [
        ("cube_mte2", "gm", "l1"),
        ("mte1", "l1", "l0"),
        ("fixpipe", "l0c", "gm"),
        ("vector_mte2", "gm", "ub"),
        ("mte3", "ub", "gm"),
    ]
    for key, src, dst in _mte_paths:
        mte = movers.get(key, {})
        bw_gb = float(mte.get("bandwidth_gbps", 200))
        name = f"BW_{src}_{dst}"
        db.constants[name] = CalibrationConstant(
            name=name,
            value=bw_gb,
            unit="GB/s",
            ci_95=0.0,
            source="datasheet_seed",
            n_runs=0,
            notes=f"DATASHEET PEAK for {key} — replace with DMA sweep microbench",
        )
        db.memory.bw[(src, dst, -1)] = MemBandwidth(
            src_mem=src,
            dst_mem=dst,
            bw_gb_per_s=bw_gb,
            alignment_bytes=int(mte.get("alignment_bytes", 32)),
            max_burst_bytes=int(mte.get("max_burst_bytes", 65536)),
        )

    # Scalar overhead (trust as seed)
    scalar = calib.get("scalar_overhead", {})
    db.scalar_overhead_factor = float(scalar.get("aiv_scalar_overhead_factor", 3.74))

    # Startup latencies (trust as seed)
    startup = calib.get("startup_latencies", {})
    db.startup_latency = {
        "vector": float(startup.get("vector_startup_cycles", 35)),
        "mte2": float(startup.get("mte2_startup_cycles", 50)),
        "mte3": float(startup.get("mte3_startup_cycles", 40)),
        "cube": float(startup.get("cube_startup_cycles", 20)),
        "fixpipe": float(startup.get("fixpipe_startup_cycles", 30)),
    }

    # Pipe barrier (for attribution only)
    barrier = calib.get("pipe_barrier", {})
    db.pipe_barrier_cycles_per_iter = float(barrier.get("cycles_per_inner_iteration", 7500))

    # Mandatory handoff (not present in seed — must measure)
    db.mandatory_handoff_cycles = 0.0

    if output_path:
        db.save(output_path)

    return db
