# M5 — Bound Combiner (two-tier max + T_serial_irreducible)
#
# T_bound = max(T_grid_floor, T_core_floor + T_serial_irreducible)
#
# Composition is max (two independent lower bounds on the same wall-clock
# time).  T_serial_irreducible attaches to the Tier-2 term because
# serialization is intra-core (Cube↔Vector on the same core).
#
# **SPEC DIVERGENCE (intentional):** The spec (performance_bound_model.md §4.1,
# §7) literally writes max(T_grid_floor, T_core_floor) + T_serial_irreducible.
# This additive form is UNSOUND: max(a,b)+c ≥ max(a, b+c) for c≥0, which can
# overstate a lower bound and risk T_bound > T_measured (violating the
# conservatism theorem T_bound ≤ T_measured from spec §4.0).
#
# The implemented form (max(grid, core+serial)) is the tightest provable lower
# bound and matches the spec's own prose (§4.0: "+T_serial attaches to the
# Tier-2 term"). Recommendation: Update spec §4.1/§7 formulas to match this.
#
# Five-way attribution decomposes the gap between T_bound and a hypothetical
# zero-overhead kernel.  This is diagnostic output, NOT part of the bound.
#
# Source spec: .omc/specs/performance_bound_model.md §3, §4.2, §A.5

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..model.grid_model import GridBound
from ..model.component_model import ComponentBound, compute_component_floor
from ..model.serialization import SerializationSplit, classify_handoffs
from ..extract.op_classifier import Component
from ..extract.hivm_extractor import HIVMExtract
from ..extract.eligibility_oracle import get_eligibility
from ..calibration.constants import CalibrationDB


class BindingTier(str, Enum):
    """Which tier binds the overall performance."""
    GRID = "grid"
    COMPONENT = "component"


@dataclass
class Attribution:
    """Five-way gap attribution for a single kernel.

    Gaps are expressed as both absolute (microseconds) and as fractions
    of T_bound.  The five gaps are:

    grid:   Realized grid worse than optimal partition (occupancy, load_balance)
    gap1:   Wrong-unit placement — ops running on suboptimal unit
            (eligibility vs realized unit assignment)
    gap2:   Coalescing / transfer efficiency — MTE small-packet amortization,
            alignment waste, unused burst capacity
    gap3:   Avoidable serialization — handoffs that could be eliminated
            by scheduling/ping-pong  (the avoidable complement of T_serial)
    gap4:   Intra-unit execution inefficiency — low SIMD repeat/mask
            utilization within compute ops
    """
    grid_gap_us: float = 0.0
    gap1_wrong_unit_us: float = 0.0
    gap2_coalescing_us: float = 0.0
    gap3_avoidable_serial_us: float = 0.0
    gap4_intra_unit_exec_us: float = 0.0

    grid_gap_frac: float = 0.0
    gap1_frac: float = 0.0
    gap2_frac: float = 0.0
    gap3_frac: float = 0.0
    gap4_frac: float = 0.0

    @property
    def total_gap_us(self) -> float:
        return (self.grid_gap_us + self.gap1_wrong_unit_us +
                self.gap2_coalescing_us + self.gap3_avoidable_serial_us +
                self.gap4_intra_unit_exec_us)

    def dominant_gap(self) -> tuple[str, float]:
        """Return (gap_name, fraction) of the largest gap."""
        gaps = [
            ("grid", self.grid_gap_frac),
            ("gap1_wrong_unit", self.gap1_frac),
            ("gap2_coalescing", self.gap2_frac),
            ("gap3_avoidable_serial", self.gap3_frac),
            ("gap4_intra_unit_exec", self.gap4_frac),
        ]
        return max(gaps, key=lambda x: x[1])


# Map raw DES pipe names to a coarse engine/component label, so the structural
# split can be compared against msprof's measured engine split and the bound's
# binding component.
_PIPE_TO_ENGINE = {
    "PIPE_S": "scalar",
    "PIPE_V": "vector",
    "PIPE_M": "cube",
    "PIPE_MTE1": "mte",
    "PIPE_MTE2_V": "mte",
    "PIPE_MTE2_C": "mte",
    "PIPE_MTE3": "mte",
    "PIPE_FIX": "fixpipe",
    "PIPE_ALL": "sync",
    "PIPE_UNKNOWN": "unknown",
}


@dataclass
class ComponentAttribution:
    """Diagnostic decomposition of author headroom by execution engine (A.8).

    Two independent views of where time goes, NEVER part of the bound:

      * structural — HIVM/DES per-pipe busy (Σ duration·loop_multiplier),
        clock-independent fractions.  This is what the model accounts for.
      * measured   — msprof per-engine time (populated only for MIX_AIC-class
        kernels whose ratio columns exist).  This is what the hardware did.

    When both rank the same engine first and the bound binds a *different*
    component, ``mis_binding`` is set: the bound under-prices the dominant
    engine.  For chunk_kda this is scalar (the bound models scalar at the
    vector rate — see component_model.py:290 / constants.py:252, US-SB-007).
    """
    structural_pipe_cycles: dict[str, int] = field(default_factory=dict)
    structural_pipe_frac: dict[str, float] = field(default_factory=dict)
    dominant_structural_pipe: str = ""
    dominant_structural_engine: str = ""

    measured_engine_us: Optional[dict[str, float]] = None
    measured_scalar_frac: Optional[float] = None
    dominant_measured_engine: Optional[str] = None

    binding_component: Optional[str] = None
    mis_binding: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "structural_pipe_frac": self.structural_pipe_frac,
            "dominant_structural_pipe": self.dominant_structural_pipe,
            "dominant_structural_engine": self.dominant_structural_engine,
            "measured_engine_us": self.measured_engine_us,
            "measured_scalar_frac": self.measured_scalar_frac,
            "dominant_measured_engine": self.dominant_measured_engine,
            "binding_component": self.binding_component,
            "mis_binding": self.mis_binding,
            "note": self.note,
        }


def attribute_by_component(
    extract: HIVMExtract,
    binding_component: Optional[str] = None,
    measured=None,
) -> ComponentAttribution:
    """Decompose author headroom by engine from DES structure + msprof (A.8).

    Pure diagnostic — does not touch the bound.  Combines the HIVM structural
    per-pipe busy split with, optionally, an ``EngineAttribution`` from msprof,
    and flags when the bound binds a component other than the dominant engine.

    Args:
        extract: HIVM extract (uses op.pipe, duration_cycles, loop_multiplier).
        binding_component: The bound's binding component value (e.g. "vector").
        measured: Optional validate.msprof_parser.EngineAttribution.

    Returns:
        ComponentAttribution with structural + (optional) measured views.
    """
    cycles: dict[str, int] = {}
    for op in extract.operations:
        lm = op.loop_multiplier or 1
        cycles[op.pipe] = cycles.get(op.pipe, 0) + op.duration_cycles * lm
    total = sum(cycles.values())
    frac = {p: (c / total if total else 0.0) for p, c in cycles.items()}
    dom_pipe = max(cycles, key=cycles.get) if cycles else ""
    dom_engine = _PIPE_TO_ENGINE.get(dom_pipe, dom_pipe)

    ca = ComponentAttribution(
        structural_pipe_cycles=cycles,
        structural_pipe_frac=frac,
        dominant_structural_pipe=dom_pipe,
        dominant_structural_engine=dom_engine,
        binding_component=binding_component,
    )

    measured_engine = None
    if measured is not None and getattr(measured, "populated", False):
        ca.measured_engine_us = dict(measured.engine_us)
        ca.measured_scalar_frac = measured.scalar_frac
        ca.dominant_measured_engine = measured.dominant_engine
        # Collapse the measured engine label ("scalar (AIV)") to a bare engine.
        measured_engine = measured.dominant_engine.split(" ")[0].lower()

    # Mis-binding: the dominant engine (structural, corroborated by measured if
    # present) differs from what the bound binds.
    consensus_engine = measured_engine or dom_engine
    if binding_component and consensus_engine and consensus_engine != binding_component.lower():
        ca.mis_binding = True
        note = (
            f"Bound binds '{binding_component}' but the dominant engine is "
            f"'{consensus_engine}' (structural {dom_engine} "
            f"{frac.get(dom_pipe, 0.0) * 100:.0f}%"
        )
        if ca.measured_scalar_frac is not None:
            note += f", measured scalar {ca.measured_scalar_frac * 100:.0f}%"
        note += "). "
        if consensus_engine == "scalar":
            note += (
                "Scalar is modeled at the vector rate (component_model.py:290, "
                "constants.py:252) — most author headroom is unmodeled scalar "
                "cost (model headroom); calibrate scalar (US-SB-007)."
            )
        ca.note = note

    return ca


@dataclass
class BoundResult:
    """Final bound output for a single kernel."""
    kernel_name: str
    t_bound_us: float

    # Decomposed
    t_grid_floor_us: float
    t_core_floor_us: float
    t_serial_irreducible_us: float

    binding_tier: BindingTier
    binding_component: Optional[Component] = None

    attribution: Attribution = field(default_factory=Attribution)

    def __repr__(self) -> str:
        return (f"BoundResult({self.kernel_name}: "
                f"T_bound={self.t_bound_us:.2f} us, "
                f"binding={self.binding_tier.value})")


def combine(
    grid: GridBound,
    component: ComponentBound,
    serial: SerializationSplit,
    kernel_name: str = "unknown",
    extract: Optional[HIVMExtract] = None,
    calibration: Optional[dict] = None,
) -> BoundResult:
    """Combine Tier 1 + Tier 2 + serialization into a single conservative bound.

    T_bound = max(T_grid_floor, T_core_floor + T_serial_irreducible)

    NOTE: This deliberately differs from the spec's written formula
    max(T_grid_floor, T_core_floor) + T_serial_irreducible, which is
    unsound (can overstate the bound). See module header for details.

    The binding tier is determined by which floor is higher:
    - Grid binds when occupancy/load_balance constrain more than per-component BW
    - Component binds when a specific HW unit (Cube, MTE, Vector) is the bottleneck

    The five-way attribution is initialized from the component model's
    per-component rates and the serialization split.  Gap 3 comes directly
    from the avoidable serialization sum.  When an HIVM extract is provided,
    Gaps 1, 2, and 4 are estimated from per-op data.

    Args:
        grid: Tier 1 grid floor.
        component: Tier 2 component floor with per-component decomposition.
        serial: Mandatory/avoidable serialization split.
        kernel_name: Label for the result.
        extract: Optional M3 HIVM extract for per-op Gap 1/2/4 computation.
        calibration: Optional dict with keys "cube", "vector", "memory", "core"
                     for rate-based gap quantification.  If omitted, gap
                     estimates use proportional allocation from component times.

    Returns:
        BoundResult with T_bound, binding tier/component, and attribution.
    """
    # Sound composition: serialization is intra-core, attaches to Tier-2 term
    core_plus_serial_us = component.t_core_floor_us + serial.t_serial_irreducible_us
    t_bound_us = max(grid.t_grid_floor_us, core_plus_serial_us)

    # Determine binding tier: grid binds iff grid floor ≥ core+serial floor
    if grid.t_grid_floor_us >= core_plus_serial_us:
        binding_tier = BindingTier.GRID
        binding_component = None  # grid binds, not a specific component
    else:
        binding_tier = BindingTier.COMPONENT
        binding_component = component.binding_component

    # Attribution: initialize from available data
    attribution = Attribution()

    # Grid gap: realized floor minus ideal-grid floor
    # grid_gap_us = T_grid_floor × (1 − occupancy · load_balance)
    # Perfect grid (occ=lb=1) → 0; imperfect → positive gap
    occ = grid.occupancy
    lb = grid.load_balance
    attribution.grid_gap_us = grid.t_grid_floor_us * (1.0 - occ * lb)

    # Gap 3 (avoidable serial): from serialization split (deduped)
    attribution.gap3_avoidable_serial_us = serial.t_serial_avoidable_us

    # Gap 1/2/4 from extract data
    if extract is not None:
        _wire_gaps(attribution, extract, component, calibration)

    # Convert gaps to fractions
    if t_bound_us > 0:
        attribution.grid_gap_frac = attribution.grid_gap_us / t_bound_us
        attribution.gap1_frac = attribution.gap1_wrong_unit_us / t_bound_us
        attribution.gap2_frac = attribution.gap2_coalescing_us / t_bound_us
        attribution.gap3_frac = attribution.gap3_avoidable_serial_us / t_bound_us
        attribution.gap4_frac = attribution.gap4_intra_unit_exec_us / t_bound_us

    return BoundResult(
        kernel_name=kernel_name,
        t_bound_us=t_bound_us,
        t_grid_floor_us=grid.t_grid_floor_us,
        t_core_floor_us=component.t_core_floor_us,
        t_serial_irreducible_us=serial.t_serial_irreducible_us,
        binding_tier=binding_tier,
        binding_component=binding_component,
        attribution=attribution,
    )


def bound_from_extract(
    extract: HIVMExtract,
    calib_db: Optional[CalibrationDB] = None,
    kernel_name: str = "unknown",
    n_cores: int = 20,
    occupancy: float = 1.0,
    load_balance: float = 1.0,
    total_programs: int | None = None,
) -> BoundResult:
    """High-level entry point: compute T_bound from an HIVM extract + calibration.

    Auto-loads the default 910B3 CalibrationDB when calib_db is None.
    Falls back to I_c = 0 (T_core_floor = inf/0) gracefully when no
    calibration file exists.

    Args:
        extract: M3 HIVM extraction result.
        calib_db: Calibration DB with real sustained rates.  Auto-loaded
                  from the package data directory when None.
        kernel_name: Label for the BoundResult.
        n_cores: Number of cores assigned to this kernel.
        occupancy: Grid occupancy fraction (0, 1].
        load_balance: Load balance fraction (0, 1].
        total_programs: Total program instances.  Defaults to 1 (single program).

    Returns:
        BoundResult with T_bound and decomposed floors.
    """
    from ..calibration.calib_loader import load_default_calib_db
    from ..model.bounds import compute_bounds
    from ..extract.dsl_extractor import GridInfo

    if calib_db is None:
        try:
            calib_db = load_default_calib_db()
        except FileNotFoundError:
            calib_db = None

    if calib_db is not None:
        # Route through compute_bounds: picks i_binding and total_work
        # consistently (memory-bound or compute-bound) from the binding component.
        # Wave scaling: pass total_programs and n_cores for correct multi-wave bounds.
        # Default total_programs=1 (single program) preserves pre-A.5 behavior
        # for callers that don't specify a real launch grid.
        _total_programs = total_programs if total_programs is not None else 1
        grid_info = GridInfo(
            grid_dims=(n_cores,),
            total_programs=_total_programs,
            tile_assignment={},
            work={},
            occupancy=occupancy,
            load_balance=load_balance,
            redundancy=1.0,
            busiest_core_id=0,
        )
        pieces = compute_bounds(grid_info, extract, calib_db,
                                n_cores=n_cores, total_programs=_total_programs)
        grid = pieces.grid
        comp = pieces.component
        serial = pieces.serial
    else:
        # No calibration: fall back to defaults (zero rates → inf times)
        from ..calibration.constants import CubeConfig, VectorConfig, MemHierarchy, CoreConfig
        from ..model.component_model import compute_component_floor
        from ..model.grid_model import compute_grid_floor

        core = CoreConfig()
        comp = compute_component_floor(
            extract,
            CubeConfig(), VectorConfig(), MemHierarchy(), core,
        )

        total_bytes = sum(
            float(op.bytes_transferred) * float(op.loop_multiplier)
            for op in extract.operations
        )
        grid_info = GridInfo(
            grid_dims=(n_cores,),
            total_programs=n_cores,
            tile_assignment={},
            work={},
            occupancy=occupancy,
            load_balance=load_balance,
            redundancy=1.0,
            busiest_core_id=0,
        )
        grid = compute_grid_floor(grid_info, core, 1.0, max(total_bytes, 1.0))
        serial = classify_handoffs(
            extract.handoffs,
            mandatory_handoff_cycles=0.0,
            clock_ghz=1.85,
        )

    return combine(grid, comp, serial, kernel_name=kernel_name, extract=extract)


# ── Gap helpers (diagnostic only — not part of the bound) ──────────────────

# Op-name prefixes for eligibility category lookup
_MATMUL_KEYWORDS = ("matmul", "mm", "bmm")
_REDUCTION_KEYWORDS = ("reduce", "sum", "max", "min", "arg")
_COMPARE_KEYWORDS = ("cmp", "compare")


def _op_category(op_name: str) -> str:
    """Map an op name to an eligibility-oracle category."""
    lower = op_name.lower()
    if any(k in lower for k in _MATMUL_KEYWORDS):
        return "matmul"
    if any(k in lower for k in _REDUCTION_KEYWORDS):
        return "reduction"
    if any(k in lower for k in _COMPARE_KEYWORDS):
        return "compare"
    return "elementwise"


def _compute_gap1(
    extract: HIVMExtract,
    component: ComponentBound,
) -> float:
    """Estimate Gap 1: wrong-unit placement cost.

    For each compute op whose realized (assigned) component is NOT in the
    eligible set, estimate its contribution to the component's total time.
    Over-estimation is safe here — gap attribution is diagnostic, not part
    of T_bound, and over-counting serves as a flag to the user.

    Returns:
        Estimated wrong-unit time in microseconds.
    """
    gap1_us = 0.0

    for op in extract.operations:
        # MTE has fixed assignment — no placement choice
        if op.component in (Component.MTE_GM, Component.MTE_L1, Component.MTE_UB):
            continue
        # NOTE: Do NOT skip Scalar ops here.  A Scalar op whose semantic-
        # eligibility set includes Vector/Cube (e.g. the seeded i32-compare
        # forced to Scalar by the compiler) IS the canonical Gap-1
        # mis-placement (spec §3 Gap 1; implementation_and_paper_plan.md:126).
        # The eligibility oracle correctly returns {Scalar} for true
        # Scalar-only ops (i32 compare), so those won't trigger a false Gap 1.

        category = _op_category(op.op_name)
        prec_str = op.precision.value if op.precision else None
        eligible = get_eligibility(category, prec_str)

        if op.component not in eligible:
            # Mis-placed: count its share of the realized component's time
            comp_str = op.component.value
            if comp_str not in component.per_component_us:
                continue
            comp_time = component.per_component_us[comp_str]
            if comp_time <= 0:
                continue

            # Estimate this op's share of the component's total work.
            # Use flops fallback when elements is 0 (C++ JSON for Cube ops).
            if op.elements > 0:
                op_work = float(op.elements)
            elif op.flops > 0:
                op_work = float(op.flops)
            else:
                continue  # no work to attribute — skip
            op_work *= float(op.loop_multiplier)

            total_work = component.total_ops.get(comp_str, 0) or component.total_bytes.get(comp_str, 0)
            if total_work <= 0:
                continue

            op_share = op_work / total_work
            gap1_us += comp_time * op_share

    return gap1_us


def _compute_gap2(
    extract: HIVMExtract,
    calibration: Optional[dict] = None,
) -> float:
    """Estimate Gap 2: coalescing / transfer-efficiency gap.

    For each MTE op, compare the time at its actual transfer size
    (which may incur small-packet amortization) vs ideal large-packet BW.

    Returns:
        Estimated coalescing gap in microseconds.
    """
    if calibration is None:
        return 0.0

    memory = calibration.get("memory")
    if memory is None:
        return 0.0

    gap2_us = 0.0

    for op in extract.operations:
        if op.component not in (Component.MTE_GM, Component.MTE_L1, Component.MTE_UB):
            continue
        if op.bytes_transferred <= 0:
            continue

        # Determine src/dst path
        if op.component == Component.MTE_GM:
            src, dst = "gm", "ub"
        elif op.component == Component.MTE_L1:
            src, dst = "l1", "l0a"
        elif op.component == Component.MTE_UB:
            src, dst = "ub", "gm"
        else:
            continue

        try:
            # Ideal: large-packet BW (pkt_size=-1)
            bw_ideal, _ = memory.lookup_bw(src, dst, pkt_size=-1)
            # Actual: with per-transfer size
            bw_actual, _ = memory.lookup_bw(src, dst, pkt_size=op.bytes_transferred)
        except KeyError:
            continue

        if bw_ideal <= 0 or bw_actual <= 0:
            continue

        total_bytes = float(op.bytes_transferred) * float(op.loop_multiplier)
        t_ideal = total_bytes / bw_ideal
        t_actual = total_bytes / bw_actual
        gap2_us += max(0.0, t_actual - t_ideal)

    return gap2_us


def _compute_gap4(
    extract: HIVMExtract,
    component: ComponentBound,
    calibration: Optional[dict] = None,
) -> float:
    """Estimate Gap 4: intra-unit execution inefficiency (per-instruction).

    Models the **per-instruction issue overhead** of CCE compute instructions
    (the paper's AvgPool ``repeat=1`` → 4.31× case).  Each vector/cube op runs
    ``repeat`` SIMD iterations (256-byte register iterations) split across
    ``ceil(repeat / MAX_REPEAT)`` hardware instructions, and every instruction
    pays a fixed startup latency.  When ``repeat`` is small the startup cost is
    a large fraction of the op's busy time (overhead-dominated); as ``repeat``
    grows the startup amortizes and the inefficiency → 0.

    ``repeat`` is derived analytically by the C++ emitter from each op's
    element count and width (the hivm.hir IR has no per-op repeat/mask — they
    only appear in later CCE codegen; see Task 4b).  ``mask`` is not used by
    this model.

    The per-op inefficiency is::

        overhead_cyc = ceil(repeat / MAX_REPEAT) * startup_cyc[component]
        inefficiency = overhead_cyc / (overhead_cyc + repeat * PER_ITER_CYC)

    bounded in (0, 1), so Gap 4 ≤ Σ op_time ≤ component time ≤ T_measured
    (keeps the bound sound).

    Returns:
        Estimated per-instruction issue overhead in microseconds.
    """
    # CCE constants.  startup cycles mirror calibration.startup_latencies
    # (ascend_910b.json); MAX_REPEAT is the 8-bit CCE repeat field limit;
    # PER_ITER_CYC is the 1-iteration/cycle vector/cube throughput floor.
    MAX_REPEAT = 255
    PER_ITER_CYC = 1.0
    startup_cyc = {Component.VECTOR: 35.0, Component.CUBE: 20.0}
    if calibration:
        startups = calibration.get("startup_latencies") or {}
        if "vector_startup_cycles" in startups:
            startup_cyc[Component.VECTOR] = float(startups["vector_startup_cycles"])
        if "cube_startup_cycles" in startups:
            startup_cyc[Component.CUBE] = float(startups["cube_startup_cycles"])

    # SIMD lanes per 256-byte register iteration.  Matches the 128-wide SIMD
    # convention used elsewhere in the model (a 4-byte/lane assumption).
    LANES_PER_ITER = 128

    gap4_us = 0.0

    for op in extract.operations:
        if op.component not in (Component.CUBE, Component.VECTOR):
            continue

        # Intrinsic SIMD iterations this op must perform.
        if op.elements > 0:
            optimal_iters = -(-op.elements // LANES_PER_ITER)
        else:
            optimal_iters = max(1, op.repeat)
        # op.repeat is the per-instruction iteration count the compiler used
        # (CCE repeat field, ≤ MAX_REPEAT).  Optimal batching packs every
        # iteration into the fewest instructions; a suboptimally-low repeat
        # (the paper's AvgPool repeat=1) issues many instructions, each paying
        # the fixed startup latency.  Gap 4 = the *avoidable* startup overhead.
        per_instr = max(1, min(op.repeat, MAX_REPEAT))
        n_instr = -(-optimal_iters // per_instr)            # ceil
        best_instr = -(-optimal_iters // MAX_REPEAT)        # ceil, maximal batch
        avoidable_instr = max(0, n_instr - best_instr)
        if avoidable_instr == 0:
            continue  # already optimally batched — no intra-unit overhead
        overhead_cyc = avoidable_instr * startup_cyc[op.component]
        busy_cyc = n_instr * startup_cyc[op.component] + optimal_iters * PER_ITER_CYC
        inefficiency = overhead_cyc / busy_cyc  # (0, 1)

        # Estimate this op's time on its component.
        # Use op.flops as fallback when op.elements is 0 (C++ JSON path
        # may emit flops but not elements for Cube ops).
        comp_str = op.component.value
        if op.elements > 0:
            op_work = float(op.elements)
        elif op.flops > 0:
            op_work = float(op.flops)
        else:
            continue  # no work to attribute — skip
        op_work *= float(op.loop_multiplier)

        if comp_str in component.total_ops and component.total_ops[comp_str] > 0:
            total_work = component.total_ops[comp_str]
            if comp_str in component.per_component_us:
                comp_time = component.per_component_us[comp_str]
                op_time = comp_time * (op_work / total_work)
            else:
                continue
        elif comp_str in component.total_bytes and component.total_bytes[comp_str] > 0:
            total_work = component.total_bytes[comp_str]
            if comp_str in component.per_component_us:
                comp_time = component.per_component_us[comp_str]
                op_time = comp_time * (op_work / total_work)
            else:
                continue
        else:
            continue

        # Gap 4 = per-instruction startup overhead share of this op's time
        gap4_us += inefficiency * op_time

    return gap4_us


def _wire_gaps(
    attribution: Attribution,
    extract: HIVMExtract,
    component: ComponentBound,
    calibration: Optional[dict] = None,
) -> None:
    """Populate Gap 1/2/4 into an Attribution from extract data."""
    gap1 = _compute_gap1(extract, component)
    if gap1 > 0:
        attribution.gap1_wrong_unit_us = gap1

    gap2 = _compute_gap2(extract, calibration)
    if gap2 > 0:
        attribution.gap2_coalescing_us = gap2

    gap4 = _compute_gap4(extract, component, calibration)
    if gap4 > 0:
        attribution.gap4_intra_unit_exec_us = gap4
