# M5 — Bound Combiner (two-tier max + T_serial_irreducible)
#
# T_bound = max(T_grid_floor, T_core_floor) + T_serial_irreducible
#
# Composition is max (two independent lower bounds on the same wall-clock
# time), with + T_serial_irreducible attaching to the Tier-2 term.
#
# The max reflects the insight that the grid-level and component-level
# floors are independent lower bounds — whichever is higher constrains
# the overall time.  T_serial_irreducible is added because it is NOT
# captured by either tier's ideal overlap assumption.
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
from ..model.component_model import ComponentBound
from ..model.serialization import SerializationSplit
from ..extract.op_classifier import Component
from ..extract.hivm_extractor import HIVMExtract
from ..extract.eligibility_oracle import get_eligibility


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

    T_bound = max(T_grid_floor, T_core_floor) + T_serial_irreducible

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
    # Compute the max of the two independent floors
    max_floor_us = max(grid.t_grid_floor_us, component.t_core_floor_us)

    # T_bound = max(floors) + mandatory serialization
    t_bound_us = max_floor_us + serial.t_serial_irreducible_us

    # Determine binding tier
    if grid.t_grid_floor_us >= component.t_core_floor_us:
        binding_tier = BindingTier.GRID
        binding_component = None  # grid binds, not a specific component
    else:
        binding_tier = BindingTier.COMPONENT
        binding_component = component.binding_component

    # Attribution: initialize from available data
    attribution = Attribution()

    # Gap 3 (avoidable serial): directly from serialization split
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
        # MTE and Scalar have fixed assignment — no placement choice
        if op.component in (Component.MTE_GM, Component.MTE_L1, Component.MTE_UB):
            continue
        if op.component == Component.SCALAR:
            continue  # scalar ops are always on the scalar pipe

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
    """Estimate Gap 4: intra-unit execution inefficiency.

    For compute ops (Cube, Vector), use repeat/mask fields to estimate
    SIMD/fractal utilization.  With default values (repeat=1, mask=0)
    utilization is 100% and Gap 4 = 0.

    Note: repeat/mask are future-populated fields from the C++ emitter.
    Until HIVMAnalysis exposes per-op repeat and mask in its JSON output,
    Gap 4 will always be 0.0 for real HIVM data (conservative default).

    Returns:
        Estimated intra-unit inefficiency in microseconds.
    """
    gap4_us = 0.0

    for op in extract.operations:
        if op.component not in (Component.CUBE, Component.VECTOR):
            continue

        # Compute utilization from repeat/mask
        if op.component == Component.VECTOR:
            # Vector: mask disables lanes (128-wide SIMD)
            active_lanes = max(0, 128 - op.mask)
            utilization = active_lanes / 128.0
        else:
            # Cube: repeat > 1 means internal iterations
            utilization = 1.0 / max(1, op.repeat)

        if utilization >= 1.0:
            continue  # fully utilized — no gap

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

        # Gap 4 = time lost to sub-optimal utilization
        gap4_us += (1.0 - utilization) * op_time

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
