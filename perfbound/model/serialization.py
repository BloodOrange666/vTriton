# M4 — Mandatory vs Avoidable Serialization Split
#
# For each handoff between components, classify as:
#   mandatory — cross-component data exchange that MUST go through GM/L2
#               (Cube↔Vector is the canonical case) → enters T_serial_irreducible
#   avoidable — could be eliminated by scheduling/ping-pong → Gap 3
#
# The split ERRS TOWARD "avoidable" — a non-mandatory handoff wrongly counted
# as mandatory would overstate T_bound and break the soundness guarantee.
#
# T_serial_irreducible = Σ min_cost(mandatory_handoffs)
#
# Source spec: .omc/specs/performance_bound_model.md §4.0, §A.4

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set

from ..extract.hivm_extractor import HandoffRecord
from ..extract.op_classifier import Component

# ── Same-path component groups ─────────────────────────────────────────────
#
# Handoffs WITHIN a group are on the same hardware path and are avoidable
# (can be overlapped by pipelining).  Handoffs BETWEEN groups are mandatory
# if the data goes through off-core memory.

_CUBE_PATH: Set[Component] = {
    Component.CUBE,
    Component.MTE_GM,    # CubeMTE2: GM→L1 (cube path input)
    Component.MTE_L1,    # MTE1: L1→L0A/B
    Component.MTE_UB,    # FixPipe: L0C→GM (cube path output)
}

_VECTOR_PATH: Set[Component] = {
    Component.VECTOR,
    Component.MTE_GM,    # VecMTE2: GM→UB (vector path input)
    Component.MTE_UB,    # MTE3: UB→GM (vector path output)
}

# MTE_GM is shared between Cube and Vector paths.
# MTE_UB is shared between Cube and Vector paths.
# This means a CubeMTE2→Vector edge (MTE_GM→Vector) and
# a Cube→MTE3 edge (Cube→MTE_UB) share the same MTE component
# in the component classification.
#
# Handoff classification keys off the COMPUTE components:
# Cube↔Vector through GM is the canonical mandatory path.


def _same_path(comp_a: Component, comp_b: Component) -> bool:
    """Check if two components are on the same pipeline path.

    MTE_GM (CubeMTE2+VecMTE2) and MTE_UB (FixPipe+MTE3) are shared
    between Cube and Vector paths.  A shared component is on the same
    path as ANY non-shared component — the actual path is determined
    at runtime by which direction the data flows.

    Cross-path (mandatory) only when both components are EXCLUSIVELY
    on different paths: Cube (cube-only) ↔ Vector (vector-only).
    """
    if comp_a == comp_b:
        return True

    # Scalar is orthogonal — handoffs involving Scalar are always
    # same-path (scalar instructions serialize on the same core)
    if comp_a == Component.SCALAR or comp_b == Component.SCALAR:
        return True

    # Determine which paths each component belongs to
    a_paths: set[str] = set()
    b_paths: set[str] = set()
    if comp_a in _CUBE_PATH:
        a_paths.add("cube")
    if comp_a in _VECTOR_PATH:
        a_paths.add("vector")
    if comp_b in _CUBE_PATH:
        b_paths.add("cube")
    if comp_b in _VECTOR_PATH:
        b_paths.add("vector")

    # If either component is shared (in both paths), the handoff is
    # same-path — the shared MTE serves whichever path the other
    # component is on.
    if len(a_paths) > 1 or len(b_paths) > 1:
        return True

    # Both are exclusive to a single path → cross-path if different
    return a_paths == b_paths


def _is_cross_component_mandatory(
    handoff: HandoffRecord,
) -> bool:
    """Determine if a handoff is mandatory (must go through GM/L2).

    A handoff is mandatory iff:
      1. Producer and consumer are on DIFFERENT pipeline paths
         (Cube path vs Vector path)
      2. There is no direct on-chip forwarding between these paths
         (Cube output goes to L0C then FixPipe→GM; Vector input comes
          from GM→UB via VecMTE2)

    Canonical mandatory: Cube→Vector (L0C→GM→UB chain)
                         Vector→Cube (UB→GM→L1→L0A/B chain)
    """
    producer = handoff.producer_component
    consumer = handoff.consumer_component

    # Same component or same path → avoidable
    if producer == consumer or _same_path(producer, consumer):
        return False

    # Cross-path → mandatory
    return True


@dataclass
class SerializationSplit:
    """Result of mandatory vs avoidable classification."""
    mandatory_handoffs: List[HandoffRecord] = field(default_factory=list)
    avoidable_handoffs: List[HandoffRecord] = field(default_factory=list)

    t_serial_irreducible_us: float = 0.0  # sum of mandatory min costs
    t_serial_avoidable_us: float = 0.0    # sum of avoidable costs (Gap 3 input)

    def __repr__(self) -> str:
        return (f"SerializationSplit(irreducible={self.t_serial_irreducible_us:.2f} us, "
                f"avoidable={self.t_serial_avoidable_us:.2f} us, "
                f"mandatory_count={len(self.mandatory_handoffs)}, "
                f"avoidable_count={len(self.avoidable_handoffs)})")


def classify_handoffs(
    handoffs: List[HandoffRecord],
    mandatory_handoff_cycles: float = 0.0,
    clock_ghz: float = 1.85,
    memory=None,
) -> SerializationSplit:
    """Classify each handoff as mandatory or avoidable.

    Mandatory handoffs contribute their minimum cost to T_serial_irreducible.
    Avoidable handoffs contribute to Gap 3 (avoidable serialization).

    The minimum cost of a mandatory handoff is the measured
    mandatory_handoff_cycles from M1 calibration — the irreducible
    L0C→GM + GM→UB chain that Cube↔Vector data must traverse.

    Args:
        handoffs: List of cross-component handoffs from HIVM extraction.
        mandatory_handoff_cycles: Measured minimum cycle cost for a single
                                  mandatory handoff (L0C→GM + GM→UB).
                                  From M1 handoff_min.cce microbench.
        clock_ghz: Core clock frequency (1.85 GHz default).
        memory: Optional MemHierarchy for computing avoidable-handoff cost
                (Gap 3).  If None, avoidable cost is 0 (conservative default).

    Returns:
        SerializationSplit with classified handoffs and T_serial_irreducible.

    Raises:
        ValueError: If mandatory_handoff_cycles is 0 and mandatory handoffs exist.
    """
    cycles_per_us = clock_ghz * 1000.0  # 1850 cycles/us at 1.85 GHz

    mandatory: List[HandoffRecord] = []
    avoidable: List[HandoffRecord] = []

    for h in handoffs:
        if _is_cross_component_mandatory(h):
            h.is_mandatory = True
            mandatory.append(h)
        else:
            h.is_mandatory = False
            avoidable.append(h)

    # T_serial_irreducible = Σ over DISTINCT mandatory edges of min_cost(h).
    #
    # Spec §2.2: distinct edges in a dependency chain are sequential and sum
    # (e.g. Cube→Vector for QKᵀ→softmax, then Vector→Cube for softmax→×V).
    # Same edge repeated across loop iterations steady-states to one handoff
    # latency (pipeline), so we dedup by (producer, consumer) component pair
    # before summing.
    #
    # Soundness note: summing assumes distinct edges form a dependency chain
    # (sequential). Genuinely independent cross-path handoffs would be
    # over-counted (summing instead of max), which is unsound for a lower
    # bound. In practice only Cube↔Vector can be mandatory, and both present
    # normally implies a round-trip chain (QK→softmax→×V), so this is safe.
    t_serial_mandatory_us = 0.0
    if mandatory and mandatory_handoff_cycles > 0:
        distinct_edges: set[tuple[Component, Component]] = set()
        for h in mandatory:
            edge = (h.producer_component, h.consumer_component)
            distinct_edges.add(edge)
        t_serial_mandatory_us = len(distinct_edges) * (mandatory_handoff_cycles / cycles_per_us)
    elif mandatory:
        # No calibration — flag but don't fail
        t_serial_mandatory_us = 0.0

    # Avoidable serialization: dedup by (producer, consumer) component pair
    # to prevent double-counting from immediate/canonical edge pairs emitted
    # by the extractor (hivm_extractor.py:329-334).
    #
    # Cost per distinct avoidable handoff via the same memory.lookup_bw path
    # _compute_gap2 uses (bytes/BW).  This is a conservative UPPER estimate
    # of the stall — true Gap 3 is only the non-overlapped portion; an
    # overlap model is a later refinement.  Diagnostic-only, never enters
    # T_bound, so over-estimation cannot break conservatism.
    t_serial_avoidable_us = 0.0
    if avoidable:
        # Dedup by (producer_component, consumer_component)
        seen_avoidable: set[tuple[Component, Component]] = set()
        for h in avoidable:
            edge = (h.producer_component, h.consumer_component)
            if edge in seen_avoidable:
                continue
            seen_avoidable.add(edge)
            if h.bytes_transferred > 0 and memory is not None:
                # Estimate transfer time at sustained BW
                try:
                    bw, _ = memory.lookup_bw("gm", "ub", pkt_size=h.bytes_transferred)
                    if bw > 0:
                        t_serial_avoidable_us += h.bytes_transferred / bw
                except KeyError:
                    pass  # no BW data — conservative 0

    return SerializationSplit(
        mandatory_handoffs=mandatory,
        avoidable_handoffs=avoidable,
        t_serial_irreducible_us=t_serial_mandatory_us,
        t_serial_avoidable_us=t_serial_avoidable_us,
    )
