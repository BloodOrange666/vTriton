"""
M3 — HIVM Extractor / Tier 2 input.

Consumes the C++-emitted structural JSON (preferred path per §4b):
  - HIVMAnalysis::emitDESGraph() JSON (per-op: pipe, bytes, elements, duration, ...)
  - PipelineScheduler::emitDependencyGraphJSON() JSON (per-op: hw_unit, deps, cycles)

Falls back to walking .npuir.mlir via MLIR Python bindings only if C++ emit
path is unavailable.

Extracts per-component:
  - O_prec[component] — op/byte counts per precision/transfer-type
  - transfer_size[mte], transfer_alignment[mte] (Gap 2)
  - realized unit_assignment[op] (Gap 1)
  - repeat/mask/SIMD-lane params per compute op (Gap 4)
  - handoff list with producer/consumer components (serialization split)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .op_classifier import classify_op, Component, Precision


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MTE_COMPONENTS = {"mte_gm", "mte_l1", "mte_ub"}

# Canonical memory-space vocabulary — applied at load time so all
# downstream consumers (handoffs, aggregation, A.4) see normalized names.
_SPACE_NORMALIZE = {
    "global": "gm", "hbm": "gm", "dram": "gm",
    "l0a": "l0a", "l0b": "l0b", "l0c": "l0c",
    "l1": "l1", "cube_l1": "l1",
    "ub": "ub", "uniform_buffer": "ub",
    "gm": "gm",
}

# Required fields in each DES graph operation entry.
_DES_REQUIRED_FIELDS = ("id", "name", "pipe")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OpRecord:
    """Single operation extracted from HIVM dump."""
    op_id: int
    op_name: str
    component: Component
    precision: Optional[Precision]
    pipe: str                  # HIVM pipe name (e.g., "Cube", "Vector", "MTE2")
    bytes_transferred: int = 0
    elements: int = 0
    flops: int = 0
    duration_cycles: int = 0
    loop_multiplier: int = 1
    depends_on: List[int] = field(default_factory=list)
    src_space: str = ""
    dst_space: str = ""

    # Repeat/mask/SIMD-lane params (for Gap 4 intra-unit attribution).
    # Future-populated: HIVMAnalysis does not yet emit repeat/mask in its
    # JSON output.  Until the C++ emitter exposes these, both default to
    # their fully-utilized values (repeat=1, mask=0).
    repeat: int = 1       # repeat count (>1 means the op iterates internally)
    mask: int = 0         # mask lanes disabled (0 = all lanes active)

    # Scheduling (from PipelineScheduler JSON)
    start_cycle: int = 0
    end_cycle: int = 0
    hw_unit: str = ""


@dataclass
class HandoffRecord:
    """A cross-component data handoff (producer → consumer)."""
    producer_op_id: int
    consumer_op_id: int
    producer_component: Component
    consumer_component: Component
    bytes_transferred: int
    is_mandatory: Optional[bool] = None  # classified by serialization.py


@dataclass
class HIVMExtract:
    """Complete Tier 2 extraction from one core's HIVM."""
    operations: List[OpRecord]
    handoffs: List[HandoffRecord]

    # Per-component aggregate O_prec
    o_prec: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # o_prec[component_name][precision_name] = total ops (or bytes for MTE)

    # Per-component total work
    total_flops: Dict[str, int] = field(default_factory=dict)
    total_bytes: Dict[str, int] = field(default_factory=dict)

    # Transfer metadata (Gap 2 input)
    transfer_sizes: Dict[str, List[int]] = field(default_factory=dict)
    # Alignment per MTE transfer: 0 = unknown (C++ emitter does not yet
    # expose address alignment).  When 0, Gap 2 must treat alignment as
    # unknown rather than "aligned".  A non-zero value indicates the
    # transfer's byte offset was confirmed aligned to that boundary.
    transfer_alignments: Dict[str, List[int]] = field(default_factory=dict)

    # Realized unit assignment (Gap 1 input)
    unit_assignment: Dict[int, str] = field(default_factory=dict)

    @property
    def total_cycles(self) -> int:
        if not self.operations:
            return 0
        return max(op.end_cycle for op in self.operations)


# ---------------------------------------------------------------------------
# JSON consumers (primary path — C++ emits, Python consumes)
# ---------------------------------------------------------------------------

def load_hivm_desgraph(path: Path | str) -> List[OpRecord]:
    """Load HIVMAnalysis::emitDESGraph() JSON into OpRecord list.

    Parses the canonical ``operations`` array emitted by the C++
    ``HIVMAnalysisReport::emitDESGraph()`` method.  The legacy ``nodes``
    key is accepted as a fallback for older emitters but is not preferred.

    Raises:
        ValueError: If the JSON contains neither ``operations`` nor ``nodes``,
                    or if any operation is missing required fields.
    """
    with open(path) as f:
        data = json.load(f)

    # Canonical key: "operations" (matches C++ emitDESGraph at HIVMAnalysis.cpp:3134)
    raw_ops = data.get("operations")
    if raw_ops is None:
        # Legacy fallback for older emitters that used "nodes"
        raw_ops = data.get("nodes")
    if raw_ops is None:
        raise ValueError(
            "DES graph JSON must contain an 'operations' array "
            "(or legacy 'nodes'). Found keys: "
            + str(list(data.keys()))
        )

    ops: List[OpRecord] = []
    for idx, node in enumerate(raw_ops):
        # Validate required fields — fail fast on malformed JSON
        missing = [f for f in _DES_REQUIRED_FIELDS if f not in node]
        if missing:
            raise ValueError(
                f"DES graph operation at index {idx} missing required fields: "
                f"{missing}. Got keys: {list(node.keys())}"
            )
        comp, prec = classify_op(
            op_name=node.get("name", ""),
            pipe=node.get("pipe", ""),
            elem_type=node.get("elem_type", ""),
        )
        ops.append(OpRecord(
            op_id=node.get("id", 0),
            op_name=node.get("name", ""),
            component=comp,
            precision=prec,
            pipe=node.get("pipe", ""),
            bytes_transferred=node.get("bytes", 0),
            elements=node.get("elements", 0),
            flops=node.get("flops", 0),  # C++ emitDESGraph emits flops for compute ops
            duration_cycles=node.get("duration", 0),
            loop_multiplier=node.get("loop_multiplier", 1),
            depends_on=node.get("depends_on", []),
            src_space=_SPACE_NORMALIZE.get(node.get("src_space", ""), node.get("src_space", "")),
            dst_space=_SPACE_NORMALIZE.get(node.get("dst_space", ""), node.get("dst_space", "")),
            repeat=node.get("repeat", 1),
            mask=node.get("mask", 0),
            start_cycle=node.get("start_cycle", 0),
            end_cycle=node.get("end_cycle", 0),
        ))
    return ops


def load_pipeline_depgraph(path: Path | str) -> List[OpRecord]:
    """Load PipelineScheduler::emitDependencyGraphJSON() into OpRecord list."""
    with open(path) as f:
        data = json.load(f)

    ops = []
    for node in data.get("operations", []):
        comp, prec = classify_op(
            op_name=node.get("op_name", ""),
            pipe=node.get("hw_unit", ""),
            elem_type="",
        )
        ops.append(OpRecord(
            op_id=node.get("id", 0),
            op_name=node.get("op_name", ""),
            component=comp,
            precision=prec,
            hw_unit=node.get("hw_unit", ""),
            bytes_transferred=node.get("bytes", 0),
            flops=node.get("flops", 0),
            duration_cycles=node.get("duration", 0),
            loop_multiplier=node.get("loop_multiplier", 1),
            depends_on=node.get("depends_on", []),
            start_cycle=node.get("start_cycle", 0),
            end_cycle=node.get("end_cycle", 0),
        ))
    return ops


def _compute_producer_component(
    op_id: int,
    op_by_id: Dict[int, OpRecord],
) -> Optional[Component]:
    """Trace through MTE intermediaries to find the ultimate compute producer.

    For dependency chains like Cube → FixPipe(MTE_UB) → Vector, the immediate
    edge is MTE_UB → Vector, but the canonical handoff for serialization
    analysis is Cube → Vector.  This function walks backward through MTE
    intermediaries to find the originating compute component.

    Returns:
        The compute Component of the ultimate producer, or None if the chain
        does not reach a compute component.
    """
    visited = set()
    current_id = op_id
    while current_id in op_by_id:
        if current_id in visited:
            return None  # cycle guard
        visited.add(current_id)
        op = op_by_id[current_id]
        if op.component.value not in _MTE_COMPONENTS:
            return op.component
        # MTE intermediary: trace its dependencies
        if not op.depends_on:
            return op.component  # MTE with no deps: use it as-is
        current_id = op.depends_on[0]  # follow primary dependency
    return None


def _infer_flops_from_loads(ops: List[OpRecord]) -> None:
    """Infer flops for compute ops where the C++ emitter left flops=0.

    For matmul (Cube pipe): traces input loads to compute 2·M·N·K.
    Given load_A has elements=M·K and load_B has elements=K·N, and
    the matmul output has elements=M·N, then K = √(M·K · K·N / M·N)
    and flops = 2 · M · N · K.

    Modifies ops in place.
    """
    op_by_id = {op.op_id: op for op in ops}

    for op in ops:
        if op.flops > 0:
            continue  # already populated by C++ or depgraph merge
        if op.component != Component.CUBE:
            continue  # only infer for Cube matmul ops

        # Find CubeMTE2 loads feeding this op
        input_loads = []
        for dep_id in op.depends_on:
            dep = op_by_id.get(dep_id)
            if dep and dep.component == Component.MTE_GM and dep.elements > 0:
                input_loads.append(dep)

        if len(input_loads) < 2:
            continue  # can't infer without two input loads

        load_a, load_b = input_loads[0], input_loads[1]
        output_elements = op.elements

        if output_elements <= 0 or load_a.elements <= 0 or load_b.elements <= 0:
            continue

        # K = sqrt(load_A_elems * load_B_elems / output_elems)
        k_squared = load_a.elements * load_b.elements / output_elements
        if k_squared <= 0:
            continue
        k = int(round(k_squared ** 0.5))
        if k <= 0:
            continue

        op.flops = 2 * output_elements * k


def extract_hivm(
    desgraph_path: Path | str,
    depgraph_path: Path | str | None = None,
) -> HIVMExtract:
    """Full Tier 2 extraction from C++-emitted JSON dumps.

    Args:
        desgraph_path: Path to emitDESGraph() JSON (HIVMAnalysis).
        depgraph_path: Optional path to emitDependencyGraphJSON() (PipelineScheduler).

    Returns:
        HIVMExtract with per-component aggregates, handoffs, and metadata.
    """
    ops = load_hivm_desgraph(desgraph_path)

    # Merge scheduling info from dependency graph if available
    if depgraph_path is not None:
        dep_ops = load_pipeline_depgraph(depgraph_path)
        dep_by_id = {op.op_id: op for op in dep_ops}
        for op in ops:
            if op.op_id in dep_by_id:
                dep = dep_by_id[op.op_id]
                op.start_cycle = dep.start_cycle
                op.end_cycle = dep.end_cycle
                op.hw_unit = dep.hw_unit
                op.flops = dep.flops

    # Infer flops for compute ops where C++ emitted flops=0.
    # For matmul: flops = 2 * M * N * K, derived from input load sizes.
    _infer_flops_from_loads(ops)

    # Build handoff list: cross-component edges in depends_on.
    #
    # For each cross-component dependency, we emit two handoffs:
    #   1. The immediate edge (e.g., MTE_UB → Vector) — for byte-level tracing.
    #   2. The canonical compute-to-compute edge (e.g., Cube → Vector) —
    #      traced through MTE intermediaries, used by serialization.py for
    #      mandatory/avoidable classification.
    op_by_id = {op.op_id: op for op in ops}
    handoffs: List[HandoffRecord] = []

    for op in ops:
        for dep_id in op.depends_on:
            if dep_id not in op_by_id:
                continue
            dep = op_by_id[dep_id]

            # Emit immediate cross-component handoff
            if dep.component != op.component:
                handoffs.append(HandoffRecord(
                    producer_op_id=dep_id,
                    consumer_op_id=op.op_id,
                    producer_component=dep.component,
                    consumer_component=op.component,
                    bytes_transferred=max(dep.bytes_transferred, op.bytes_transferred),
                ))

                # Emit canonical compute-to-compute handoff if this is an
                # MTE intermediary between two different compute components.
                # E.g., Cube → FixPipe(MTE_UB) → Vector yields Cube → Vector.
                if (op.component.value not in _MTE_COMPONENTS
                        and dep.component.value in _MTE_COMPONENTS):
                    producer_comp = _compute_producer_component(dep_id, op_by_id)
                    if (producer_comp is not None
                            and producer_comp != op.component):
                        handoffs.append(HandoffRecord(
                            producer_op_id=dep_id,
                            consumer_op_id=op.op_id,
                            producer_component=producer_comp,
                            consumer_component=op.component,
                            bytes_transferred=max(dep.bytes_transferred, op.bytes_transferred),
                        ))

    # Aggregate per-component O_prec and work totals.
    #
    # Compute components (cube, vector, scalar): O_prec counts ops/elements.
    # MTE components (mte_gm, mte_l1, mte_ub): O_prec counts bytes transferred.
    # This distinction is critical — counting MTE as elements corrupts the
    # memory component floor used downstream (A.3 plan AC-2).
    o_prec: Dict[str, Dict[str, float]] = {}
    total_flops: Dict[str, int] = {}
    total_bytes: Dict[str, int] = {}
    transfer_sizes: Dict[str, List[int]] = {}
    transfer_alignments: Dict[str, List[int]] = {}

    for op in ops:
        comp_name = op.component.value
        prec_name = op.precision.value if op.precision else "unknown"

        if comp_name not in o_prec:
            o_prec[comp_name] = {}

        if comp_name in _MTE_COMPONENTS:
            # MTE: aggregate bytes (not elements!)
            work_amount = op.bytes_transferred * op.loop_multiplier
            o_prec[comp_name][prec_name] = (
                o_prec[comp_name].get(prec_name, 0) + work_amount
            )
            # Transfer metadata (Gap 2 input)
            if comp_name not in transfer_sizes:
                transfer_sizes[comp_name] = []
                transfer_alignments[comp_name] = []
            transfer_sizes[comp_name].append(op.bytes_transferred)
            # Alignment: 0 = unknown.  The C++ emitter does not yet expose
            # address/offset alignment.  Gap 2 must treat 0 as "alignment
            # unknown" — never as "confirmed aligned".
            transfer_alignments[comp_name].append(0)
        else:
            # Compute: aggregate flops when available, otherwise elements
            work_amount = op.flops if op.flops > 0 else op.elements
            work_amount *= op.loop_multiplier
            o_prec[comp_name][prec_name] = (
                o_prec[comp_name].get(prec_name, 0) + work_amount
            )

        total_flops[comp_name] = (
            total_flops.get(comp_name, 0) + op.flops * op.loop_multiplier
        )
        total_bytes[comp_name] = (
            total_bytes.get(comp_name, 0) + op.bytes_transferred * op.loop_multiplier
        )

    return HIVMExtract(
        operations=ops,
        handoffs=handoffs,
        o_prec=o_prec,
        total_flops=total_flops,
        total_bytes=total_bytes,
        transfer_sizes=transfer_sizes,
        transfer_alignments=transfer_alignments,
        unit_assignment={op.op_id: op.component.value for op in ops},
    )
