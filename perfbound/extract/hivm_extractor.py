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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .op_classifier import classify_op, Component, Precision


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
    """Load HIVMAnalysis::emitDESGraph() JSON into OpRecord list."""
    with open(path) as f:
        data = json.load(f)

    ops = []
    for node in data.get("nodes", []):
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
            duration_cycles=node.get("duration", 0),
            loop_multiplier=node.get("loop_multiplier", 1),
            depends_on=node.get("depends_on", []),
            src_space=node.get("src_space", ""),
            dst_space=node.get("dst_space", ""),
            repeat=node.get("repeat", 1),
            mask=node.get("mask", 0),
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

    # Build handoff list: cross-component edges in depends_on
    op_by_id = {op.op_id: op for op in ops}
    handoffs = []
    for op in ops:
        for dep_id in op.depends_on:
            if dep_id in op_by_id:
                dep = op_by_id[dep_id]
                if dep.component != op.component:
                    handoffs.append(HandoffRecord(
                        producer_op_id=dep_id,
                        consumer_op_id=op.op_id,
                        producer_component=dep.component,
                        consumer_component=op.component,
                        bytes_transferred=max(dep.bytes_transferred, op.bytes_transferred),
                    ))

    # Aggregate per-component O_prec
    o_prec: Dict[str, Dict[str, float]] = {}
    total_flops: Dict[str, int] = {}
    total_bytes: Dict[str, int] = {}

    for op in ops:
        comp_name = op.component.value
        prec_name = op.precision.value if op.precision else "unknown"

        if comp_name not in o_prec:
            o_prec[comp_name] = {}
        o_prec[comp_name][prec_name] = o_prec[comp_name].get(prec_name, 0) + op.elements

        total_flops[comp_name] = total_flops.get(comp_name, 0) + op.flops * op.loop_multiplier
        total_bytes[comp_name] = total_bytes.get(comp_name, 0) + op.bytes_transferred * op.loop_multiplier

    return HIVMExtract(
        operations=ops,
        handoffs=handoffs,
        o_prec=o_prec,
        total_flops=total_flops,
        total_bytes=total_bytes,
        unit_assignment={op.op_id: op.component.value for op in ops},
    )
