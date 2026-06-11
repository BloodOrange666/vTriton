# Tests for Stage-B gap closure: Task 4a (repeat/mask extraction) and Task 4.5 (schedule_truncated).
#
# US-SB-001: Value-asserting tests for repeat/mask extraction through the full pipeline.
# US-SB-002: Schedule truncation guard — refuses bounds from incomplete DES schedules.

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from perfbound.extract.hivm_extractor import (
    OpRecord,
    HIVMExtract,
    load_hivm_desgraph,
    extract_hivm,
)
from perfbound.extract.op_classifier import Component
from perfbound.model.component_model import ComponentBound
from perfbound.combine.bound_combiner import _compute_gap4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_des_json(
    ops: list[dict] | None = None,
    schedule_truncated: bool = False,
) -> dict:
    """Build a minimal DES graph JSON dict."""
    if ops is None:
        ops = [
            {"id": 0, "name": "load", "pipe": "VectorMTE2", "duration": 100,
             "bytes": 4096, "elements": 1024, "flops": 0, "repeat": 1, "mask": 0,
             "depends_on": [], "start_cycle": 0, "end_cycle": 100},
            {"id": 1, "name": "vadd", "pipe": "Vector", "duration": 200,
             "bytes": 0, "elements": 1024, "flops": 1024, "repeat": 1, "mask": 0,
             "depends_on": [0], "start_cycle": 100, "end_cycle": 300},
            {"id": 2, "name": "store", "pipe": "MTE3", "duration": 100,
             "bytes": 4096, "elements": 1024, "flops": 0, "repeat": 1, "mask": 0,
             "depends_on": [1], "start_cycle": 300, "end_cycle": 400},
        ]
    return {
        "schema_version": "a3_hivm_des_v1",
        "schedule_truncated": schedule_truncated,
        "clock_ghz": 1.0,
        "operations": ops,
    }


def _write_des_json(data: dict, tmp_path: Path, name: str = "test_des.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def _make_extract_with_repeat_mask(
    repeat: int = 1, mask: int = 0,
    component: Component = Component.VECTOR,
    op_name: str = "vadd",
    elements: int = 1024,
) -> HIVMExtract:
    """Build a minimal HIVMExtract with one compute op having given repeat/mask."""
    ops = [
        OpRecord(
            op_id=1, op_name=op_name, component=component,
            precision=None, pipe="Vector",
            elements=elements, flops=elements,
            repeat=repeat, mask=mask,
        ),
    ]
    comp_name = component.value
    return HIVMExtract(
        operations=ops,
        handoffs=[],
        o_prec={comp_name: {"fp32": float(elements)}},
        total_flops={comp_name: elements},
        total_bytes={},
        transfer_sizes={},
        transfer_alignments={},
        unit_assignment={1: comp_name},
    )


def _make_component_bound(
    comp_name: str = "vector",
    total_ops: float = 1024.0,
    per_component_us: float = 10.0,
) -> ComponentBound:
    """Build a minimal ComponentBound for Gap 4 testing."""
    return ComponentBound(
        t_core_floor_us=per_component_us,
        binding_component=Component.VECTOR,
        total_ops={comp_name: total_ops},
        total_bytes={},
        per_component_us={comp_name: per_component_us},
    )


# ===========================================================================
# US-SB-001: Repeat / mask value-asserting tests
# ===========================================================================

class TestRepeatMaskExtraction:
    """Tests that repeat/mask values flow correctly through the pipeline.

    The C++ emitter writes repeat/mask into the DES JSON.  These tests
    verify the Python side reads them correctly and computes Gap 4 from them.
    """

    def test_default_repeat_mask(self, tmp_path):
        """Ops with default repeat=1, mask=0 load correctly."""
        data = _make_des_json()
        p = _write_des_json(data, tmp_path)
        ops = load_hivm_desgraph(p)
        assert len(ops) == 3
        for op in ops:
            assert op.repeat == 1
            assert op.mask == 0

    def test_non_default_repeat_mask(self, tmp_path):
        """Ops with repeat=8, mask=64 are loaded with those values."""
        ops_data = [
            {"id": 0, "name": "load", "pipe": "VectorMTE2", "duration": 100,
             "bytes": 4096, "elements": 1024, "flops": 0, "repeat": 1, "mask": 0,
             "depends_on": [], "start_cycle": 0, "end_cycle": 100},
            {"id": 1, "name": "vadd", "pipe": "Vector", "duration": 200,
             "bytes": 0, "elements": 1024, "flops": 1024,
             "repeat": 8, "mask": 64,  # non-default values
             "depends_on": [0], "start_cycle": 100, "end_cycle": 300},
            {"id": 2, "name": "store", "pipe": "MTE3", "duration": 100,
             "bytes": 4096, "elements": 1024, "flops": 0, "repeat": 1, "mask": 0,
             "depends_on": [1], "start_cycle": 300, "end_cycle": 400},
        ]
        data = _make_des_json(ops=ops_data)
        p = _write_des_json(data, tmp_path)
        ops = load_hivm_desgraph(p)

        vadd_op = [o for o in ops if o.op_name == "vadd"][0]
        assert vadd_op.repeat == 8, f"Expected repeat=8, got {vadd_op.repeat}"
        assert vadd_op.mask == 64, f"Expected mask=64, got {vadd_op.mask}"

    def test_non_default_repeat_flows_to_extract(self, tmp_path):
        """Non-default repeat/mask values survive through extract_hivm()."""
        ops_data = [
            {"id": 0, "name": "load", "pipe": "VectorMTE2", "duration": 100,
             "bytes": 4096, "elements": 1024, "flops": 0, "repeat": 1, "mask": 0,
             "depends_on": []},
            {"id": 1, "name": "vadd", "pipe": "Vector", "duration": 200,
             "bytes": 0, "elements": 1024, "flops": 1024,
             "repeat": 8, "mask": 64,
             "depends_on": [0]},
            {"id": 2, "name": "store", "pipe": "MTE3", "duration": 100,
             "bytes": 4096, "elements": 1024, "flops": 0, "repeat": 1, "mask": 0,
             "depends_on": [1]},
        ]
        data = _make_des_json(ops=ops_data)
        p = _write_des_json(data, tmp_path)
        extract = extract_hivm(p)

        vadd_ops = [o for o in extract.operations if o.op_name == "vadd"]
        assert len(vadd_ops) == 1
        assert vadd_ops[0].repeat == 8
        assert vadd_ops[0].mask == 64

    def test_gap4_low_repeat_is_overhead_dominated(self):
        """Per-instruction model: a suboptimally-low repeat is overhead-heavy.

        This is the paper's AvgPool case.  The op needs 8 SIMD iterations
        (1024 elems / 128 lanes) but the compiler emitted repeat=1 → it issues
        8 instructions instead of 1, each paying the 35-cycle startup, so most
        of its time is avoidable Gap-4 issue overhead (~0.85).
        """
        extract = _make_extract_with_repeat_mask(repeat=1, mask=0, component=Component.VECTOR)
        comp = _make_component_bound("vector", total_ops=1024.0, per_component_us=10.0)
        gap4 = _compute_gap4(extract, comp)
        # 7 avoidable instrs × 35 / (8×35 + 8) ≈ 0.85 of the 10us op_time
        assert 8.0 < gap4 < 9.0, (
            f"repeat=1 (8 optimal iters) should be ~0.85 overhead; got {gap4}"
        )

    def test_gap4_zero_when_optimally_batched(self):
        """Gap 4 is 0 when repeat already packs all iterations (no waste).

        With elements=1024 (16 SIMD iters), repeat≥16 fits every iteration into
        a single instruction — optimally batched, so there is no avoidable
        per-instruction overhead and Gap 4 = 0.
        """
        comp = _make_component_bound("vector", total_ops=1024.0, per_component_us=10.0)
        gap4_opt = _compute_gap4(
            _make_extract_with_repeat_mask(repeat=16, component=Component.VECTOR), comp)
        assert gap4_opt == 0.0, f"optimal repeat should give Gap-4=0; got {gap4_opt}"
        # And Gap-4 shrinks monotonically as repeat rises toward optimal.
        gap4_lo = _compute_gap4(
            _make_extract_with_repeat_mask(repeat=1, component=Component.VECTOR), comp)
        gap4_mid = _compute_gap4(
            _make_extract_with_repeat_mask(repeat=4, component=Component.VECTOR), comp)
        assert gap4_lo > gap4_mid > gap4_opt, (
            f"Gap-4 should fall with repeat: lo={gap4_lo}, mid={gap4_mid}, opt={gap4_opt}"
        )

    def test_cube_low_repeat_overhead(self):
        """Cube ops also carry avoidable per-instruction overhead (cube=20 cyc)."""
        extract = _make_extract_with_repeat_mask(repeat=4, mask=0, component=Component.CUBE, op_name="mmadL1")
        comp = ComponentBound(
            t_core_floor_us=10.0,
            binding_component=Component.CUBE,
            total_ops={"cube": 1024.0},
            total_bytes={},
            per_component_us={"cube": 10.0},
        )
        gap4 = _compute_gap4(extract, comp)
        # 8 iters / repeat 4 = 2 instrs, 1 avoidable: 1×20 / (2×20+8) ≈ 0.417
        assert 3.5 < gap4 < 5.0, (
            f"Cube repeat=4 (8 optimal iters) overhead should be ~4.17us; got {gap4}"
        )


# ===========================================================================
# US-SB-002: Schedule truncation guard
# ===========================================================================

class TestScheduleTruncationGuard:
    """Tests that schedule_truncated flag is checked and bounds are refused."""

    def test_load_raises_on_truncated_schedule(self, tmp_path):
        """load_hivm_desgraph() raises ValueError when schedule_truncated=True."""
        data = _make_des_json(schedule_truncated=True)
        p = _write_des_json(data, tmp_path)
        with pytest.raises(ValueError, match="truncated"):
            load_hivm_desgraph(p)

    def test_load_succeeds_on_normal_schedule(self, tmp_path):
        """load_hivm_desgraph() succeeds when schedule_truncated=False."""
        data = _make_des_json(schedule_truncated=False)
        p = _write_des_json(data, tmp_path)
        ops = load_hivm_desgraph(p)
        assert len(ops) == 3

    def test_load_succeeds_when_flag_missing(self, tmp_path):
        """load_hivm_desgraph() succeeds when schedule_truncated key is absent (legacy)."""
        data = _make_des_json()
        del data["schedule_truncated"]
        p = _write_des_json(data, tmp_path)
        ops = load_hivm_desgraph(p)
        assert len(ops) == 3

    def test_extract_hivm_raises_on_truncated(self, tmp_path):
        """extract_hivm() raises ValueError when schedule_truncated=True."""
        data = _make_des_json(schedule_truncated=True)
        p = _write_des_json(data, tmp_path)
        with pytest.raises(ValueError, match="truncated"):
            extract_hivm(p)
