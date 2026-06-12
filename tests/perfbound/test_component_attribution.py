# A.8 — Author-headroom component attribution tests.
#
# Covers the two inputs (msprof per-engine ratios + HIVM/DES structural pipes),
# the mis-binding diagnosis, and the Gap-OVL (exposed control/sync overlap
# deficit) diagnostic.

from __future__ import annotations

from pathlib import Path

import pytest

from perfbound.extract.hivm_extractor import HIVMExtract, OpRecord
from perfbound.extract.op_classifier import Component, Precision
from perfbound.combine.bound_combiner import (
    attribute_by_component, _schedule_overlap,
)
from perfbound.validate.msprof_parser import parse_engine_attribution

FIXTURES = Path(__file__).parent / "fixtures"


def _scalar_heavy_extract() -> HIVMExtract:
    """A tiny extract whose scalar pipe dominates structural busy."""
    ops = [
        OpRecord(op_id=1, op_name="addr_calc", component=Component.SCALAR,
                 precision=None, pipe="PIPE_S",
                 duration_cycles=10, loop_multiplier=10),       # 100 cyc
        OpRecord(op_id=2, op_name="vadd", component=Component.VECTOR,
                 precision=Precision.FP16, pipe="PIPE_V",
                 duration_cycles=5, loop_multiplier=2),          # 10 cyc
        OpRecord(op_id=3, op_name="load", component=Component.MTE_GM,
                 precision=Precision.FP16, pipe="PIPE_MTE2_V",
                 duration_cycles=5, loop_multiplier=1),          # 5 cyc
    ]
    return HIVMExtract(operations=ops, handoffs=[])


# ── msprof measured engine attribution ──────────────────────────────────────

def test_parse_engine_attribution_chunk_kda_is_scalar_bound():
    """The real MIX_AIC chunk_kda row resolves to scalar-dominant engine time."""
    ea = parse_engine_attribution(
        FIXTURES / "chunk_kda_op_summary_910b3.csv", "chunk_kda_bwd"
    )
    assert ea is not None
    assert ea.populated, "MIX_AIC row should carry per-engine ratios"
    assert ea.dominant_engine.startswith("scalar")
    # Scalar is the overwhelming majority of measured execution.
    assert ea.scalar_frac > 0.8
    # Same-core scalar share (scalar(AIV)/aiv_time) is the Gap-OVL term, ~84.5%,
    # and is strictly below the blended wall-clock scalar_frac (~91.6%).
    assert 0.83 < ea.aiv_scalar_frac < 0.86
    assert ea.aiv_scalar_frac < ea.scalar_frac
    # Sanity: cube is occupied-but-idle (high util, ~no MAC) — not the cause.
    assert ea.cube_util_pct > 50.0


def test_parse_engine_attribution_softmax_populated():
    """The AI_VECTOR_CORE softmax row also carries per-engine ratios."""
    ea = parse_engine_attribution(
        FIXTURES / "softmax_op_summary_910b3.csv", "softmax"
    )
    assert ea is not None and ea.populated
    assert 0.0 < ea.scalar_frac < 1.0
    assert ea.dominant_engine != ""


def test_parse_engine_attribution_handles_na_ratios(tmp_path):
    """A row whose ratio columns are N/A yields populated=False, not a crash."""
    csv = tmp_path / "na.csv"
    csv.write_text(
        "Op Name,Task Type,Task Duration(us),aiv_time(us),aicore_time(us),"
        "aiv_scalar_ratio,aiv_vec_ratio\n"
        "mykernel,MIX_AIC,100.0,N/A,N/A,N/A,N/A\n"
    )
    ea = parse_engine_attribution(csv, "mykernel")
    assert ea is not None
    assert not ea.populated
    assert ea.t_measured_us == 100.0


def test_parse_engine_attribution_missing_kernel_returns_none():
    ea = parse_engine_attribution(
        FIXTURES / "chunk_kda_op_summary_910b3.csv", "no_such_kernel_xyz"
    )
    assert ea is None


# ── HIVM structural attribution + mis-binding ────────────────────────────────

def test_structural_attribution_ranks_scalar_first():
    ca = attribute_by_component(_scalar_heavy_extract())
    assert ca.dominant_structural_pipe == "PIPE_S"
    assert ca.dominant_structural_engine == "scalar"
    # 100 / (100+10+5) ≈ 0.87
    assert ca.structural_pipe_frac["PIPE_S"] > 0.8


def test_mis_binding_flagged_when_bound_binds_vector_but_scalar_dominates():
    ca = attribute_by_component(_scalar_heavy_extract(), binding_component="vector")
    assert ca.mis_binding is True
    assert "scalar" in ca.note.lower()
    # Gap-OVL framing: note must mention overlap/sync, NOT the retired US-SB-007.
    assert "overlap" in ca.note.lower() or "sync" in ca.note.lower()
    assert "US-SB-007" not in ca.note


def test_no_mis_binding_when_bound_binds_the_dominant_engine():
    ca = attribute_by_component(_scalar_heavy_extract(), binding_component="scalar")
    assert ca.mis_binding is False
    assert ca.note == ""


def test_attribution_is_diagnostic_only_no_bound_fields():
    """ComponentAttribution must not carry or mutate any bound value."""
    ca = attribute_by_component(_scalar_heavy_extract(), binding_component="vector")
    d = ca.to_dict()
    assert "t_bound_us" not in d
    assert set(d) >= {"structural_pipe_frac", "dominant_structural_engine",
                      "binding_component", "mis_binding", "note",
                      "critical_path_cycles", "model_exposed_control_frac",
                      "gap_ovl_pts", "gap_ovl_us", "n_sync_ops"}


def test_empty_extract_is_safe():
    ca = attribute_by_component(HIVMExtract(operations=[], handoffs=[]),
                                binding_component="vector")
    assert ca.dominant_structural_pipe == ""
    assert ca.mis_binding is False


# ── end-to-end: measured view enriches the report ────────────────────────────

def test_report_end_to_end_surfaces_scalar_mis_binding():
    """report_from_desgraph + msprof CSV → mis-binding + overlap line in report."""
    des = Path(".omc/research/hw_runs/kda_des.json")
    csv = FIXTURES / "chunk_kda_op_summary_910b3.csv"
    if not des.exists():
        pytest.skip("real kda_des.json not present")

    from perfbound.combine.run_report import (
        report_from_desgraph, _merge_engine_attribution,
    )
    report = report_from_desgraph(
        des_json=des, grid_dims=(128, 32),
        kernel_name="chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2",
    )
    _merge_engine_attribution(report, csv, "chunk_kda_bwd")

    ca = report.component_attribution
    assert ca is not None
    assert ca["dominant_structural_pipe"] == "PIPE_S"
    assert ca["measured_engine_us"] is not None
    assert ca["measured_scalar_frac"] > 0.8
    assert ca["mis_binding"] is True
    # Gap-OVL uses the same-core scalar share (~84.5%), NOT the blended 91.6%.
    assert 0.83 < ca["measured_aiv_scalar_frac"] < 0.86
    # +72.6 pts (= 84.5% − 11.9% model-exposed), not +79.8 from the blend.
    assert ca["gap_ovl_pts"] is not None
    assert 0.70 < ca["gap_ovl_pts"] < 0.75
    # The report text must show the overlap line.
    text = report.to_text()
    assert "Author-Headroom Component Attribution" in text
    assert "overlap" in text.lower()
    assert "Gap-OVL" in text


# ── Gap-OVL: schedule overlap walker tests ─────────────────────────────────────

def test_schedule_overlap_on_real_des():
    """Load real kda_des.json; assert model_exposed_control_frac and gap_ovl."""
    des = Path(".omc/research/hw_runs/kda_des.json")
    if not des.exists():
        pytest.skip("real kda_des.json not present")

    from perfbound.extract.hivm_extractor import extract_hivm
    extract = extract_hivm(des)
    ca = attribute_by_component(extract, binding_component="vector",
                                t_bound_dsl_us=46110.0, t_measured_us=104326.0)
    # Model should expose a small fraction of control (5-30%)
    assert ca.model_exposed_control_frac is not None
    assert 0.05 < ca.model_exposed_control_frac < 0.30
    # Sync ops should be numerous (>300 for chunk_kda)
    assert ca.n_sync_ops > 300
    # Control busy fraction on the swept timeline is positive
    assert ca.control_busy_frac > 0.0
    # Critical path should be positive
    assert ca.critical_path_cycles > 0


def test_schedule_overlap_synthetic():
    """Hand-computed overlap: PIPE_S alone, then PIPE_V overlaps PIPE_S."""
    ops = [
        # PIPE_S alone for 100 cycles (exposed control)
        OpRecord(op_id=1, op_name="addr_calc", component=Component.SCALAR,
                 precision=None, pipe="PIPE_S",
                 duration_cycles=100, loop_multiplier=1,
                 start_cycle=0, end_cycle=100),
        # PIPE_V compute for 200 cycles
        OpRecord(op_id=2, op_name="vadd", component=Component.VECTOR,
                 precision=Precision.FP16, pipe="PIPE_V",
                 duration_cycles=200, loop_multiplier=1,
                 start_cycle=100, end_cycle=300),
        # PIPE_S overlapping with PIPE_V for 50 cycles (overlapped control)
        OpRecord(op_id=3, op_name="set_flag", component=Component.SCALAR,
                 precision=None, pipe="PIPE_S",
                 duration_cycles=50, loop_multiplier=1,
                 start_cycle=100, end_cycle=150),
    ]
    result = _schedule_overlap(ops)
    # Critical path = max(end_cycle) = 300
    assert result["critical_path"] == 300
    # Exposed control: [0,100) = 100 cycles (PIPE_S alone)
    assert result["exposed_control_cycles"] == 100
    # Overlapped control: [100,150) = 50 cycles (PIPE_S + PIPE_V)
    assert result["control_overlapped_cycles"] == 50
    # 1 sync op (set_flag)
    assert result["n_sync_ops"] == 1


def test_degenerate_schedule_no_div_zero():
    """Ops with start_cycle == end_cycle == 0 must not cause division by zero."""
    ops = [
        OpRecord(op_id=1, op_name="addr_calc", component=Component.SCALAR,
                 precision=None, pipe="PIPE_S",
                 duration_cycles=10, loop_multiplier=1,
                 start_cycle=0, end_cycle=0),
    ]
    extract = HIVMExtract(operations=ops, handoffs=[])
    ca = attribute_by_component(extract, binding_component="vector")
    # Overlap fields should be None (degenerate schedule)
    assert ca.critical_path_cycles == 0
    assert ca.model_exposed_control_frac is None
    assert ca.gap_ovl_pts is None
    assert ca.gap_ovl_us is None
    # No exception should have been raised


def test_gap_ovl_us_capped_at_headroom():
    """gap_ovl_us must never exceed author_headroom_us."""
    ops = [
        OpRecord(op_id=1, op_name="addr_calc", component=Component.SCALAR,
                 precision=None, pipe="PIPE_S",
                 duration_cycles=100, loop_multiplier=1,
                 start_cycle=0, end_cycle=100),
        OpRecord(op_id=2, op_name="vadd", component=Component.VECTOR,
                 precision=Precision.FP16, pipe="PIPE_V",
                 duration_cycles=50, loop_multiplier=1,
                 start_cycle=0, end_cycle=50),
    ]
    extract = HIVMExtract(operations=ops, handoffs=[])

    # Create a scenario with a small headroom
    t_measured = 200.0
    t_bound = 180.0  # headroom = 20 µs

    # Fake a measured scalar frac that would produce a large raw µs
    class FakeMeasured:
        populated = True
        engine_us = {"scalar (AIV)": 180.0, "vector (AIV)": 20.0}
        scalar_frac = 0.90
        dominant_engine = "scalar (AIV)"

    ca = attribute_by_component(
        extract, binding_component="vector",
        measured=FakeMeasured(),
        t_bound_dsl_us=t_bound, t_measured_us=t_measured,
    )
    assert ca.gap_ovl_us is not None
    author_headroom = t_measured - t_bound  # 20 µs
    assert ca.gap_ovl_us <= author_headroom
