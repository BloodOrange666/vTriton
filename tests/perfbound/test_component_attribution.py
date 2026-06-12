# A.8 — Author-headroom component attribution tests.
#
# Covers the two inputs (msprof per-engine ratios + HIVM/DES structural pipes)
# and the mis-binding diagnosis that flags unmodeled scalar cost (US-SB-007).

from __future__ import annotations

from pathlib import Path

import pytest

from perfbound.extract.hivm_extractor import HIVMExtract, OpRecord
from perfbound.extract.op_classifier import Component, Precision
from perfbound.combine.bound_combiner import attribute_by_component
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
    assert "US-SB-007" in ca.note


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
                      "binding_component", "mis_binding", "note"}


def test_empty_extract_is_safe():
    ca = attribute_by_component(HIVMExtract(operations=[], handoffs=[]),
                                binding_component="vector")
    assert ca.dominant_structural_pipe == ""
    assert ca.mis_binding is False


# ── end-to-end: measured view enriches the report ────────────────────────────

def test_report_end_to_end_surfaces_scalar_mis_binding():
    """report_from_desgraph + msprof CSV → mis-binding note in the report dict."""
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
    # The report text must show the attribution block.
    assert "Author-Headroom Component Attribution" in report.to_text()
