# Tests for M3 — Eligibility Oracle.
#
# Covers:
#   - matmul FP16/BF16/INT8 → Cube
#   - elementwise/reduction → Vector
#   - i32 compare → Scalar fallback (Gap 1 flag)
#   - unknown category → conservative (include more units)
#   - compute_gap1() against realized assignment
#
# Acceptance: A.3 plan AC-5

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from perfbound.extract.eligibility_oracle import get_eligibility, compute_gap1
from perfbound.extract.op_classifier import Component, Precision


# ── Test: Matmul eligibility ──────────────────────────────────────────────


class TestMatmulEligibility:
    """matmul + (FP16|BF16|INT8) → {Cube}."""

    def test_fp16_cube(self):
        eligible = get_eligibility("matmul", "fp16")
        assert Component.CUBE in eligible
        assert Component.VECTOR not in eligible

    def test_bf16_cube(self):
        eligible = get_eligibility("matmul", "bf16")
        assert Component.CUBE in eligible

    def test_int8_cube(self):
        eligible = get_eligibility("matmul", "int8")
        assert Component.CUBE in eligible

    def test_fp32_cube_or_vector(self):
        """FP32 matmul can fall back to Vector — conservative."""
        eligible = get_eligibility("matmul", "fp32")
        assert Component.CUBE in eligible
        assert Component.VECTOR in eligible  # conservative: include fallback


# ── Test: Elementwise eligibility ─────────────────────────────────────────


class TestElementwiseEligibility:
    """elementwise → {Vector} for FP16/BF16/FP32."""

    def test_fp16_vector(self):
        eligible = get_eligibility("elementwise", "fp16")
        assert Component.VECTOR in eligible
        assert Component.CUBE not in eligible

    def test_fp32_vector(self):
        eligible = get_eligibility("elementwise", "fp32")
        assert Component.VECTOR in eligible

    def test_int32_vector_and_scalar(self):
        """int32 elementwise: conservative, includes Vector + Scalar."""
        eligible = get_eligibility("elementwise", "int32")
        assert Component.VECTOR in eligible
        assert Component.SCALAR in eligible


# ── Test: Reduction eligibility ───────────────────────────────────────────


class TestReductionEligibility:
    """reduction → {Vector} for standard precisions."""

    def test_fp16_vector(self):
        eligible = get_eligibility("reduction", "fp16")
        assert Component.VECTOR in eligible

    def test_fp32_vector(self):
        eligible = get_eligibility("reduction", "fp32")
        assert Component.VECTOR in eligible


# ── Test: i32 compare Scalar fallback (AC-5) ─────────────────────────────


class TestCompareScalarFallback:
    """i32 compare → {Scalar} — the seeded Gap 1 case."""

    def test_i32_compare_scalar_only(self):
        """i32 compare eligible set is {Scalar} — narrow, not conservative."""
        eligible = get_eligibility("compare", "int32")
        assert Component.SCALAR in eligible
        assert Component.VECTOR not in eligible

    def test_fp16_compare_vector(self):
        """FP16 compare goes to Vector (normal path)."""
        eligible = get_eligibility("compare", "fp16")
        assert Component.VECTOR in eligible
        assert Component.SCALAR not in eligible

    def test_gap1_flagged_for_scalar_fallback(self):
        """An FP16 compare realized on Scalar should flag Gap 1."""
        # FP16 compare eligible={Vector}, realized=Scalar → Gap 1!
        assert compute_gap1(
            op_id=1, op_category="compare",
            precision="fp16", realized_component=Component.SCALAR,
        ) is True

    def test_gap1_not_flagged_for_correct_placement(self):
        """An i32 compare on Scalar is NOT Gap 1 (correct placement)."""
        # i32 compare eligible={Scalar}, realized=Scalar → no gap
        assert compute_gap1(
            op_id=1, op_category="compare",
            precision="int32", realized_component=Component.SCALAR,
        ) is False

    def test_gap1_not_flagged_for_fp16_vector(self):
        """An FP16 compare on Vector is NOT Gap 1."""
        assert compute_gap1(
            op_id=1, op_category="compare",
            precision="fp16", realized_component=Component.VECTOR,
        ) is False


# ── Test: Unknown category conservative ───────────────────────────────────


class TestUnknownConservative:
    """Unknown categories must be conservative: include more eligible units."""

    def test_unknown_includes_scalar(self):
        """Unknown op at minimum gets Scalar as eligible."""
        eligible = get_eligibility("unknown_op_type")
        assert Component.SCALAR in eligible

    def test_unknown_precision_default(self):
        """Known category with unknown precision falls back to 'default' rule."""
        eligible = get_eligibility("cast")
        # cast has a "default" rule: {Vector}
        assert Component.VECTOR in eligible


# ── Test: Gap 1 against realized HIVM assignment ──────────────────────────


class TestGap1AgainstRealized:
    """compute_gap1() against the HIVM unit_assignment data."""

    def test_add_on_cube_is_gap1(self):
        """Elementwise add realized on Cube: eligible={Vector}, not in {Cube}."""
        assert compute_gap1(
            op_id=1, op_category="elementwise",
            precision="fp16", realized_component=Component.CUBE,
        ) is True

    def test_matmul_on_cube_not_gap1(self):
        """Matmul FP16 on Cube: eligible={Cube}, in set → no gap."""
        assert compute_gap1(
            op_id=1, op_category="matmul",
            precision="fp16", realized_component=Component.CUBE,
        ) is False

    def test_matmul_on_vector_is_gap1(self):
        """Matmul FP16 on Vector: eligible={Cube}, not in {Vector} → Gap 1."""
        assert compute_gap1(
            op_id=1, op_category="matmul",
            precision="fp16", realized_component=Component.VECTOR,
        ) is True


# ── Test: analyze_gap1_from_extract (primary path) ─────────────────────────


class TestAnalyzeGap1FromExtract:
    """analyze_gap1_from_extract() works directly on HIVMExtract."""

    def test_matmul_on_cube_no_gap(self):
        """Matmul op on Cube via extract: no Gap 1."""
        from perfbound.extract.hivm_extractor import HIVMExtract, OpRecord
        from perfbound.extract.semantic_extractor import analyze_gap1_from_extract

        ops = [
            OpRecord(op_id=1, op_name="matmul", component=Component.CUBE,
                     precision=Precision.FP16, pipe="Cube",
                     bytes_transferred=0, elements=1000),
        ]
        extract = HIVMExtract(operations=ops, handoffs=[],
                              unit_assignment={1: "cube"})
        reports = analyze_gap1_from_extract(extract)
        assert len(reports) == 1
        assert not reports[0].is_gap1

    def test_add_on_cube_is_gap(self):
        """Elementwise add op on Cube via extract: Gap 1 flagged."""
        from perfbound.extract.hivm_extractor import HIVMExtract, OpRecord
        from perfbound.extract.semantic_extractor import analyze_gap1_from_extract

        ops = [
            OpRecord(op_id=1, op_name="add", component=Component.CUBE,
                     precision=Precision.FP16, pipe="Cube",
                     bytes_transferred=0, elements=4096),
        ]
        extract = HIVMExtract(operations=ops, handoffs=[],
                              unit_assignment={1: "cube"})
        reports = analyze_gap1_from_extract(extract)
        assert len(reports) == 1
        assert reports[0].is_gap1
        assert reports[0].eligible_components == frozenset({Component.VECTOR})

    def test_unknown_ops_skipped(self):
        """Ops that can't be classified are skipped (conservative)."""
        from perfbound.extract.hivm_extractor import HIVMExtract, OpRecord
        from perfbound.extract.semantic_extractor import analyze_gap1_from_extract

        ops = [
            OpRecord(op_id=1, op_name="data_load", component=Component.MTE_GM,
                     precision=Precision.FP16, pipe="VectorMTE2",
                     bytes_transferred=8192, elements=0),
        ]
        extract = HIVMExtract(operations=ops, handoffs=[],
                              unit_assignment={1: "mte_gm"})
        reports = analyze_gap1_from_extract(extract)
        # "data_load" doesn't match any HIVM op category → unknown → skipped
        assert len(reports) == 0


# ── Test: AC-2 hardware-realistic Gap 1 semantics ──────────────────────────


class TestAC2HardwareRealistic:
    """AC-2: Gap 1 uses hardware-realistic eligibility semantics.

    i32-compare on Scalar is NOT a Gap 1 (Scalar is the only hardware option).
    The seeded Gap 1 case is fp16-compare forced to Scalar (compiler bug).
    """

    def test_i32_compare_on_scalar_not_gap1(self):
        """i32 compare realized on Scalar: eligible={Scalar}, in set → no gap.

        Hardware-realistic: Scalar is genuinely the only unit that can do
        i32 compare. No wrong-unit placement.
        """
        assert compute_gap1(
            op_id=1, op_category="compare",
            precision="int32", realized_component=Component.SCALAR,
        ) is False

    def test_fp16_compare_on_scalar_is_gap1(self):
        """fp16 compare realized on Scalar: eligible={Vector}, not in set → Gap 1.

        This is the actual seeded wrong-unit case: fp16 compare should run
        on Vector (high throughput) but HIVM placed it on Scalar.
        """
        assert compute_gap1(
            op_id=1, op_category="compare",
            precision="fp16", realized_component=Component.SCALAR,
        ) is True

    def test_matmul_unknown_precision_not_gap1(self):
        """matmul with unknown precision on Cube: conservative union includes Cube.

        The eligibility oracle unions all eligible sets for the category
        when precision is unknown, so Cube is included → no false Gap 1.
        """
        eligible = get_eligibility("matmul", precision=None)
        assert Component.CUBE in eligible, \
            f"Unknown-precision matmul eligible set should include Cube, got {eligible}"

    def test_fixpipe_not_classified_as_compute(self):
        """FixPipe ops are MTE, not compute — excluded from Gap 1 analysis."""
        from perfbound.extract.semantic_extractor import _classify_hivm_op
        category = _classify_hivm_op("fixpipe")
        assert category == "unknown", \
            f"FixPipe should be unknown (MTE, not compute), got {category}"

    def test_unknown_category_returns_all_compute(self):
        """Unknown op category: conservative, returns all compute components."""
        eligible = get_eligibility("nonexistent_op_type")
        assert Component.CUBE in eligible
        assert Component.VECTOR in eligible
        assert Component.SCALAR in eligible

    def test_matmul_missing_precision_conservative(self):
        """matmul with no precision info: union of all matmul eligible sets."""
        eligible = get_eligibility("matmul", precision=None)
        # Union of fp16={Cube}, bf16={Cube}, int8={Cube}, fp32={Cube,Vector}
        assert Component.CUBE in eligible
        assert Component.VECTOR in eligible  # from fp32 fallback
