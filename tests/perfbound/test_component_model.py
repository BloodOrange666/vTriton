# Golden-number tests for M4 component model.
#
# Hand-computed kernel with known work totals and sustained rates.
# Every intermediate (I_c, T_core_floor, T_serial_irreducible) must
# match a spreadsheet to 3 significant figures.
#
# Source: .omc/specs/performance_bound_model.md §A.4 acceptance

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from perfbound.extract.hivm_extractor import HIVMExtract, OpRecord, HandoffRecord
from perfbound.extract.op_classifier import Component, Precision
from perfbound.calibration.constants import (
    CubeConfig, VectorConfig, MemHierarchy, MemBandwidth, CoreConfig, DType,
)
from perfbound.model.component_model import (
    compute_component_floor, ComponentBound, ComponentRate,
)
from perfbound.model.serialization import classify_handoffs, SerializationSplit


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def calibration() -> dict:
    """Sustained calibration for 910B3 (measured rates, not peaks)."""
    cube = CubeConfig(
        throughput={
            DType.FP16: 280.0,    # TFLOPS sustained
            DType.BF16: 280.0,
            DType.FP32: 140.0,
            DType.INT8: 560.0,
        },
        fractal_sizes={
            DType.FP16: (16, 16, 16),
            DType.FP32: (16, 8, 16),
        },
    )
    vector = VectorConfig(
        throughput_fp16_tflops=18.0,
        throughput_fp32_tflops=9.0,
    )
    memory = MemHierarchy(bw={
        ("gm", "ub", -1): MemBandwidth("gm", "ub", bw_gb_per_s=180.0),
        ("ub", "gm", -1): MemBandwidth("ub", "gm", bw_gb_per_s=180.0),
        ("gm", "l1", -1): MemBandwidth("gm", "l1", bw_gb_per_s=180.0),
        ("l1", "l0a", -1): MemBandwidth("l1", "l0a", bw_gb_per_s=350.0),
        ("l0c", "gm", -1): MemBandwidth("l0c", "gm", bw_gb_per_s=180.0),
    })
    core = CoreConfig(aic_core_num=20, aiv_core_num=40, clock_freq_ghz=1.85)
    return {"cube": cube, "vector": vector, "memory": memory, "core": core}


@pytest.fixture
def matmul_extract() -> HIVMExtract:
    """Minimal matmul kernel: Cube + MTE only, no vector."""
    ops = [
        OpRecord(op_id=1, op_name="matmul", component=Component.CUBE,
                 precision=Precision.FP16, pipe="Cube",
                 bytes_transferred=0, elements=2 * 128 * 64 * 32,
                 duration_cycles=100, loop_multiplier=32, depends_on=[]),
        OpRecord(op_id=2, op_name="cube_load", component=Component.MTE_GM,
                 precision=Precision.FP16, pipe="CubeMTE2",
                 bytes_transferred=128 * 32 * 2 + 32 * 64 * 2,
                 elements=0, duration_cycles=50, loop_multiplier=32,
                 depends_on=[]),
        OpRecord(op_id=3, op_name="cube_store", component=Component.MTE_UB,
                 precision=Precision.FP16, pipe="FixPipe",
                 bytes_transferred=128 * 64 * 2, elements=0,
                 duration_cycles=50, loop_multiplier=1, depends_on=[1]),
    ]
    return HIVMExtract(operations=ops, handoffs=[], unit_assignment={
        op.op_id: op.component.value for op in ops
    })


@pytest.fixture
def matmul_vector_extract() -> HIVMExtract:
    """MatMul + Vector bias + store: Cube→Vector cross-path."""
    ops = [
        OpRecord(op_id=1, op_name="matmul", component=Component.CUBE,
                 precision=Precision.FP16, pipe="Cube",
                 bytes_transferred=0, elements=2 * 128 * 64 * 32,
                 duration_cycles=100, loop_multiplier=32, depends_on=[]),
        OpRecord(op_id=2, op_name="cube_load", component=Component.MTE_GM,
                 precision=Precision.FP16, pipe="CubeMTE2",
                 bytes_transferred=128 * 32 * 2 + 32 * 64 * 2,
                 elements=0, duration_cycles=50, loop_multiplier=32,
                 depends_on=[]),
        OpRecord(op_id=3, op_name="add_bias", component=Component.VECTOR,
                 precision=Precision.FP16, pipe="Vector",
                 bytes_transferred=0, elements=128 * 64,
                 duration_cycles=10, loop_multiplier=1, depends_on=[1]),
        OpRecord(op_id=4, op_name="store", component=Component.MTE_UB,
                 precision=Precision.FP16, pipe="MTE3",
                 bytes_transferred=128 * 64 * 2, elements=0,
                 duration_cycles=50, loop_multiplier=1, depends_on=[3]),
    ]
    return HIVMExtract(operations=ops, handoffs=[
        HandoffRecord(1, 3, Component.CUBE, Component.VECTOR, 128 * 64 * 2,
                      is_mandatory=None),
        HandoffRecord(2, 1, Component.MTE_GM, Component.CUBE, 128 * 32 * 2,
                      is_mandatory=None),
        HandoffRecord(3, 4, Component.VECTOR, Component.MTE_UB, 128 * 64 * 2,
                      is_mandatory=None),
    ], unit_assignment={op.op_id: op.component.value for op in ops})


# ── Harmonic Mean Tests ────────────────────────────────────────────────────

class TestHarmonicMean:
    """I_c = Σ O_prec / Σ (O_prec / P_prec) — the weighted-harmonic mean.

    With a single precision, I_c reduces to the sustained rate for that
    precision (the harmonic mean of one value is that value).
    """

    def test_single_precision_harmonic_reduces_to_rate(self, calibration,
                                                        matmul_extract):
        """With only FP16 work, I_cube = P_cube[fp16]."""
        result = compute_component_floor(
            matmul_extract,
            calibration["cube"],
            calibration["vector"],
            calibration["memory"],
            calibration["core"],
        )
        # Cube: 2*128*64*32*32 = 16,777,216 ops at 280 TFLOPS = 280e6 FLOP/us
        # T_cube = 16,777,216 / 280,000,000 = 0.0599 us → ~0.060 us
        assert "cube" in result.per_component_us
        assert abs(result.per_component_us["cube"] - 0.060) < 0.002, \
            f"Cube time {result.per_component_us['cube']:.4f} not ~0.060 us"

    def test_mte_bytes_correct(self, calibration, matmul_extract):
        """MTE bytes: (128*32*2 + 32*64*2)*32 + 128*64*2*1 = 409,600 bytes.

        At 180 GB/s = 180,000 B/us: T_mte_gm = 393,216/180,000 = 2.185 us
        """
        result = compute_component_floor(
            matmul_extract,
            calibration["cube"],
            calibration["vector"],
            calibration["memory"],
            calibration["core"],
        )
        # Total MTE_GM bytes: (8192+4096)*32 = 393,216
        # T = 393216 / 180000 = 2.185... us
        assert "mte_gm" in result.per_component_us
        assert abs(result.per_component_us["mte_gm"] - 2.185) < 0.01, \
            f"MTE_GM time {result.per_component_us['mte_gm']:.3f} not ~2.185 us"

    def test_binding_component_is_mte(self, calibration, matmul_extract):
        """MTE_GM should bind — it has the largest per-component time."""
        result = compute_component_floor(
            matmul_extract,
            calibration["cube"],
            calibration["vector"],
            calibration["memory"],
            calibration["core"],
        )
        assert result.binding_component == Component.MTE_GM, \
            f"Expected MTE_GM, got {result.binding_component.value}"


# ── Serialization Tests ────────────────────────────────────────────────────

class TestSerialization:
    """Mandatory vs avoidable handoff classification.

    Cube↔Vector through GM is the canonical mandatory handoff.
    Same-path handoffs (Cube→MTE_GM, Vector→MTE_UB) are avoidable.
    """

    def test_cube_to_vector_is_mandatory(self, matmul_vector_extract):
        """Cube→Vector is cross-path → mandatory."""
        serial = classify_handoffs(
            matmul_vector_extract.handoffs,
            mandatory_handoff_cycles=2000,
            clock_ghz=1.85,
        )
        mandatory_count = len(serial.mandatory_handoffs)
        assert mandatory_count == 1, \
            f"Expected 1 mandatory handoff, got {mandatory_count}"
        h = serial.mandatory_handoffs[0]
        assert h.producer_component == Component.CUBE
        assert h.consumer_component == Component.VECTOR

    def test_same_path_handoffs_are_avoidable(self, matmul_vector_extract):
        """MTE_GM→Cube and Vector→MTE_UB are same-path → avoidable."""
        serial = classify_handoffs(
            matmul_vector_extract.handoffs,
            mandatory_handoff_cycles=2000,
            clock_ghz=1.85,
        )
        assert len(serial.avoidable_handoffs) == 2

    def test_serial_irreducible_from_calibration(self, matmul_vector_extract):
        """T_serial_irreducible = mandatory_handoff_cycles / (clock * 1000).

        With 2000 cycles at 1.85 GHz: 2000/1850 = 1.081... us
        """
        serial = classify_handoffs(
            matmul_vector_extract.handoffs,
            mandatory_handoff_cycles=2000,
            clock_ghz=1.85,
        )
        expected = 2000.0 / 1850.0  # 1.081 us
        assert abs(serial.t_serial_irreducible_us - expected) < 0.005, \
            f"T_serial_irreducible {serial.t_serial_irreducible_us:.3f} != {expected:.3f}"

    def test_no_mandatory_without_calibration(self, matmul_vector_extract):
        """Without mandatory_handoff_cycles, T_serial = 0 (conservative)."""
        serial = classify_handoffs(
            matmul_vector_extract.handoffs,
            mandatory_handoff_cycles=0.0,
            clock_ghz=1.85,
        )
        assert serial.t_serial_irreducible_us == 0.0
        # Still counts handoffs correctly
        assert len(serial.mandatory_handoffs) == 1


# ── Mixed-Precision Harmonic Mean ──────────────────────────────────────────

class TestMixedPrecision:
    """Harmonic mean correctly weights precisions by work share."""

    def test_mixed_fp16_int8(self, calibration):
        """Cube with 50% FP16 and 50% INT8 work.

        FP16: 280 TFLOPS, INT8: 560 TFLOPS (but INT8 ops are 2x in elements)
        I = (840 * 10^6) / (280*10^6/280e6 + 560*10^6/560e6) = 840e6 / (1 + 1)
          = 420e6 FLOP/us → harmonic, not arithmetic (420 vs 420 → same here)

        Actually: FP16 work 50% 280 TFLOPS → 280e6 ops/us
                 INT8 work 50% 560 TFLOPS → 560e6 ops/us
                 I_harm = 2 / (1/280 + 1/560) = 2 / (0.003571 + 0.001786)
                        = 2 / 0.005357 = 373.3e6 ops/us
        The harmonic is 373 vs arithmetic 420 — harmonic penalizes the slow dtype.
        """
        ops = [
            OpRecord(op_id=1, op_name="matmul_fp16", component=Component.CUBE,
                     precision=Precision.FP16, pipe="Cube",
                     bytes_transferred=0, elements=1000000,
                     duration_cycles=100, loop_multiplier=1, depends_on=[]),
            OpRecord(op_id=2, op_name="matmul_int8", component=Component.CUBE,
                     precision=Precision.INT8, pipe="Cube",
                     bytes_transferred=0, elements=1000000,
                     duration_cycles=100, loop_multiplier=1, depends_on=[]),
        ]
        extract = HIVMExtract(operations=ops, handoffs=[],
                              unit_assignment={op.op_id: op.component.value for op in ops})

        cube = CubeConfig(
            throughput={DType.FP16: 280.0, DType.INT8: 560.0},
            fractal_sizes={},
        )
        vector = VectorConfig(throughput_fp16_tflops=18.0)
        memory = MemHierarchy()
        core = CoreConfig()

        result = compute_component_floor(extract, cube, vector, memory, core)

        # Expected I_cube (harm): 373.3e6 FLOP/us for 2e6 total ops
        # T_cube = 2,000,000 / 373,333,333 = 0.00536 us
        expected_i = 2.0 / (1.0 / 280.0 + 1.0 / 560.0) * 1e6  # 373.3e6
        expected_t = 2_000_000 / expected_i  # 0.00536
        assert abs(result.per_component_us["cube"] - expected_t) < 0.0001, \
            f"Mixed-precision T_cube {result.per_component_us['cube']:.6f} != {expected_t:.6f}"


# ── Edge Cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Empty extract, all-scalar, zero work."""

    def test_empty_extract_returns_zero(self, calibration):
        """No operations → T_core_floor = 0."""
        extract = HIVMExtract(operations=[], handoffs=[])
        result = compute_component_floor(
            extract,
            calibration["cube"],
            calibration["vector"],
            calibration["memory"],
            calibration["core"],
        )
        assert result.t_core_floor_us == 0.0

    def test_scalar_only_zero_time(self, calibration):
        """Scalar work has no separate throughput (accounted via overhead)."""
        ops = [
            OpRecord(op_id=1, op_name="cmp", component=Component.SCALAR,
                     precision=Precision.INT32, pipe="Scalar",
                     bytes_transferred=0, elements=1000,
                     duration_cycles=10, loop_multiplier=1, depends_on=[]),
        ]
        extract = HIVMExtract(operations=ops, handoffs=[])
        result = compute_component_floor(
            extract,
            calibration["cube"],
            calibration["vector"],
            calibration["memory"],
            calibration["core"],
        )
        # Scalar should be in per_component_us but with 0 time
        assert "scalar" in result.per_component_us
        assert result.per_component_us["scalar"] == 0.0


# ── Integration: Full M4+M5 Pipeline ───────────────────────────────────────

class TestIntegration:
    """End-to-end bound computation with known intermediate values."""

    def test_matmul_bound_matches_golden(self, calibration, matmul_extract):
        """MatMul-only kernel: MTE_GM binds, no serialization."""
        from perfbound.model.component_model import compute_component_floor
        from perfbound.model.grid_model import compute_grid_floor, GridBound
        from perfbound.model.serialization import classify_handoffs
        from perfbound.combine.bound_combiner import combine
        from perfbound.extract.dsl_extractor import GridInfo

        comp = compute_component_floor(
            matmul_extract,
            calibration["cube"],
            calibration["vector"],
            calibration["memory"],
            calibration["core"],
        )

        # Grid: uniform work, 512 programs, 20 cores, occupancy=1.0
        grid_info = GridInfo(
            grid_dims=(16, 32, 1), total_programs=512,
            tile_assignment={}, work={p: 1.0 for p in range(512)},
            occupancy=1.0, load_balance=1.0, redundancy=1.0,
            busiest_core_id=0,
        )
        i_binding = calibration["memory"].lookup_bw("gm", "ub")[0]
        total_bytes = sum(op.bytes_transferred * op.loop_multiplier
                         for op in matmul_extract.operations)
        grid = GridBound(
            t_grid_floor_us=total_bytes / (20 * 1.0 * 1.0 * i_binding),
            total_work=float(total_bytes), n_cores=20,
            occupancy=1.0, load_balance=1.0, redundancy=1.0,
            i_binding=i_binding, busiest_core_id=0,
        )

        serial = classify_handoffs([], mandatory_handoff_cycles=2000, clock_ghz=1.85)

        result = combine(grid, comp, serial, kernel_name="test_matmul")

        # Golden: T_bound = max(0.114, 2.185) + 0 = 2.185 us
        assert abs(result.t_bound_us - 2.185) < 0.02, \
            f"T_bound {result.t_bound_us:.3f} not ~2.185 us"
        assert result.binding_tier.value == "component"
        assert result.binding_component == Component.MTE_GM

    def test_attribution_gaps_wired(self, calibration):
        """Five-way attribution with mis-placed op: Gap 1 > 0, Gap 2/4 = 0.

        An 'add' op assigned to Cube (eligible={Vector}) should produce
        gap1_wrong_unit_us > 0, while Gap 2 and Gap 4 remain 0 (no
        small-packet params, no repeat/mask data).
        """
        from perfbound.model.component_model import compute_component_floor
        from perfbound.model.grid_model import GridBound
        from perfbound.model.serialization import classify_handoffs
        from perfbound.combine.bound_combiner import combine

        # Extract with a mis-placed op: 'add' on Cube pipe (should be Vector)
        ops = [
            OpRecord(op_id=1, op_name="add", component=Component.CUBE,
                     precision=Precision.FP16, pipe="Cube",
                     bytes_transferred=0, elements=4096,
                     duration_cycles=10, loop_multiplier=1, depends_on=[]),
            OpRecord(op_id=2, op_name="vector_add", component=Component.VECTOR,
                     precision=Precision.FP16, pipe="Vector",
                     bytes_transferred=0, elements=4096,
                     duration_cycles=10, loop_multiplier=1, depends_on=[]),
        ]
        extract = HIVMExtract(operations=ops, handoffs=[])

        comp = compute_component_floor(
            extract, calibration["cube"], calibration["vector"],
            calibration["memory"], calibration["core"],
        )

        grid = GridBound(
            t_grid_floor_us=0.1, total_work=8192, n_cores=20,
            occupancy=1.0, load_balance=1.0, redundancy=1.0,
            i_binding=100.0, busiest_core_id=0,
        )
        serial = classify_handoffs([], mandatory_handoff_cycles=0, clock_ghz=1.85)

        result = combine(grid, comp, serial, kernel_name="gap_test",
                         extract=extract)

        # Gap 1: 'add' on Cube (eligible=Vector) → mis-placed
        assert result.attribution.gap1_wrong_unit_us > 0, \
            f"Expected Gap 1 > 0, got {result.attribution.gap1_wrong_unit_us:.4f}"
        # Gap 2: no small-packet params → 0
        assert result.attribution.gap2_coalescing_us == 0.0
        # Gap 3: no handoffs → 0
        assert result.attribution.gap3_avoidable_serial_us == 0.0
        # Gap 4: no repeat/mask data → 0
        assert result.attribution.gap4_intra_unit_exec_us == 0.0

        # Dominant gap should be gap1
        name, frac = result.attribution.dominant_gap()
        assert name == "gap1_wrong_unit", \
            f"Expected dominant gap 'gap1_wrong_unit', got '{name}'"
