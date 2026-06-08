"""
Tests for dsl_extractor.py — 10-kernel parametrized reference suite.

Each test case verifies extract_grid_info() produces correct Tier 1 quantities:
- occupancy, load_balance, tile_assignment, work distribution
- Buffer pressure validation
- Integration with grid floor computation
"""

import pytest
from pathlib import Path

# Project root and test fixture paths (absolute, not CWD-relative)
_PROJECT_ROOT = Path(__file__).parents[2]
_TRITONSIM_OPT = _PROJECT_ROOT / "build" / "bin" / "tritonsim-opt"
_TEST_DIR = _PROJECT_ROOT / "test"

# Mark for tests that require tritonsim-opt binary
requires_tritonsim = pytest.mark.skipif(
    not _TRITONSIM_OPT.exists(),
    reason=f"tritonsim-opt not built at {_TRITONSIM_OPT}"
)

from perfbound.extract.dsl_extractor import extract_grid_info


# === Reference Kernel Cases (K1-K10) ===

REFERENCE_KERNELS = [
    # K1: 1D exact division (idiom path — no tritonsim-opt needed)
    {
        "name": "K1_1d_exact",
        "description": "M=128, BM=32, G=(4,) → exact division, uniform work",
        "input": {"source": "python", "launch_grid": (4,), "problem_shape": (128,), "block_sizes": {"BLOCK_M": 32}, "n_cores": 20},
        "expected": {"occ": 0.2, "lb": 1.0, "total_programs": 4, "tile_assignment_sample": {0: (0,)}},
    },

    # K2: 1D with remainder (idiom path)
    {
        "name": "K2_1d_remainder",
        "description": "M=100, BM=32, G=(4,) → last program gets 4 rows",
        "input": {"source": "python", "launch_grid": (4,), "problem_shape": (100,), "block_sizes": {"BLOCK_M": 32}, "n_cores": 20},
        "expected": {"occ": 0.2, "lb": pytest.approx(0.781, rel=0.01), "total_programs": 4, "work_3": 4},
    },

    # K3: 2D exact (idiom path)
    {
        "name": "K3_2d_exact",
        "description": "M=128, N=256, BM=32, BN=64, G=(4,4) → 16 programs, uniform work",
        "input": {"source": "python", "launch_grid": (4, 4), "problem_shape": (128, 256), "block_sizes": {"BLOCK_M": 32, "BLOCK_N": 64}, "n_cores": 20},
        "expected": {"occ": 0.8, "lb": 1.0, "total_programs": 16, "work_0": 32*64},
    },

    # K4: Persistent flash_attention (requires tritonsim-opt)
    pytest.param(
        {
            "name": "K4_persistent_flash_attn",
            "description": "4 tiles, 20 programs → occupancy=0.2, round-robin assignment",
            "input": {"source": str(_TEST_DIR / "flash_attention.ttir"), "launch_grid": (20,), "problem_shape": (128,), "block_sizes": {}, "n_cores": 20},
            "expected": {"occ": 0.2, "lb": 1.0, "total_programs": 20, "work_0": 1, "work_4": 0},
        },
        marks=requires_tritonsim,
    ),

    # K5: Persistent 21/20 tiles (requires tritonsim-opt)
    pytest.param(
        {
            "name": "K5_persistent_21_20",
            "description": "21 tiles, 20 programs → occupancy=1.0, prog 0 gets 2 tiles",
            "input": {"source": str(_TEST_DIR / "persistent_21.ttir"), "launch_grid": (20,), "problem_shape": (672,), "block_sizes": {}, "n_cores": 20},
            "expected": {"occ": 1.0, "lb": pytest.approx(0.525, rel=0.01), "total_programs": 20, "work_0": 2, "work_1": 1},
        },
        marks=requires_tritonsim,
    ),

    # K6: 2D uneven (idiom path)
    {
        "name": "K6_2d_uneven",
        "description": "M=100, N=100, BM=32, BN=64, G=(4,2) → 8 programs, uneven work",
        "input": {"source": "python", "launch_grid": (4, 2), "problem_shape": (100, 100), "block_sizes": {"BLOCK_M": 32, "BLOCK_N": 64}, "n_cores": 20},
        "expected": {"occ": 0.4, "lb": pytest.approx(0.610, rel=0.01), "total_programs": 8, "work_0": 32*64, "work_7": 144},
    },

    # K7: UB violation (idiom path)
    {
        "name": "K7_ub_violation",
        "description": "BM=512KB → tile exceeds UB capacity",
        "input": {"source": "python", "launch_grid": (1,), "problem_shape": (4096, 4096), "block_sizes": {"BLOCK_M": 512*1024, "BLOCK_N": 512*1024}, "n_cores": 20},
        "expected": {"buffer_ok": False, "occ": 0.05, "lb": 1.0},
    },

    # K8: G > n_cores (saturated, idiom path)
    {
        "name": "K8_oversubscribed",
        "description": "G=(16,16) → 256 programs, only 20 cores → occupancy=1.0",
        "input": {"source": "python", "launch_grid": (16, 16), "problem_shape": (2048, 2048), "block_sizes": {"BLOCK_M": 128, "BLOCK_N": 128}, "n_cores": 20},
        "expected": {"occ": 1.0, "lb": 1.0, "total_programs": 256},
    },

    # K9: 1D over-subscribed (idiom path)
    {
        "name": "K9_1d_oversubscribed",
        "description": "M=4096, BM=128, G=(32,) → 32 programs, 20 cores → occupancy=1.0",
        "input": {"source": "python", "launch_grid": (32,), "problem_shape": (4096,), "block_sizes": {"BLOCK_M": 128}, "n_cores": 20},
        "expected": {"occ": 1.0, "lb": 1.0, "total_programs": 32},
    },

    # K10: Persistent 1/20 tiles (requires tritonsim-opt)
    pytest.param(
        {
            "name": "K10_persistent_1_20",
            "description": "1 tile, 20 programs → occupancy=0.05, only prog 0 works",
            "input": {"source": str(_TEST_DIR / "persistent_1.ttir"), "launch_grid": (20,), "problem_shape": (32,), "block_sizes": {}, "n_cores": 20},
            "expected": {"occ": 0.05, "lb": 1.0, "total_programs": 20, "work_0": 1, "work_1": 0},
        },
        marks=requires_tritonsim,
    ),
]


@pytest.mark.parametrize("case", REFERENCE_KERNELS)
def test_reference_kernel(case):
    """Parametrized test for all 10 reference kernel cases."""
    input_data = case["input"]
    expected = case["expected"]

    # Extract grid info
    if isinstance(input_data["source"], str) and input_data["source"].endswith(".ttir"):
        grid = extract_grid_info(
            input_data["source"],
            input_data["launch_grid"],
            input_data["problem_shape"],
            input_data["block_sizes"],
            input_data["n_cores"],
        )
    else:
        # Idiom path — use empty string (source="python" goes through idioms)
        grid = extract_grid_info(
            "",
            input_data["launch_grid"],
            input_data["problem_shape"],
            input_data["block_sizes"],
            input_data["n_cores"],
        )

    # Verify occupancy
    if "occ" in expected:
        assert grid.occupancy == pytest.approx(expected["occ"], rel=0.01), \
            f"{case['name']}: occupancy mismatch: expected {expected['occ']}, got {grid.occupancy}"

    # Verify load balance
    if "lb" in expected:
        expected_lb = expected["lb"]
        if isinstance(expected_lb, float):
            assert grid.load_balance == pytest.approx(expected_lb, rel=0.01), \
                f"{case['name']}: load_balance mismatch: expected {expected_lb}, got {grid.load_balance}"
        else:
            assert grid.load_balance == expected_lb, \
                f"{case['name']}: load_balance mismatch: expected {expected_lb}, got {grid.load_balance}"

    # Verify total_programs
    if "total_programs" in expected:
        assert grid.total_programs == expected["total_programs"], \
            f"{case['name']}: total_programs mismatch: expected {expected['total_programs']}, got {grid.total_programs}"

    # Verify tile assignment sample
    if "tile_assignment_sample" in expected:
        for prog_id, expected_tiles in expected["tile_assignment_sample"].items():
            actual_tiles = grid.tile_assignment.get(int(prog_id))
            assert actual_tiles == expected_tiles, \
                f"{case['name']}: tile_assignment[{prog_id}] mismatch: expected {expected_tiles}, got {actual_tiles}"

    # Verify specific work entries
    for key, expected_value in [("work_0", "work_0"), ("work_1", "work_1"), ("work_3", "work_3"), ("work_4", "work_4"), ("work_7", "work_7")]:
        if key in expected:
            prog_id = int(key.split("_")[1])
            actual_value = grid.work.get(prog_id)
            assert actual_value == expected[key], \
                f"{case['name']}: work[{prog_id}] mismatch: expected {expected[key]}, got {actual_value}"

    # Verify buffer pressure
    if "buffer_ok" in expected:
        assert grid.buffer_pressure_ok == expected["buffer_ok"], \
            f"{case['name']}: buffer_pressure_ok mismatch: expected {expected['buffer_ok']}, got {grid.buffer_pressure_ok}"


@requires_tritonsim
def test_extract_persistent_flash_attn():
    """End-to-end test on flash_attention.ttir."""
    grid = extract_grid_info(
        str(_TEST_DIR / "flash_attention.ttir"), (20,), (128,), {}, 20
    )

    assert grid.occupancy == 0.2
    assert grid.load_balance == 1.0
    assert len(grid.work) == 20
    assert grid.work[0] == 1
    assert grid.work[4] == 0
    assert grid.busiest_core_id == 0


@requires_tritonsim
def test_persistent_21_tiles():
    """Test persistent kernel with 21 tiles, 20 programs (K5)."""
    grid = extract_grid_info(
        str(_TEST_DIR / "persistent_21.ttir"), (20,), (672,), {}, 20
    )

    assert grid.occupancy == 1.0
    assert len(grid.work) == 20
    assert grid.work[0] == 2

    for i in range(1, 20):
        assert grid.work[i] == 1

    assert grid.load_balance == pytest.approx(0.525, rel=0.01)


@requires_tritonsim
def test_persistent_1_tile():
    """Test persistent kernel with 1 tile, 20 programs (K10)."""
    grid = extract_grid_info(
        str(_TEST_DIR / "persistent_1.ttir"), (20,), (32,), {}, 20
    )

    assert grid.occupancy == 0.05
    assert len(grid.work) == 20
    assert grid.work[0] == 1
    for i in range(1, 20):
        assert grid.work[i] == 0
    assert grid.load_balance == 1.0
