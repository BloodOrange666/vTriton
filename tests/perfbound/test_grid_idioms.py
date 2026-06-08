"""
Tests for grid_idioms.py — idiom template validation.

Tests verify that 1D and 2D grid idiom templates correctly compute
tile assignments, work distribution, and hardware-legality constraints.
"""

import pytest
from perfbound.extract.grid_idioms import (
    idiom_1d_row_block,
    idiom_2d_tile_grid,
    get_capacities,
    DEFAULT_UB_CAPACITY_BYTES,
    DEFAULT_L1_CAPACITY_BYTES,
)


def test_1d_exact():
    """Test 1D row-block idiom with exact division."""
    # M=128, BM=32 → G=4, each program handles 32 rows
    result = idiom_1d_row_block(M=128, BLOCK_M=32)

    # Should have 4 programs
    assert len(result.tile_assignment) == 4
    assert len(result.work) == 4

    # Each program should have equal work (32 rows)
    assert result.work[0] == 32
    assert result.work[1] == 32
    assert result.work[2] == 32
    assert result.work[3] == 32

    # Load balance should be perfect (1.0)
    works = list(result.work.values())
    assert sum(works) / (max(works) * len(works)) == 1.0

    # Buffer pressure should be OK (32 rows of FP16 = 64 bytes << 256KB UB)
    assert result.buffer_pressure_ok is True


def test_1d_remainder():
    """Test 1D row-block idiom with remainder."""
    # M=100, BM=32 → G=4, last program handles 4 rows
    result = idiom_1d_row_block(M=100, BLOCK_M=32)

    # Should have 4 programs
    assert len(result.work) == 4

    # Last program should handle remainder (4 rows)
    assert result.work[0] == 32
    assert result.work[1] == 32
    assert result.work[2] == 32
    assert result.work[3] == 4  # remainder

    # Load balance should be (100/4)/32 = 25/32 ≈ 0.781
    works = list(result.work.values())
    mean_work = sum(works) / len(works)  # 100/4 = 25
    max_work = max(works)  # 32
    expected_lb = mean_work / max_work  # 25/32 = 0.78125
    actual_lb = sum(works) / (max_work * len(works))
    assert abs(actual_lb - expected_lb) < 0.001


def test_2d_tile_grid():
    """Test 2D tile grid idiom."""
    # M=128, N=256, BM=32, BN=64 → G_m=4, G_n=4, G=16
    result = idiom_2d_tile_grid(M=128, N=256, BLOCK_M=32, BLOCK_N=64)

    # Should have 16 programs
    assert len(result.work) == 16

    # First program (tile_m=0, tile_n=0) should work 32*64 = 2048 elements
    assert result.work[0] == 2048

    # Buffer pressure should be OK (32*64*2 = 4KB << 256KB UB, 32*2 = 64 bytes << 1MB L1)
    assert result.buffer_pressure_ok is True


def test_ub_violation():
    """Test UB capacity violation detection."""
    # BM=512*1024 = 524288 elements > 256KB UB
    # With FP16 (2 bytes per element): 512KB * 2 = 1MB > 256KB
    result = idiom_1d_row_block(M=4096, BLOCK_M=512*1024)

    # Buffer pressure should NOT be OK
    assert result.buffer_pressure_ok is False


def test_config_loading(tmp_path):
    """Test config loading from custom JSON file."""
    import json

    # Create temporary config with non-default UB size
    config_data = {
        "memory_spaces": {
            "ub": {"size_kb": 192},  # 192KB instead of 256KB
            "l1": {"size_kb": 1024},
            "l0a": {"size_kb": 64},
            "l0b": {"size_kb": 64},
            "l0c": {"size_kb": 256},
        }
    }

    config_file = tmp_path / "test_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f)

    # Force reload and verify UB capacity
    caps = get_capacities(str(config_file), force_reload=True)
    assert caps["ub"] == 192 * 1024, f"Expected {192*1024}, got {caps['ub']}"
    assert caps["l1"] == 1024 * 1024
