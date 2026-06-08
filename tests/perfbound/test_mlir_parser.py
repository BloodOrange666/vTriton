"""
Tests for mlir_parser.py — C++ ExtractTTIRInfo pass output validation.

Tests verify that the C++ pass correctly extracts TTIR structural information
and the Python wrapper parses the JSON output correctly.
"""

import json
import pytest
from pathlib import Path

# Skip all tests if tritonsim-opt not built
TRITONSIM_OPT = Path(__file__).parents[2] / "build" / "bin" / "tritonsim-opt"
pytestmark = pytest.mark.skipif(
    not TRITONSIM_OPT.exists(),
    reason=f"tritonsim-opt not built at {TRITONSIM_OPT}"
)


def test_parse_grid_axes(flash_attention_ttir):
    """Verify grid_axes extraction for flash_attention.ttir."""
    from perfbound.extract.mlir_parser import parse_ttir

    info = parse_ttir(flash_attention_ttir)
    assert info["grid_axes"] == [0], f"Expected [0], got {info['grid_axes']}"


def test_parse_persistent_loop(flash_attention_ttir):
    """Verify persistent loop extraction for flash_attention.ttir."""
    from perfbound.extract.mlir_parser import parse_ttir

    info = parse_ttir(flash_attention_ttir)
    loops = info["persistent_loops"]
    assert len(loops) >= 1, f"Expected at least 1 loop, got {len(loops)}"

    loop = loops[0]
    assert loop["lb_is_pid"] is True, f"Expected lb_is_pid=True, got {loop['lb_is_pid']}"
    assert loop["ub_value"] == 4, f"Expected ub_value=4, got {loop['ub_value']}"
    assert loop["step_value"] == 20, f"Expected step_value=20, got {loop['step_value']}"


def test_parse_tensor_ptr_shapes_first(flash_attention_ttir):
    """Verify tensor_ptr_shapes extraction returns tile size, not matrix bounds."""
    from perfbound.extract.mlir_parser import parse_ttir

    info = parse_ttir(flash_attention_ttir)
    shapes = info["tensor_ptr_shapes"]
    assert len(shapes) >= 1, f"Expected at least 1 shape, got {len(shapes)}"

    # First shape should be [32, 128] (tile size), NOT [128, 128] (matrix bounds)
    # This is Bug 2 fix verification
    first_shape = shapes[0]
    assert first_shape == [32, 128], f"Expected [32, 128], got {first_shape}"


def test_parse_has_dot(flash_attention_ttir):
    """Verify Cube kernel detection via tt.dot presence."""
    from perfbound.extract.mlir_parser import parse_ttir

    info = parse_ttir(flash_attention_ttir)
    assert info["has_dot"] is True, f"Expected has_dot=True, got {info['has_dot']}"


def test_parse_ttir_returns_complete_dict(flash_attention_ttir):
    """Verify parse_ttir returns all expected keys."""
    from perfbound.extract.mlir_parser import parse_ttir

    info = parse_ttir(flash_attention_ttir)
    expected_keys = {"grid_axes", "persistent_loops", "tensor_ptr_shapes", "has_dot"}
    assert set(info.keys()) == expected_keys, f"Expected {expected_keys}, got {set(info.keys())}"


# === Fixtures ===

@pytest.fixture
def flash_attention_ttir():
    """Path to flash_attention.ttir test fixture."""
    return Path(__file__).parents[2] / "test" / "flash_attention.ttir"
