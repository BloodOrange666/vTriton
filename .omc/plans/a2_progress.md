# A.2 Implementation Progress

**Completed**: 2026-06-08  
**Status**: ✅ Complete — all acceptance criteria verified, code review findings resolved

---

## Architecture

```
TTIR (.ttir) → tritonsim-opt --extract-ttir-info → JSON stdout
                                          │
              ┌───────────────────────────┘
              │ Python perfbound/extract/ package
              │
              │ mlir_parser.py   (parse_ttir: subprocess → json.loads)
              │ dsl_extractor.py (GridInfo extraction, persistent path)
              │ grid_idioms.py   (idiom templates: 1D/2D)
              │
              │ Output: GridInfo, Tier 1 quantities
              └──────────────────────────────────────
```

**Note**: Direct single-stage pipeline (`tritonsim-opt --extract-ttir-info`) is used
instead of two-stage (`triton-opt | tritonsim-opt`) because `--mlir-print-op-generic`
converts `tt.*` ops to generic representation, preventing the C++ pass from recognizing
`triton::GetProgramIdOp`, `triton::MakeTensorPtrOp`, etc.

---

## Bug Fixes (Original A.2 scope)

| Bug | Location | Fix | Impact |
|-----|----------|-----|--------|
| **Bug 1** | `grid_idioms.py:46` | `parents[3]` → `parents[2]` | Correct path depth; `ascend_910b3.json` now loads |
| **Bug 2** | `dsl_extractor.py` | Removed all 4 `_parse_*()` regex functions | No fragile text matching; replaced by C++ AST pass |
| **Bug 3** | `dsl_extractor.py` | New `_persistent_kernel_info()` with round-robin model | Correct `occupancy`/`load_balance` for persistent kernels |

---

## C++ Implementation

| File | Key Features |
|------|--------------|
| `lib/AscendModel/Transforms/ExtractTTIRInfo.cpp` | Walks TTIR via `triton::GetProgramIdOp`, `scf::ForOp`, `triton::MakeTensorPtrOp`, `triton::DotOp`. Emits JSON via `llvm::json::OStream`. Only emits persistent loops (`lb_is_pid=true` — emits only when lower bound is `tt.get_program_id`). |
| `include/AscendModel/Transforms/Passes.td` | `ExtractTTIRInfoPass` entry with dependent dialects |
| `lib/AscendModel/Transforms/CMakeLists.txt` | Added `ExtractTTIRInfo.cpp` |

**JSON output schema:**
```json
{
  "grid_axes": [0],
  "persistent_loops": [{"lb_is_pid": true, "ub_value": 4, "step_value": 20}],
  "tensor_ptr_shapes": [[32, 128], [128, 128]],
  "has_dot": true
}
```

---

## Python Implementation

| File | Key Features |
|------|--------------|
| `perfbound/extract/mlir_parser.py` | `parse_ttir()` calls `tritonsim-opt --extract-ttir-info` via subprocess. `PROJECT_ROOT = Path(__file__).resolve().parents[2]` (one-liner, no complex helper). Brace-counting JSON extractor isolates JSON from surrounding MLIR dump text. |
| `perfbound/extract/dsl_extractor.py` | Refactored `_extract_from_ttir()` to use `parse_ttir()`. New `_persistent_kernel_info()` implements round-robin tile assignment. Inline TTIR text raises clear `ValueError` (unsupported). |
| `perfbound/extract/grid_idioms.py` | Path depth fixed (`parents[2]`). `_load_capacity_from_config()` correctly reads `ascend_910b3.json`. |

**Persistent kernel work model:**
```python
stride = pers_loop["step_value"]          # = n_programs (e.g., 20)
block_m = shapes[0][0]                    # tile dimension from AST (e.g., 32)
total_tiles = math.ceil(M / block_m)      # from problem_shape

for p in range(n_programs):
    my_tiles = list(range(p, total_tiles, n_programs))  # round-robin
    work[p] = len(my_tiles)

occupancy = n_active / n_programs         # e.g., 4/20 = 0.2
load_balance = sum(active) / (max_w * n_active)
```

---

## Code Review Fixes (2026-06-08)

All 7 findings from post-implementation review were resolved:

| # | Finding | Fix Applied |
|---|---------|-------------|
| F1 | `block_sizes` NameError in `_persistent_kernel_info` | Removed unreachable `elif "BLOCK_M" in block_sizes:` — default `block_m=128` handles empty shapes |
| F2 | Module-level `pytestmark` skipped all 13 tests when tritonsim-opt absent | Replaced with named `requires_tritonsim` mark applied per-case via `pytest.param(..., marks=requires_tritonsim)` |
| F3 | CWD-relative TTIR paths in parametrize table and standalone tests | All paths now use `str(_TEST_DIR / "flash_attention.ttir")` where `_TEST_DIR = Path(__file__).parents[2] / "test"` |
| F4 | C++ emitted all const-bounds `scf.for` loops as `persistent_loops` | Condition changed to `if (lbIsPid)` — only emits persistent loops |
| F5 | Inline TTIR text silently produced confusing `FileNotFoundError` | Raises `ValueError("Inline TTIR text is not supported. Write the TTIR to a file...")` |
| F6 | Redundant nested `#ifdef TRITONSIM_HAS_TRITON` in ExtractTTIRInfo.cpp | Inner ifdef removed |
| F7 | Dead `elif` branch in `_get_project_root()` | Entire function replaced with `PROJECT_ROOT = Path(__file__).resolve().parents[2]` |

---

## Test Coverage (63/63 passing)

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `tests/perfbound/test_mlir_parser.py` | 5 | C++ pass JSON output (grid_axes, persistent_loops, tensor_ptr_shapes, has_dot) |
| `tests/perfbound/test_grid_idioms.py` | 5 | Idiom templates (1D exact, 1D remainder, 2D tile grid, UB violation, config loading) |
| `tests/perfbound/test_dsl_extractor.py` | 13 | 10 parametrized reference kernels (K1–K10) + 3 standalone end-to-end tests |

Skip guards: K4, K5, K10 and 3 standalone tests use `@requires_tritonsim`; K1-K3, K6-K9 run without binary.

### Reference Kernel Cases (K1-K10)

| Case | Description | Expected |
|------|-------------|---------|
| K1 | 1D exact | occ=0.2, lb=1.0 |
| K2 | 1D remainder | occ=0.2, lb≈0.781, work[3]=4 |
| K3 | 2D exact | occ=0.8, lb=1.0 |
| K4 | Persistent flash_attn (4/20 tiles) | occ=0.2, lb=1.0, work[0]=1, work[4]=0 |
| K5 | Persistent 21/20 tiles | occ=1.0, lb≈0.525, work[0]=2, work[1]=1 |
| K6 | 2D uneven | occ=0.4, lb≈0.610, work[7]=144 |
| K7 | UB capacity violation | buffer_pressure_ok=False |
| K8 | G > n_cores (saturated) | occ=1.0, lb=1.0 |
| K9 | 1D over-subscribed | occ=1.0, lb=1.0 |
| K10 | Persistent 1/20 tiles | occ=0.05, lb=1.0, work[0]=1, work[1]=0 |

---

## Acceptance Criteria — All Verified

| AC | Criterion | Status |
|----|-----------|--------|
| AC-1 | `parse_ttir('test/flash_attention.ttir')['tensor_ptr_shapes'][0] == [32, 128]` | ✅ |
| AC-2 | `extract_grid_info(...).occupancy == 0.2`, `load_balance == 1.0` | ✅ |
| AC-3 | `len(grid.work) == 20`, `work[0]==1`, `work[4]==0` | ✅ |
| AC-4 | `get_capacities(192kb_json, force_reload=True)["ub"] == 192*1024` | ✅ |
| AC-5 | 10 reference kernel cases pass (tritonsim-opt cases skipped when not built) | ✅ |
| AC-6 | `pytest tests/ -q` — 63/63 passing | ✅ |
| AC-7 | No `re.finditer`/`re.search`/`re.match` in `perfbound/extract/` | ✅ |

---

## PROGRESS.md Items Closed

| Item | Resolution |
|------|-----------|
| `configs/ascend_910b3.json` path not wired | Bug 1 fix; `grid_idioms.py` now loads it correctly |
| M2 buffer capacity checks hardcoded | Config loading verified by `test_config_loading` |

---

## Ready for A.3

A.2 completes the **Tier 1 input pipeline** (DSL → TTIR → GridInfo). Next: **A.3 HIVM Extractor** — verify C++ JSON round-trip end-to-end on a real kernel via `tritonsim-opt`.
