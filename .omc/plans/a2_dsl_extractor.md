# A.2 Plan: DSL Extractor — C++ MLIR Pass for TTIR Extraction + Bug Fixes

## Context

A.2 (M2 DSL Extractor) turns TTIR into `GridInfo` (Tier 1 input for `T_grid_floor`).
The scaffold in `dsl_extractor.py` + `grid_idioms.py` has three bugs and zero tests. All four
`_parse_*()` functions use `re.finditer` — fragile text matching that silently captures wrong
MLIR structure (e.g., shape operands instead of result type annotation).

**Design decision**: Replace all TTIR text parsing with a new C++ MLIR pass in `tritonsim-opt`
that walks the IR AST and emits structured JSON. Python reads JSON via subprocess. This follows
the same pattern as `emitDESGraph()` / `emitDependencyGraphJSON()` already in the project.

**Empirical proof of bugs** (`extract_grid_info('test/flash_attention.ttir', (20,), (128,), {}, 20)`):
```
occupancy: 1.0   (WRONG — should be 0.2; only 4/20 programs have tiles)
work:      {0: 128}  (WRONG — 1 entry not 20; BLOCK_M 32 misread as 128)
```

---

## Bug Inventory

**Bug 1 — `grid_idioms.py:46` wrong `parents` depth**
`parents[3]` resolves to `/mnt/d/work/git/`, not the project root.
Config is never loaded → capacity checks silently use hardcoded defaults.
Fix: `parents[3]` → `parents[2]`.

**Bug 2 — All `_parse_*()` functions use fragile regex on MLIR text**
Regex matches structural parts of MLIR wrong (e.g., `_parse_tile_shapes` captures shape
operands `[%c128_i64, %c128_i64]` = matrix bounds 128×128, instead of the result type
`tensor<32x128xf16>` = tile size 32×128). Fix: C++ MLIR AST pass.

**Bug 3 — Persistent kernel path uses wrong work model**
Falls through to `idiom_1d_row_block(128, 128)` → 1 program, work={0:128}.
`_idiom_to_grid` then uses `product(launch_grid)=20` as total_programs, creating a mismatch.
Even with Bug 2 fixed, zero-work programs included in `load_balance` computation conflates
occupancy with balance. Fix: dedicated persistent path with round-robin assignment and
correct active-only occupancy/load_balance formulas.

---

## C++ Changes: `ExtractTTIRInfoPass`

### New file: `lib/AscendModel/Transforms/ExtractTTIRInfo.cpp`

Pass walks the TTIR module using the MLIR C++ AST API and emits JSON to stdout.

**JSON output schema:**
```json
{
  "grid_axes": [0],
  "persistent_loops": [
    {"lb_is_pid": true, "ub_value": 4, "step_value": 20}
  ],
  "tensor_ptr_shapes": [[32, 128], [128, 128], [128, 128], [32, 128]],
  "has_dot": true
}
```

**Key C++ API calls** (no text parsing):

```cpp
#include "AscendModel/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"

// Grid axes: triton::GetProgramIdOp::getAxis() → 0/1/2
module.walk([&](triton::GetProgramIdOp op) {
    axesArr.push_back(llvm::json::Value(op.getAxisAsInteger()));
});

// Persistent loop: scf::ForOp lb defined by GetProgramIdOp
module.walk([&](scf::ForOp forOp) {
    bool lbIsPid = forOp.getLowerBound()
                       .getDefiningOp<triton::GetProgramIdOp>() != nullptr;
    auto getConst = [](Value v) -> std::optional<int64_t> {
        if (auto cst = v.getDefiningOp<arith::ConstantOp>())
            if (auto intAttr = mlir::dyn_cast<IntegerAttr>(cst.getValue()))
                return intAttr.getInt();
        return std::nullopt;
    };
    auto ubVal   = getConst(forOp.getUpperBound());
    auto stepVal = getConst(forOp.getStep());
    loopsArr.push_back(llvm::json::Object{
        {"lb_is_pid", lbIsPid},
        {"ub_value",   ubVal.value_or(-1)},
        {"step_value", stepVal.value_or(-1)},
    });
});

// Tile shapes: triton::MakeTensorPtrOp result type → tensor shape
module.walk([&](triton::MakeTensorPtrOp op) {
    auto ptrType = mlir::dyn_cast<triton::PointerType>(op.getResult().getType());
    if (!ptrType) return;
    auto tensorType = mlir::dyn_cast<RankedTensorType>(ptrType.getPointeeType());
    if (!tensorType) return;
    llvm::json::Array dims;
    for (int64_t d : tensorType.getShape())
        dims.push_back(llvm::json::Value(d));
    shapesArr.push_back(llvm::json::Value(std::move(dims)));
});

// Cube kernel detection: presence of triton::DotOp
bool hasDot = false;
module.walk([&](triton::DotOp) { hasDot = true; });
```

Emit via `llvm::outs()` (same pattern as `RooflineAnalysis.cpp:255`).

### `include/AscendModel/Transforms/Passes.td` — add entry

```tablegen
def ExtractTTIRInfoPass : Pass<"extract-ttir-info", "ModuleOp"> {
  let summary = "Extract TTIR structural info as JSON (grid axes, loop bounds, tile shapes)";
  let description = [{
    Walks the TTIR module and emits a JSON object to stdout containing:
    - grid_axes: program_id axis indices (0=x, 1=y, 2=z)
    - persistent_loops: scf.for loops where lb = program_id, with ub and step constants
    - tensor_ptr_shapes: shapes of tt.make_tensor_ptr result tensor types
    - has_dot: true if any tt.dot op is present (indicates Cube kernel)
  }];
  let constructor = "mlir::ascend::createExtractTTIRInfoPass()";
  let dependentDialects = [
    "triton::TritonDialect",
    "scf::SCFDialect",
    "arith::ArithDialect"
  ];
}
```

### `lib/AscendModel/Transforms/CMakeLists.txt` — add `ExtractTTIRInfo.cpp`

---

## Python Changes

### New file: `perfbound/extract/mlir_parser.py`

Wraps the C++ pass invocation. No MLIR text parsing — just subprocess + `json.loads`.

```python
import subprocess, json
from pathlib import Path

TRITONSIM_OPT = Path(__file__).parents[3] / "build" / "bin" / "tritonsim-opt"
TRITON_OPT = Path("/mnt/d/work/git/triton-ascend/python/build"
                  "/cmake.linux-x86_64-cpython-3.12/bin/triton-opt")

def parse_ttir(ttir_path: str) -> dict:
    """Run the C++ ExtractTTIRInfoPass and return parsed JSON."""
    # Two-stage pipeline: triton-opt converts to generic form, tritonsim-opt extracts
    cmd = (
        f"{TRITON_OPT} {ttir_path} --allow-unregistered-dialect --mlir-print-op-generic"
        f" | {TRITONSIM_OPT} - --allow-unregistered-dialect --extract-ttir-info"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return json.loads(result.stdout)
```

**`parse_ttir()` returns**: `dict` with keys `grid_axes`, `persistent_loops`,
`tensor_ptr_shapes`, `has_dot` — as specified by the C++ JSON schema above.

### Refactor `perfbound/extract/dsl_extractor.py`

- Remove `_parse_constants()`, `_parse_get_program_id()`, `_parse_persistent_loop()`,
  `_parse_tile_shapes()` (the 4 regex functions)
- Replace the body of `_extract_from_ttir()` with:
  ```python
  from .mlir_parser import parse_ttir
  info = parse_ttir(source_path)  # source_path is the .ttir file path
  axes  = [["x","y","z"][i] for i in info["grid_axes"]]
  loops = info["persistent_loops"]      # [{lb_is_pid, ub_value, step_value}]
  shapes = [tuple(s) for s in info["tensor_ptr_shapes"]]
  has_dot = info["has_dot"]
  ```
- Dispatch logic (persistent vs 1D vs 2D) unchanged in structure, but uses structured
  `info` dict instead of regex results.

### Fix Bug 1: `grid_idioms.py:46`

`parents[3]` → `parents[2]`. Update comment.

### Fix Bug 3: Persistent kernel path in `_extract_from_ttir()`

Replace the `is_persistent` branch with direct `GridInfo` construction:
```python
if is_persistent:
    pers_loop = next(l for l in loops if l["lb_is_pid"])
    stride = pers_loop["step_value"]        # = n_programs = n_cores
    block_m = shapes[0][0] if shapes else block_sizes.get("BLOCK_M", 128)
    # Derive total_tiles from problem_shape (authoritative); ub_value is cross-check
    total_tiles = math.ceil(problem_shape[0] / block_m) if problem_shape else pers_loop["ub_value"]
    n_programs = stride

    tile_assignment = {}
    work = {}
    for p in range(n_programs):
        my_tiles = list(range(p, total_tiles, n_programs))
        tile_assignment[p] = tuple(t * block_m for t in my_tiles)
        work[p] = len(my_tiles)          # tile count

    active = [w for w in work.values() if w > 0]
    n_active = len(active)
    max_w = max(active) if active else 1
    occ = n_active / n_programs
    lb  = (sum(active) / (max_w * n_active)) if active else 1.0

    return GridInfo(
        grid_dims=(n_programs,), total_programs=n_programs,
        tile_assignment=tile_assignment, work=work,
        occupancy=occ, load_balance=lb, redundancy=1.0,
        busiest_core_id=max(work, key=work.get) if work else 0,
        buffer_pressure_ok=True, divisibility_ok=True,
    )
```

**Validated values for flash_attention.ttir** (4 tiles, 20 programs):
- `shapes[0] = (32, 128)` → block_m=32 (correct tile size from C++ AST, not text)
- `total_tiles = ceil(128/32) = 4`, n_programs=20
- `occupancy = 4/20 = 0.2`, `load_balance = 1.0` ✓

---

## Build Step

After C++ changes: `cd /mnt/d/work/git/vTriton/build && ninja -j$(nproc)`
All tests require the rebuilt `tritonsim-opt` at `build/bin/tritonsim-opt`.

---

## Reference Kernel Suite — 10 Cases (Spec AC)

Spec acceptance: *"on 10 reference kernels, the recovered `tile_assignment` matches a
hand-derived map; `occupancy` and `load_balance` match a manual calculation; tilings that
violate a buffer capacity are correctly rejected as illegal."*

Cases 1–3, 6–10 use the **idiom path** (Python source / direct block_sizes — no TTIR file).
Cases 4–5 use the **TTIR path** (C++ ExtractTTIRInfoPass).

All cases use `n_cores=20`.

| # | Pattern | Inputs | occ | lb | tile_assignment (sample) | buffer_ok |
|---|---------|--------|-----|----|--------------------------|-----------|
| K1 | 1D exact | M=128, BM=32, G=(4,) | 0.2 | 1.0 | {0:(0,), 1:(32,), 2:(64,), 3:(96,)} | ✓ |
| K2 | 1D remainder | M=100, BM=32, G=(4,) | 0.2 | 0.781 | {3:(96,)}, work[3]=4 | ✓ |
| K3 | 2D exact | M=128, N=256, BM=32, BN=64, G=(4,4) | 0.8 | 1.0 | {0:(0,0), 15:(96,192)} | ✓ |
| K4 | Persistent, 4/20 tiles | flash_attention.ttir, problem=(128,) | 0.2 | 1.0 | work[0]=1, work[4]=0 | ✓ |
| K5 | Persistent, 21/20 tiles | persistent_21.ttir, problem=(672,) | 1.0 | 0.525 | work[0]=2, work[1]=1 | ✓ |
| K6 | 2D uneven | M=100, N=100, BM=32, BN=64, G=(4,2) | 0.4 | 0.610 | work[0]=2048, work[7]=144 | ✓ |
| K7 | UB violation | M=4096, N=4096, BM=512, BN=512 | — | — | any | **✗** |
| K8 | G > n_cores saturated | M=2048, N=2048, BM=128, BN=128, G=(16,16) | 1.0 | 1.0 | {0:(0,0)} | ✓ |
| K9 | 1D over-subscribed | M=4096, BM=128, G=(32,) | 1.0 | 1.0 | {0:(0,), 31:(3968,)} | ✓ |
| K10 | Persistent, 1/20 tiles | persistent_1.ttir, problem=(32,) | 0.05 | 1.0 | work[0]=1, work[1]=0 | ✓ |

**Hand-derived formulas** (reference):
- `occupancy = min(G, n_cores) / n_cores` for non-persistent
- Persistent: `occupancy = min(total_tiles, n_programs) / n_programs`
- `load_balance = mean(work) / max(work)` across all programs
- K2: `lb = (100/4)/32 = 25/32 = 0.781`
- K5: prog 0 gets tiles 0,20 (2 tiles); progs 1-19 get 1 tile → `lb = (21/20)/2 = 0.525`
- K6: total_work=10000, max=2048 → `lb = (10000/8)/2048 = 1250/2048 = 0.610`
- K7: tile_bytes = 512×512×2 = 524288 > 262144 (256 KB) → `buffer_pressure_ok=False`

**Fixture files needed** (add to `test/`):
- `test/persistent_21.ttir` — minimal TTIR with `scf.for %pid to %c21_i32 step %c20_i32`,
  `tt.make_tensor_ptr ... : <tensor<32xf32>>`
- `test/persistent_1.ttir` — same with upper=1, step=20

Fixtures are hand-written minimal valid TTIR (no Python Triton compilation needed); only
the structural ops matter for the extractor.

The 10 cases are implemented as parametrized tests in `test_dsl_extractor.py`:
```python
@pytest.mark.parametrize("case", REFERENCE_KERNELS)  # list of (input, expected) dicts
def test_reference_kernel(case):
    grid = extract_grid_info(...)
    assert grid.occupancy == pytest.approx(case["occ"], rel=1e-3)
    assert grid.load_balance == pytest.approx(case["lb"], rel=1e-3)
    assert grid.buffer_pressure_ok == case["buffer_ok"]
    for p, tiles in case["tile_assignment_sample"].items():
        assert grid.tile_assignment[p] == tiles
```

---

## Open Items Closed by A.2

| PROGRESS.md item | Closed by |
|---|---|
| `configs/ascend_910b3.json` | File exists; Bug 1 fix completes path wiring |
| M2 buffer capacity checks hardcoded | Bug 1 fix + config-loading test with non-default UB |

---

## Tests (all in `tests/perfbound/`)

### `test_mlir_parser.py` — C++ pass output tests

| Test | Input | Expected |
|---|---|---|
| `test_parse_grid_axes` | flash_attention.ttir | `grid_axes == [0]` (x-axis) |
| `test_parse_persistent_loop` | flash_attention.ttir | `persistent_loops[0]["lb_is_pid"] == True, ub==4, step==20` |
| `test_parse_tensor_ptr_shapes_first` | flash_attention.ttir | `tensor_ptr_shapes[0] == [32, 128]` (tile, not matrix bounds) |
| `test_parse_has_dot` | flash_attention.ttir | `has_dot == True` |

All tests in this file skip gracefully if `build/bin/tritonsim-opt` is absent:
```python
pytestmark = pytest.mark.skipif(not Path("build/bin/tritonsim-opt").exists(),
                                reason="tritonsim-opt not built")
```

### `test_dsl_extractor.py` — 10-kernel parametrized suite + integration

- 10-kernel `@pytest.mark.parametrize` suite (K1–K10, see table above)
- `test_extract_persistent_flash_attn`: end-to-end on flash_attention.ttir
- Integration: `extract_grid_info()` → `compute_grid_floor()` → `t_grid_floor_us > 0`

### `test_grid_idioms.py` — idiom functions (no tritonsim-opt required)

| Test | Input | Expected |
|---|---|---|
| `test_1d_exact` | M=128, BLOCK_M=32 | G=4, all work[p]=32, load_balance=1.0 |
| `test_1d_remainder` | M=100, BLOCK_M=32 | G=4, work[3]=4 |
| `test_2d_tile_grid` | M=128, N=256, BM=32, BN=64 | G=16, work[0]=2048 |
| `test_ub_violation` | BM=512*1024, ub_limit=256*1024 | `buffer_pressure_ok=False` |
| `test_config_loading` | temp JSON with `ub.size_kb=192` | `get_capacities(force_reload=True)["ub"]==192*1024` |

---

## Acceptance Criteria

| # | Criterion |
|---|---|
| AC-1 | `parse_ttir("test/flash_attention.ttir")["tensor_ptr_shapes"][0] == [32, 128]` |
| AC-2 | `extract_grid_info(...).occupancy == 0.2`, `load_balance == 1.0` for flash_attention |
| AC-3 | `len(grid.work) == 20`, `work[0]==1`, `work[4]==0` |
| AC-4 | `get_capacities(192kb_json, force_reload=True)["ub"] == 192*1024` |
| AC-5 | All 10 reference kernel cases pass |
| AC-6 | `pytest tests/ -q` — all pass (tritonsim-opt tests skipped if not built) |
| AC-7 | No `re.finditer`/`re.search`/`re.match` in `perfbound/extract/` |

---

## Scope Boundary

Covers: 1D persistent, 1D non-persistent, 2D tile grid idioms + C++ MLIR extraction.
Deferred: General symbolic affine recovery (arbitrary `arith.divsi`/`remsi` chains) — Phase 2.

---

## Verification

```bash
# Build after C++ changes
cd /mnt/d/work/git/vTriton/build && ninja -j$(nproc)

# Smoke-check C++ pass directly (two-stage pipeline)
triton_opt=/mnt/d/work/git/triton-ascend/python/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt
tritonsim=/mnt/d/work/git/vTriton/build/bin/tritonsim-opt
${triton_opt} test/flash_attention.ttir --allow-unregistered-dialect --mlir-print-op-generic \
  | ${tritonsim} - --allow-unregistered-dialect --extract-ttir-info 2>/dev/null

# Smoke-check Python extractor (post-fix)
python3 -c "
from perfbound.extract.dsl_extractor import extract_grid_info
g = extract_grid_info('test/flash_attention.ttir', (20,), (128,), {}, 20)
assert g.occupancy == 0.2, g.occupancy
assert g.load_balance == 1.0, g.load_balance
assert len(g.work) == 20, len(g.work)
print('all checks pass')
"

# Full test suite
pytest tests/ -q
```
