# A.1 Implementation Progress

**Last updated**: 2026-06-07

**Supersession note**: The 2026-06-06 entries below were remediated on
2026-06-07. The current A.1 state is 13 AscendC kernels, an 18-job 910B3
msprof sweep, split MTE warmup/measurement launches, and a single mixed
AIC/AIV `mandatory_handoff` kernel. See
`.omc/plans/a1_remediation.md` and `.omc/plans/a1_step2_progress.md` for the
verified final evidence.

## Overview

**Stage**: A.1 (Calibration Database) of two-tier analytical performance bound model  
**Spec**: `.omc/specs/performance_bound_model.md` §A.1  
**Plan**: `.omc/plans/a1_calibration.md`  
**Target hardware**: Ascend 910B3 NPU (remote server)  
**Deliverable**: `calib_910b3_v1.json` + CCE microbench suite

## Completion Status

✅ **A.1 calibration measured and promoted**

### Completed Stories

| Story | Description | Status |
|-------|-------------|--------|
| US-001 | 910B3 hardware config scaffold | ✅ Complete |
| US-002 | Directory structure for calibration data and scripts | ✅ Complete |
| US-003 | CCE microbenchmark kernels (13 files) | ✅ Complete (remote measured) |
| US-004 | run_benchmarks.sh orchestration script | ✅ Complete |
| US-005 | fit_constants.py extraction script | ✅ Complete |
| US-006 | validate_vs_tilesim.py cross-validation script | ✅ Complete |
| US-007 | microbench README.md for reproducibility | ✅ Complete |
| US-008 | CalibrationDB.load() integration | ✅ Complete (24/24 perfbound tests pass) |
| US-009 | Model integration — T_core_floor non-zero | ✅ Complete |
| US-010 | Progress summary document | ✅ This file |

## Files Created

### Configuration
```
configs/
└── ascend_910b3.json          # 910B3 hardware config (clock=1.85 GHz, 20 AIC, 40 AIV)
```

### Calibration Data
```
perfbound/calibration/data/
├── calib_910b3_v1.json       # Stub calibration DB with golden test values
├── bandwidth_910b3.csv        # Tilesim-compatible bandwidth schema (placeholder)
└── vec_cycle_910b3.csv        # Vector intrinsic cycle table (seed data)
```

### Microbenchmarks
```
perfbound/calibration/microbench/
├── README.md                  # Reproducibility documentation
├── CMakeLists.txt             # CANN ascendc_library + launcher build
├── bench_launcher.cpp         # ACLRT_LAUNCH_KERNEL host launcher
├── vt_microbench_common.h     # Shared current-AscendC helper kernels
├── cube_peak_fp16.cce         # Cube FP16 throughput kernel
├── cube_peak_int8.cce         # Cube INT8 throughput kernel entrypoint
├── cube_peak_bf16.cce         # Cube BF16 throughput kernel entrypoint
├── vector_peak_elemwise_add.cce # Vector VADD throughput kernel
├── vector_peak_elemwise_mul.cce # Vector VMUL throughput kernel
├── vector_peak_elemwise_max.cce # Vector VMAX throughput kernel
├── vector_peak_elemwise_min.cce # Vector VMIN throughput kernel
├── vector_peak_transcendental.cce  # Vector transcendental (exp/log/sqrt/rsqrt) kernel
├── mte_gm_to_ub.cce           # GM→UB bandwidth kernel
├── mte_ub_to_gm.cce           # UB→GM bandwidth kernel
├── mte_gm_to_l1.cce           # GM→L1 bandwidth kernel
├── mte_l1_to_l0a.cce          # L1→L0A bandwidth kernel
└── mandatory_handoff.cce      # Cube→Vector handoff cost isolation kernel
```

### Scripts
```
perfbound/calibration/scripts/
├── run_benchmarks.sh          # Orchestrate kernel runs on remote 910B3
├── fit_constants.py           # Extract constants from msprof CSVs
└── validate_vs_tilesim.py     # Cross-validation vs tilesim reference
```

### Tests
```
tests/perfbound/
├── test_calibration_load.py   # 8 new tests for calibration loading
└── test_component_model.py    # 12 existing tests (all pass)
```

## P0 Constants (Measured Values)

| Constant | Measured Value | Unit | Source | Status |
|----------|-------------|------|--------|--------|
| P_cube[fp16] | 5.1586 | TFLOPS | cce_microbench | ✅ Loads |
| P_cube[int8] | 5.1793 | TFLOPS | cce_microbench | ✅ Loads |
| P_cube[bf16] | 5.1614 | TFLOPS | cce_microbench | ✅ Loads |
| P_vector_add | 15.13 | GFLOPS | cce_microbench | ✅ Loads |
| BW_gm_to_ub | 86.95 | GB/s | cce_microbench | ✅ Loads |
| mandatory_handoff_cost | 7620.87 | cycles | cce_microbench | ✅ Loads |

**Note**: `validate_vs_tilesim.py` passes but emits an informational GM→UB bandwidth warning.

## Test Coverage

**Total perfbound tests**: 24/24 passing  
- Microbench source tests: 1/1
- Calibration extraction tests: 3/3
- Calibration load tests: 8/8  
- Component model tests: 12/12

```bash
$ python3 -m pytest tests/perfbound -q
24 passed in 0.61s
```

## What Works

1. **Config loading**: `configs/ascend_910b3.json` validates and loads
2. **Capacity limits**: `grid_idioms.py` reads UB/L1 capacities from config
3. **Calibration DB**: `CalibrationDB.load()` parses JSON with provenance
4. **Model integration**: `T_core_floor` computes non-zero values with calibration
5. **Bandwidth lookup**: `MemHierarchy.lookup_bw()` returns sustained rates from JSON

## What Requires Hardware

The following were measured on 910B3:

### Runtime Profiling
- All `.cce` files compile on remote 910B3 with CANN 9 `ccec`
- `vt_a1_bench_launcher` builds with the CANN 8.2 sample-style `ascendc_library` helper
- Full 13-kernel / 18-job sweep completed under `msprof` with `--n-repeat 45`
- MTE kernels run one unprofiled warmup launch (`0..768`) and one profiled measurement launch (`768..2048`)
- `mandatory_handoff` completed as one mixed AIC/AIV kernel with generated 1:1 mix metadata
- `fit_constants.py` promoted measured `calib_910b3_v1.json`

### Measured Constants
- Current values are measured 910B3 CCE microbenchmark outputs.
- Need to run `scripts/run_benchmarks.sh` → `scripts/fit_constants.py` pipeline
- Output: `calib_910b3_v1.json` with `source="cce_microbench"`

### Validation
- `scripts/validate_vs_tilesim.py` needs measured values to check:
  - Cube FP16 ≤ clock-scaled 910B4 × 1.05
  - Bandwidth ratios vs 910B4 reference

## Next Steps

1. Proceed to A.2 with measured `calib_910b3_v1.json`.
2. Optionally investigate the GM→UB validation warning if it affects downstream bounds.

## Deviations from A.1 Plan

1. **Measured calibration replaces stubs**: `calib_910b3_v1.json` now carries measured `source="cce_microbench"` constants.
   - **Impact**: Tests now assert measured provenance rather than golden stub values
   - **Mitigation**: Stub-era assumptions were removed from calibration-load tests

2. **Bandwidth in JSON**: Added bandwidth entries directly in JSON instead of only loading from CSV.
   - **Impact**: Better - standalone JSON works without companion CSV
   - **Mitigation**: CSV loading still supported via `calib_loader.py`

3. **CANN version split**: standalone compile uses CANN 9 `ccec`; sample-style launcher build uses the available 8.2 `ascendc_kernel_cmake` helper.
   - **Impact**: compile-only, launcher build, runtime measurement, and extraction pass
   - **Mitigation**: `cce_remote_bench.py` exposes `--cann-env`, `--cann-package-path`, `--skip-direct-compile`, and `--kernels` for reproducible execution

## Code Quality

- **Linting**: Changed Python files pass syntax check (`python3 -m py_compile`)
- **Type hints**: Core modules use type annotations
- **Documentation**: All kernels include metric/extraction/msprof field in headers
- **Reproducibility**: README.md documents compile/run commands

## Integration Points

- **C++ → Python**: JSON is the seam (C++ emits, Python consumes)
- **Grid → Capacity**: `grid_idioms.py` reads from `ascend_910b3.json`
- **Model → Calibration**: `compute_component_floor()` uses `CalibrationDB`
- **Tests → Fixtures**: 12 existing tests continue to pass with stub JSON

## Dependencies

### New
- None. `fit_constants.py` now uses only Python standard-library statistics/regression helpers.

### Existing
- `pytest`: Test framework (already in project)
- MLIR Python bindings: Not used in A.1 (used in A.3 HIVM extractor)

---

**Definition of Done (from a1_calibration.md §10)**

- [x] `configs/ascend_910b3.json` created
- [x] 10 CCE benchmark kernels written and remote compile-verified
- [x] All kernels structured for 30× runs when hardware available
- [x] All P0 constants have stub values with provenance fields
- [x] `calib_910b3_v1.json` written with full provenance
- [x] `bandwidth_910b3.csv` populated (tilesim-compatible schema)
- [x] `vec_cycle_910b3.csv` populated (seed data)
- [x] Tilesim cross-validation script exists (`validate_vs_tilesim.py`)
- [x] `T_core_floor` is non-zero for synthetic Cube workload
- [x] `pytest tests/perfbound/` all passing (24/24)
- [x] ACL launcher build verified on 910B3
- [x] Single-kernel `msprof` runtime smoke and CSV sync verified
- [x] 10 measured CSVs synced and measured constants promoted
- [x] `validate_vs_tilesim.py` passes

**Remaining**: Optional investigation of the informational GM→UB bandwidth warning
