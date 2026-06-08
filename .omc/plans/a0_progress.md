# A.0 Implementation Progress

**Last updated**: 2026-06-06

## Architecture

```
Triton kernel (.py) → bishengir-compile → .npuir.mlir (HIVM IR)
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │ C++ MLIR Passes         │                         │
                    │ HIVMAnalysis            │ PipelineAnalysis        │
                    │   emitDESGraph() JSON   │   emitDependencyGraph() │
                    │                         │   JSON                  │
                    └─────────┬───────────────┴──────────┬──────────────┘
                              │                          │
                              ▼                          ▼
                    ┌─────────────────────────────────────────┐
                    │ Python perfbound/ package               │
                    │                                         │
                    │ M3: hivm_extractor.py (JSON → OpRecord) │
                    │ M4: component_model.py (harmonic I_c)   │
                    │     grid_model.py (T_grid_floor)        │
                    │     serialization.py (mand/avoid split) │
                    │ M5: bound_combiner.py (T_bound + attr)  │
                    │                                         │
                    │ Output: T_bound, binding tier, 5 gaps   │
                    └─────────────────────────────────────────┘
```

## Completed

### C++ Emitter Layer (committed)

| Commit | Description | Δ |
|--------|-------------|---|
| `5cfae79` | HIVMOp: +`srcSpace`, `dstSpace`, `elemType` via MLIR APIs. JSON emit in `emitDESGraph()` and `emitPerfettoTrace()`. New `PipelineScheduler::emitDependencyGraphJSON()`. | +114 |
| `a4d89d2` | Wire `emitDependencyGraphJSON()` into `PipelineAnalysisPass` → `pipeline_dep_graph.json`. | +15 |

### Python `perfbound/` Package (uncommitted, 23 files)

| Module | Files | Status |
|--------|-------|--------|
| **M1** Calibration | `calibration/constants.py`, `calib_loader.py` | `CalibrationDB`, `CalibrationConstant` (value±CI, source, n_runs), `CoreConfig`, `CubeConfig`, `VectorConfig`, `MemHierarchy`, `MemBandwidth` — full JSON ser/de |
| **M2** DSL Extractor | `extract/dsl_extractor.py`, `grid_idioms.py` | `GridInfo` dataclass, idiom template stubs |
| **M3** HIVM Extractor | `extract/hivm_extractor.py`, `op_classifier.py`, `eligibility_oracle.py` | JSON loaders for both C++ emits, pipe→component classification, per-component `O_prec` aggregation, handoff detection |
| **M4** Models | `model/component_model.py`, `grid_model.py`, `serialization.py`, `bandwidth.py` | Weighted-harmonic `I_c = ΣO / Σ(O/P)` (Eq. 4), `T_core_floor = max_c(O_c/I_c)`, `T_grid_floor`, mandatory/avoidable handoff split |
| **M5** Combiner | `combine/bound_combiner.py`, `report.py`, `two_limit.py` | `T_bound = max(grid, core) + T_serial_irreducible`, 5-way attribution dataclass, binding tier detection |
| **M6** Validation | `validate/harness.py`, `counterfactual.py` | Stubs against `remote-bench-910b3` skill |
| **Data** | `data/bandwidth_910b3.csv` | Schema ported from tilesim, unpopulated |
| **Tests** | `tests/perfbound/test_component_model.py` | **11/11 passing** |

### Test Coverage (11/11 passing)

| Test | What it verifies |
|------|-----------------|
| `test_single_precision_harmonic_reduces_to_rate` | `I_cube = P_cube[fp16]` for single-precision work |
| `test_mte_bytes_correct` | MTE time = bytes / sustained BW to 3 sig figs |
| `test_binding_component_is_mte` | Memory-bound kernel → MTE_GM binds |
| `test_cube_to_vector_is_mandatory` | Cube→Vector cross-path classified mandatory |
| `test_same_path_handoffs_are_avoidable` | MTE_GM→Cube, Vector→MTE_UB classified avoidable |
| `test_serial_irreducible_from_calibration` | `T_serial = handoff_cycles / clock_rate` |
| `test_no_mandatory_without_calibration` | No calib → T_serial = 0 (conservative default) |
| `test_mixed_fp16_int8` | Harmonic mean weights precision mix correctly |
| `test_empty_extract_returns_zero` | No ops → T_core_floor = 0 |
| `test_scalar_only_zero_time` | Scalar accounted via overhead factor, not separate I_c |
| `test_matmul_bound_matches_golden` | Full M4+M5 pipeline — spreadsheet golden value |

## Blocked (needs remote 910B3)

| Item | Priority | Why blocked |
|------|----------|-------------|
| M1 calibration constants | **P0** | All `I_c` values currently 0.0 — `P_cube`, `P_vector`, BW curves, `mandatory_handoff_cost` need CCE microbench runs |
| End-to-end C++ JSON verification | **P0** | Run real kernel through BiShengIR build → confirm `emitDESGraph()`/`pipeline_dep_graph.json` output |
| M2 DSL symbolic execution | P1 | Recover affine tile map from TTIR `program_id` arithmetic |
| M4 bandwidth CSV population | P1 | Real 910B3 sustained BW measurements |
| M5 two-limit (`two_limit.py`) | P2 | `T_bound_HIVM` vs `T_bound_DSL` — stubbed |
| M6 harness | P2 | Remote bench integration — stubbed |

## Key Design Decisions

- **C++ emits, Python consumes** — JSON is the seam. No MLIR re-walking in Python.
- **Conservative bound semantics** — soundness (100% `T_bound ≤ T_measured`) enforced by sustained (not peak) `I_c` and err-toward-avoidable serialization split.
- **Separate calibration provenance** — every constant carries CI, source, and run count; no datasheet peaks in `I_c`.
- **Pure analytical models** — M4 functions have zero I/O, zero compilation, zero search. `perfbound/` is independent of the C++ build.

---

## Review Log

### 2026-06-06 — Review Pass 2 (post-gap-fix refresh)

**Reviewer verdict: A.0 scope met, no over-implementation. 11/11 tests passing.**

Three issues from Review Pass 1 were addressed:

| Fix | File | Detail |
|-----|------|--------|
| Unit conversion soundness | `calibration/constants.py` | `* 1024.0 → * 1000.0` in `bw_bytes_per_us`. 1 GB/s = 10³ B/μs (SI), not 1024. Old value understated T_MTE by ~2.4%, risking unsound bound. Golden values updated: 2.133 μs → 2.185 μs. |
| `OpRecord` missing Gap 4 fields | `extract/hivm_extractor.py` | Added `repeat: int = 1` and `mask: int = 0` to `OpRecord`. Defaults are conservative (no gap). Populated from C++ JSON via `node.get()`. |
| Gap 1/2/4 never wired | `combine/bound_combiner.py` | Added `_compute_gap1`, `_compute_gap2`, `_compute_gap4`, `_wire_gaps`. `combine()` now accepts optional `extract: HIVMExtract` and `calibration: dict`. Attribution remains diagnostic; `T_bound` is unchanged. |

**Remaining issues noted (non-blocking for A.0):**

- `_compute_gap4` uses `op.elements` for Cube work share — falls silent when `elements == 0` (C++ emitter currently emits `flops`, not `elements` for Cube ops). Fix: use `op.flops` as secondary source. Address before first real HIVM JSON is loaded.
- Gap 4 will always be 0.0 until C++ emitter starts populating `repeat`/`mask` fields in `emitDESGraph()`. Conservative and acceptable.
- `_compute_gap2` MTE_L1 path (`l1→l0a`) silently skips if BW key absent — conservative, acceptable.
- No dedicated test for `_wire_gaps` end-to-end. Suggested: add one test with a Cube→Vector extract and a mis-placed i32 compare op asserting `gap1_wrong_unit_us > 0`.

### 2026-06-05 — Review Pass 1 (initial)

Three gaps identified vs A.0 plan: unit bug, missing `repeat`/`mask`, unconnected gap helpers. All addressed in Pass 2.
