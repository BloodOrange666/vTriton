# Performance Bound Model — Mainline Progress

**Project**: Two-Tier Analytical Performance Bound Model for Triton/Ascend NPU  
**Spec**: `.omc/specs/performance_bound_model.md` + `.omc/specs/implementation_and_paper_plan.md`  
**Plan**: `.omc/plans/a0_perf_bound_model.md`  
**Detail log**: `.omc/plans/a0_progress.md`

---

## Stage Map (A.0 → A.8 + Part B)

| Stage | Scope | Timeline | Status |
|-------|-------|----------|--------|
| **A.0** | Python `perfbound/` package scaffold (M1–M6 stubs + models) | Wk 1–7 | ✅ **Complete** |
| **A.1** | M1 Calibration — AscendC microbench suite on 910B3 | Wk 1–2 | ✅ **Complete** — 16 P0 constants measured + wired into model (40/40 tests) |
| **A.2** | M2 DSL Extractor — symbolic affine recovery from TTIR | Wk 2–3 | ✅ **Complete** — C++ MLIR pass + 10 reference kernels verified (63/63 tests) |
| **A.3** | M3 HIVM Extractor — C++ JSON round-trip verified | Wk 3–5 | 🔶 Partial (Python loaders done; C++ JSON unverified) |
| **A.4** | M4 Models — calibrated `I_c` values populated | Wk 5–7 | 🔶 Partial (formulas done; `I_c` now loadable from A.1 data) |
| **A.5** | M5 Combiner — Gap 1/2/4 wired, Gap 3 from real handoffs | Wk 7 | ✅ Code complete (untested on real data) |
| **A.6** | M6 Validation Harness — remote-bench-910b3 integration | Wk 6–9 | ⛔ Stub (hardware-gated) |
| **A.7** | Two-limit computation (`T_bound_HIVM` vs `T_bound_DSL`) | Wk 8 | ⛔ Stub (deferred) |
| **A.8** | End-to-end pipeline verified on ≥1 real kernel | Wk 9 | ⛔ Not started |
| **Part B** | Experiments, paper writing, iterate calibration | Wk 9–12 | ⛔ Not started |

---

## Current State (2026-06-08)

### What's done

- **`perfbound/` package** (23 files, pure Python, zero MLIR dependency)
  - M1: `CalibrationDB`, `CalibrationConstant` (value ± CI, source, n_runs), full JSON ser/de
  - M2: `GridInfo`, `DSLExtractor` (C++ MLIR pass + affine idiom recovery), `grid_idioms.py` (1D/2D templates), `mlir_parser.py` (subprocess wrapper), 10 reference kernels verified
  - M3: JSON loaders for both C++ emits, `OpRecord` + `HandoffRecord`, pipe→component classification, eligibility oracle, `repeat`/`mask` fields for Gap 4
  - M4: Weighted-harmonic `I_c = ΣO / Σ(O/P)` (Eq. 4), `T_core_floor`, `T_grid_floor`, mandatory/avoidable handoff split
  - M5: `T_bound = max(T_grid_floor, T_core_floor) + T_serial_irreducible`, 5-way attribution, `_wire_gaps` connecting Gap 1/2/4 helpers, binding tier detection, text+JSON report
  - M6: Stubs with correct `NotImplementedError` (hardware-gated)
- **C++ emitter layer** (committed): `emitDESGraph()` JSON, `emitDependencyGraphJSON()`, `PipelineAnalysisPass` wiring
- **A.1 calibration** (`calib_910b3_v1.json`): 16 P0 constants, n=45 each, all CI < 2.5%
  - Cube FP16/INT8/BF16: ~5.16 TFLOPS/core
  - BW GM→UB / UB→GM: ~87 GB/s; GM→L1: ~141 GB/s; L1→L0A: ~452 GB/s
  - Vector add/mul/max/min: 14.6–16.2 GFLOPS; transcendentals: 3.3 GFLOPS
  - Mandatory handoff cost: 7621 ± 82 cycles (~4.1 µs at 1.85 GHz)
- **Tests**: 63/63 passing

### What's blocked / deferred

| Item | Blocker | Priority |
|------|---------|----------|
| End-to-end C++ JSON → Python round-trip | Need built `tritonsim-opt` + real kernel | P0 |
| `_compute_gap4` Cube fallback (`elements == 0`) | Minor fix before real HIVM data lands | P1 |
| Gap 4 wire through (C++ emitter `repeat`/`mask`) | C++ emitter doesn't emit these fields yet | P1 |
| ~~`configs/ascend_910b3.json`~~ | ~~Clone from `ascend_910b.json`, wire into M2 capacity checks~~ | ~~P1~~ ✅ Closed by A.2 |
| ~~M2 buffer capacity checks~~ | ~~Hardcoded 256 KB/1 MB — should read from config JSON~~ | ~~P1~~ ✅ Closed by A.2 |
| Two-limit computation (`two_limit.py`) | Deferred to A.7 (Wk 8) | P2 |
| M6 validation harness | remote-bench-910b3 skill integration | P2 |

---

## Completed: A.2 — DSL Extractor (2026-06-08)

| Step | Status |
|------|--------|
| C++ `ExtractTTIRInfoPass` (`--extract-ttir-info`) | ✅ Done — walks AST, emits JSON (grid_axes, persistent_loops, tensor_ptr_shapes, has_dot) |
| `perfbound/extract/mlir_parser.py` | ✅ Done — subprocess wrapper, brace-counted JSON extraction |
| `dsl_extractor.py` refactor (no regex) | ✅ Done — `_extract_from_ttir()` uses `parse_ttir()`, `_persistent_kernel_info()` round-robin |
| `grid_idioms.py` Bug 1 fix | ✅ Done — `parents[2]` correct, `ascend_910b3.json` loads |
| 10 reference kernel parametrized suite | ✅ Done — K1–K10, occupancy/lb/tile_assignment/buffer_ok verified |
| Code review (7 findings) | ✅ All resolved — F1 NameError, F2 pytestmark, F3 paths, F4 C++ filter, F5 inline TTIR, F6 ifdef, F7 dead elif |

---

## Next Milestone: A.3 — HIVM Extractor (C++ JSON round-trip)

**A.3 scope**: Verify the C++ `emitDESGraph()` + `emitDependencyGraphJSON()` JSON round-trip end-to-end on a real kernel. Python loaders exist (`hivm_extractor.py`, `OpRecord`, `HandoffRecord`) but the C++ JSON output has not been validated against the Python schema.

**Closure targets** (from `open-questions.md §A.3`):
- [ ] DES graph schema compatibility — canonicalize C++/Python contract, required per-op fields, schema version
- [ ] Deterministic artifact handling — remove hard-coded `pipeline_dep_graph.json`/trace writes; use `tmp_path` in tests
- [ ] Tier 2 metadata completeness — `O_prec`, transfer size/alignment, unit assignment, repeat/mask defaults populated from real JSON
- [ ] Semantic eligibility oracle — connect TTIR/Linalg semantics to realized HIVM unit assignment; flag Scalar fallback

**Open item — L2 cache BW model** (deferred to A.5/A.7):
- Real kernels have partial L2 reuse; pure HBM BW is conservative (sound) but loose for tight bounds
- Future: `BW_eff = h × BW_l2 + (1-h) × BW_hbm` where `h = f(working_set / L2_size)`
- Tracked in `open-questions.md §Open Item — L2 Cache Bandwidth Model`

---

## Formula Reference

```
T_bound = max(T_grid_floor, T_core_floor) + T_serial_irreducible

T_grid_floor  = (total_tiles / num_cores) * max_tile_work / avg_tile_work
T_core_floor  = max_c(O_c / I_c)
I_c           = ΣO_c / Σ(O_k / P_k)   [weighted harmonic mean over ops in component c]
T_serial_irr  = Σ mandatory_handoff_cost(type)

Five-way attribution (diagnostic, not part of bound):
  grid         = T_grid_floor / T_bound
  gap1         = wrong-unit placement (eligible set vs realized unit)
  gap2         = coalescing loss (actual pkt BW vs ideal large-pkt BW)
  gap3         = avoidable serialization (non-mandatory handoffs)
  gap4         = intra-unit exec inefficiency (repeat > 1, masked lanes)
```
