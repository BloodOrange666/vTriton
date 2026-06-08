# Open Questions

## A.0 Two-Tier Performance Bound Model - 2026-06-05

### Resolved (Day-1 Spike)

- [x] **Day-1 spike: where are the compile-only TTIR and HIVM dumps written, and what format?**
  - HIVM: `triton_dsl_dump_launcher.py` intercepts `bishengir-compile`, runs secondary invocation with `--bishengir-print-ir-after=hivm-inject-sync`, normalizes via `bishengir-opt --mlir-print-op-generic` → `kernel_NNN.npuir.mlir`
  - TTIR: lands alongside in the dump dir as `.ttir` files
  - Arg bindings: `tritonsim_hivm_bindings.jsonl` captures `argN=value` pairs
  - Source: commits `cc395a5` and `4cb637e`

- [x] **Do the existing HIVM/TTIR compile-only dumps already carry per-op precision, transfer-size/alignment, and unit-assignment metadata?**
  - HIVM: `emitDESGraph()` JSON already emitted: id, name, pipe, duration, line, depends_on, is_sync, is_barrier, event_id, event_generation, sender/receiver_pipe, core_type, bytes, elements, loop_multiplier, multi_buffer_slots, read/write_buffers. **Missing** srcSpace/dstSpace/elemType → **now added** (see below).
  - `HIVMOp` struct now carries `srcSpace`, `dstSpace`, `elemType` fields extracted via MLIR Type APIs (BiShengIR path) and standard MLIR APIs (elemType always available).
  - PipelineAnalysis: `emitDependencyGraphJSON()` added to `PipelineScheduler` — emits per-op id, hw_unit, op_name, bytes, flops, duration, loop_multiplier, depends_on, start/end_cycle.

- [x] **Can `PipelineAnalysis::DependencyGraph` and `HIVMAnalysis` cheaply emit JSON?**
  - HIVMAnalysis: `emitDESGraph()` already existed; extended with 3 new fields.
  - PipelineAnalysis: `emitDependencyGraphJSON()` added (new method on `PipelineScheduler`).
  - Both compile clean. Python consumers can `json.load()` the output directly.

### A.1 Remediation (2026-06-07) — CLOSED 2026-06-07

- [x] **Fix 1**: `MteGmUbKernel` now takes `(startIter, iterCount)`; 512 MB buffer allocated; warmup launch (0–767, un-profiled) + measured launch (768–2047) wired in `cce_remote_bench.py:514–516`. `fit_constants.py` uses `N_iter=1280`.
- [x] **Fix 2**: `mandatory_handoff.cce` uses `KERNEL_TYPE_MIX_AIC_1_1`; `MandatoryHandoffKernel::Process()` branches on `ASCEND_IS_AIC`/`ASCEND_IS_AIV` with `SyncAll<false>()`; K-sweep `{128,256,384,512,1024,2048}` → kTiles `{1,2,3,4,8,16}`.
- [x] **Fix 3**: `validate_vs_tilesim.py:60` — `BW_910B4_GM_UB_GBPS = 64.36`.
- [x] **Fix 4**: `fit_constants.py:250` — `sorted_durations[:2*n//3]`; `n_runs=len(durations)`.
- [x] **Language migration**: CCE → AscendC (`kernel_operator.h`) applied throughout.

**Remaining non-blocking items before 910B3 re-run**:
- [ ] Verify `DEFAULT_SOC_VERSION` in `cce_remote_bench.py:32` — currently `"Ascend910B1"`, should be `"Ascend910B3"` if that string is valid in CANN toolkit on the remote.
- [ ] Dead code cleanup: `extract_mandatory_handoff` `has_vector_consumer` branch in `fit_constants.py:302–307` is now unreachable (mixed kernel emits one row, not two).

### Open Item — L2 Cache Bandwidth Model (deferred, A.5/A.7)

- [ ] **L2 hit rate model**: Real kernels may see partial L2 reuse. Using pure HBM BW is conservative (correct for soundness) but makes the bound loose when working set fits in L2.
  - Model: `BW_eff = h × BW_l2 + (1-h) × BW_hbm` where `h = f(working_set / L2_size)`
  - Requires: `BW_l2_sustained` measured separately (small-buffer bench, repurposed)
  - Adds `predict_l2_hit_rate()` to `perfbound/model/bandwidth.py`
  - Deferred until A.1 HBM measurements are clean (Fix 1 is prerequisite)
  - See `a1_remediation.md §Open Item`

### Still Open (pre-A.1)

- [x] **CLOSED** 910B3 sustained-rate calibration values — all 16 P0 constants measured (2026-06-07): Cube ~5.16 TFLOPS/core, BW GM→UB ~87 GB/s, handoff 7621 ± 82 cycles, n=45 each, all CI < 2.5%. `calib_910b3_v1.json` committed.
- [x] **CLOSED** Is BiShengIR exposed via MLIR Python bindings? — Resolved by architectural decision: C++ `emitDESGraph` JSON is the seam (preferred path). MLIR Python bindings not needed.
- [x] **CLOSED** Exact remote 910B3 microbench harness — AscendC harness exists (`run_benchmarks.sh` + `fit_constants.py` + 10 kernels). CCE path superseded by AscendC migration (2026-06-07). Pending 4 fixes before re-run.
- [x] **CLOSED** `mandatory_handoff_cost` measurement method — K-sweep linear-fit method fully designed (A.1 plan AC-3 + `a1_remediation.md §Fix2`). Kernel implementation is Fix 2 (pending, not an open question).
- [ ] Validation kernel suite selection (Groups I–V, ≥30 kernels) — which real LLM-workload kernels are available to compile via bishengir on the 910B3 remote.
- [ ] End-to-end dump verification: produce a real `.npuir.mlir` + `emitDESGraph` JSON on a known kernel via the Triton DSL → bishengir-compile pipeline. **Assigned to A.3** for local/compile-only closure via `tritonsim-hivm --npuir-file ... --des-graph-file ...`; remote validation remains A.6.

### A.2 Closure — CLOSED 2026-06-08

- [x] **CLOSED** DSL Extractor regex fragility — replaced all 4 `_parse_*()` regex functions with C++ `ExtractTTIRInfoPass` (`--extract-ttir-info`) that walks the MLIR AST via `triton::GetProgramIdOp`, `scf::ForOp`, `triton::MakeTensorPtrOp`. No text matching anywhere in `perfbound/extract/`.
- [x] **CLOSED** Persistent kernel occupancy/load_balance wrong — `_persistent_kernel_info()` now implements correct round-robin tile assignment; `occupancy = n_active / n_programs`; `load_balance = mean(active) / max(active)`.
- [x] **CLOSED** `configs/ascend_910b3.json` not wired into M2 — `grid_idioms.py` path depth fixed (`parents[3]` → `parents[2]`); config loading verified by `test_config_loading`.
- [x] **CLOSED** 10-reference-kernel acceptance criterion — K1–K10 parametrized suite covers 1D, 2D, persistent, UB violation, and over-subscription; all hand-derived `occupancy`/`load_balance`/`tile_assignment` values verified.

### A.3 Closure Targets (assigned 2026-06-08)

- [ ] Close DES graph schema compatibility: canonicalize the C++/Python contract around `operations`, required per-op fields, and schema version.
- [ ] Close deterministic extractor artifact handling: remove or option-gate hard-coded `pipeline_dep_graph.json`/trace writes so tests use `tmp_path`.
- [ ] Close Tier 2 metadata completeness: prove `O_prec`, transfer size/alignment, unit assignment, repeat/mask defaults, and handoff records are populated from real DES graph JSON.
- [ ] Close semantic eligibility oracle gap: connect TTIR/Linalg semantics to realized HIVM assignment and flag the seeded i32 compare Scalar fallback.
