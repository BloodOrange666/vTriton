# A.0 Implementation Plan — Two-Tier Analytical Performance Bound Model

**Target**: Ascend 910B3 NPU, Triton kernels.
**Repo**: build everything in `/mnt/d/work/git/vTriton` (the BASE repo).
**Source spec**: `/mnt/d/work/git/vTriton/.omc/specs/performance_bound_model.md`
and `/mnt/d/work/git/vTriton/.omc/specs/implementation_and_paper_plan.md`
(Part A, §A.0–A.8). This plan grounds that spec into concrete file paths,
the Python/C++ seam, and selective tilesim extraction. Module numbering
(M1–M6) and sequencing (A.1→A.8) mirror the spec exactly — do not invent a
different taxonomy.

---

## 1. Requirements Summary

Build a system that, given a Triton kernel (`.py`) + problem shape, computes a
**provably-conservative lower bound on time** (= upper bound on performance),
with no search and no profiling-in-the-loop. Measurement enters in exactly two
places: calibrating hardware constants once (M1), and validating the finished
bound on a few kernels (M6).

Six modules in dependency order (spec §A.0):
`Calibration (M1) → DSL extractor (M2) → HIVM extractor (M3) →
two analytical models (M4) → bound combiner + attribution (M5)`,
with `validation harness (M6)` built in parallel but NOT part of the model.

The bound:
```
T_bound = max(T_grid_floor, T_core_floor) + T_serial_irreducible
```
- **Tier 1 (grid)**, from the DSL: occupancy, load balance, redundancy(grid)
  (default = 1), busiest core.
- **Tier 2 (component)**, from one core's HIVM: per-component ideal rate
  `I_c` (Eq. 4 weighted-harmonic mean), `T_core_floor = max_c(O_c / I_c)`,
  plus mandatory-only `T_serial_irreducible`.
- **Five-way attribution** (a separate axis from the six components): grid +
  Gap1 (wrong-unit placement), Gap2 (coalescing/MTE-E), Gap3 (avoidable
  serialization/MTE-R), Gap4 (intra-unit execution/compute-E).
- **Two limits**: `T_bound_HIVM` (hardware-reachable) vs `T_bound_DSL`
  (compiler-reachable); their gap attributes headroom to compiler vs author.

### Conservative-bound acceptance semantics (load-bearing)
Acceptance is NOT "predicted matches measured within N%". For a conservative
bound it is two distinct properties, validated separately in M6:
- **(a) Soundness**: `T_bound ≤ T_measured` on 100% of kernels. Any violation
  is a model bug (too-high `I_c`, or a mandatory handoff misclassified as
  avoidable). This is binary, not a tolerance.
- **(b) Tightness**: median `T_measured / T_bound < 1.20` on *optimized*
  kernels. Loose bounds on *unoptimized* kernels are correct (that is the
  headroom the model exists to surface) and are NOT failures.

Numeric `±N%` tolerances apply only to *internal reconciliation* checks
(e.g. `Σ O_prec` vs analytic flop count within 2%) and to *attribution
quantification* (predicted gap vs measured gap-removal within 20%), never to
the headline bound-vs-measured comparison.

---

## 2. Architecture Decision (Python vs C++, new dirs)

### The seam (apply identically across all modules)
**C++ MLIR passes EMIT structured IR dumps; Python extractors + models CONSUME
them and reuse tilesim.**

This matches the repo's existing direction: recent commits already dump TTIR
and HIVM IR in compile-only mode (`Enable HIVM IR dump from Triton DSL`,
`Share Triton DSL dump capture across TTIR and HIVM`). We build on that
capture path rather than fighting it.

Rules:
- **Do NOT** extend the C++ `lib/AscendModel/Analysis/RooflineAnalysis.cpp`
  into component separation. It stays the legacy single-level roofline
  (kept for the Exp 4b "sustained roofline" baseline). The new component
  model is Python.
- **Do NOT** rewrite tilesim's 100+ op models or its cost-model classes in
  C++. Port the specific constants, the empirical bandwidth curves, and the
  op→component/precision classifiers into Python.
- C++ touches are limited to: (i) ensuring the TTIR and HIVM dumps carry the
  metadata the Python extractors need (op list, precisions, transfer sizes,
  unit assignment, program_id arithmetic), and (ii) optionally a small
  metadata-annotation pass. No new heavy C++ analysis.

| Module | Language | Why |
|--------|----------|-----|
| M1 Calibration DB | Python (+ AscendC/hand-HIVM microbench kernels) | JSON authoring, curve fitting, remote-bench orchestration |
| M2 DSL extractor | Python | Triton AST / TTIR parse, affine recovery, symbolic execution |
| M3 HIVM extractor | Python (consumes C++ HIVM dump) | walk `.npuir.mlir` via MLIR Python bindings; classifier |
| M4 grid + component models | Python (pure functions) | analytical math, no I/O |
| M5 combiner + attribution | Python | report generation, two-limit |
| M6 validation harness | Python (drives `remote-bench-910b3`) | compile + msprof; NOT in model |
| (C++) dump-metadata pass | C++ (only if dumps lack a field) | emit per-op precision/transfer/unit tags |

### New directory layout (all under `/mnt/d/work/git/vTriton`)
```
perfbound/                         ← new Python package, top-level
  __init__.py
  calibration/                     ← M1
    __init__.py
    constants.py                   ← dataclasses for P_cube, P_vector, BW...
    calib_loader.py                ← load/validate calib_910b3_vX.json
    microbench/                    ← CCE micro-benchmark kernels (CCE gives
      gemm_sustained.cce           ←   direct per-component control: Cube,
      vector_sweep.cce             ←   Vector, MTE-GM etc. independently)
      dma_sweep.cce
      hbm_allcore.cce
      handoff_min.cce
      run_bench.py                 ← driver: compile CCE → run → parse msprof
    fit_curves.py                  ← bandwidth/alignment/amortization fits
  extract/
    dsl_extractor.py               ← M2 (Tier 1 input)
    grid_idioms.py                 ← M2 idiom templates (1D/2D/persistent)
    hivm_extractor.py              ← M3 (Tier 2 input)
    eligibility_oracle.py          ← M3 semantic eligibility (TTIR/Linalg)
    op_classifier.py               ← M3 op→(component, precision), ported
  model/
    grid_model.py                  ← M4 Tier 1 analytical
    component_model.py             ← M4 Tier 2 analytical (Eq. 4 harmonic)
    serialization.py               ← M4 mandatory/avoidable split
    bandwidth.py                   ← ported tilesim sustained-rate lookups
  combine/
    bound_combiner.py              ← M5 T_bound + binding + 5-way attribution
    two_limit.py                   ← M5/A.7 T_bound_HIVM vs T_bound_DSL
    report.py                      ← per-kernel report (text + JSON)
  validate/
    harness.py                     ← M6 remote-bench driver + soundness check
    counterfactual.py              ← M6 hand-HIVM counterfactual runner
  data/
    calib_910b3_v1.json            ← calibration output (versioned)
    bandwidth_910b3.csv            ← ported/measured sustained-rate curves
tests/perfbound/                   ← pytest suite (golden-number tests)
configs/ascend_910b3.json          ← NEW: 910B3 config (see §6 risks)
```
Rationale for a separate top-level `perfbound/` package (not inside `lib/`):
the analytical model is Python and independent of the C++ MLIR build; keeping
it out of `lib/` avoids coupling its test/iteration loop to `ninja` rebuilds.

---

## 3. Module-by-Module Plan (M1–M6)

### M1 — Calibration Database (spec §A.1, Wk 1–2)
**Files**: `perfbound/calibration/*`, output `perfbound/data/calib_910b3_v1.json`.

Build the versioned JSON of sustained hardware constants for one 910B3.
Each constant carries `{value, ci, source, n_runs}`. Microbenchmarks are
written in **CCE** (Compute Core Engine), which gives direct per-component
control (Cube, Vector, MTE-GM, etc. independently) — essential for isolating
each constant without cross-component interference. Constants and priority
follow spec §A.1 table exactly:

| Constant | P0/P1/P2 | Microbench |
|----------|----------|------------|
| `P_cube[fp16,int8]` sustained | P0 | large square GEMM steady-state |
| `P_vector[add,mul,exp,tanh,gelu,rsqrt]` sustained | P0 | in-UB element-wise sweep |
| `BW[gm→l1,gm→ub,l1→l0,ub→gm,l0c→gm]` sustained | P0 | aligned DMA sweep |
| `BW_hbm_sustained` (all 20 cores under load) | P0 | 20-core simultaneous read |
| `mandatory_handoff_cost` (L0C→GM + GM→UB min) | P0 | minimal MatMul→Vector |
| `P_scalar` | P1 | i32 cmp+branch loop |
| `η_alignment(stride)`, `η_amortization(size)` | P1 | alignment / size sweep |
| `L2_residency_bytes` | P2 deferred | reuse-distance probe |

**Starting point**: `configs/ascend_910b.json` already holds candidate
constants AND a `calibration` block measured from flash-attention profiling
(vector_startup=35, mte2_startup=50, exp=4, pipe_barrier=7500 cycles/iter,
etc.). These are the seed; §6 lists which are datasheet peaks needing
microbench replacement.

**Acceptance** (spec §A.1): every P0 constant measured with <5% run-to-run
variance (≥30 runs each); `mandatory_handoff_cost` separable from K-scaling
(vary K, confirm the intercept is the handoff).

### M2 — DSL Extractor / Tier 1 input (spec §A.2, Wk 2–3)
**Files**: `perfbound/extract/dsl_extractor.py`, `grid_idioms.py`.

Parse the `@triton.jit` function + launch grid + shape → Tier-1 quantities:
`G`, `tile_assignment[p]`, `occupancy = min(G,n_cores)/n_cores`, `work[p]`,
`load_balance = mean(work)/max(work)`, `redundancy(grid)` (default 1).

Method: recover the affine map from `tl.program_id` → tile via TTIR
(`tt.get_program_id`, `tt.load` pointer arithmetic) using symbolic execution
(do NOT run the kernel). Implement common idioms as templates first (1D
row-block, 2D tile, persistent/grouped), general affine recovery second.

**Hardware-legality constraints are first-class** (spec §A.2): record and
enforce L0A/L0B/L0C/L1/UB capacity, register/buffer pressure, integer/tile
divisibility. A tiling the affine map permits but that overflows a buffer is
illegal and must NOT enter any bound. Capacity values come from
`configs/ascend_910b3.json` (`local_mem`).

`n_cores`: 20 for Cube-bearing kernels (each AIC drags 2 AIV), 40 for
Vector-only kernels (spec §1.1).

**Acceptance** (spec §A.2): on 10 reference kernels, recovered
`tile_assignment` matches a hand-derived map; `occupancy` and `load_balance`
match manual calculation; buffer-capacity-violating tilings rejected.

### M3 — HIVM Extractor / Tier 2 input (spec §A.3, Wk 3–5)
**Files**: `perfbound/extract/hivm_extractor.py`, `eligibility_oracle.py`,
`op_classifier.py`.

Consume the C++-emitted structural JSON (preferred, per §4b — extend
`HIVMAnalysis`/`PipelineAnalysis` to dump it) of one core's program (the
busiest, per Tier 1); fall back to walking the HIVM dump (`.npuir.mlir`) via
MLIR Python bindings only if the C++ emit path is unavailable. Reuse the
existing pipe classification and barrier/sync detection (§4b) rather than
re-deriving them. Extract per-component:
- `O_prec[component]` — op/byte counts per precision/transfer-type
- `transfer_size[mte]`, `transfer_alignment[mte]` (Gap 2)
- realized `unit_assignment[op]` (Gap 1)
- `repeat`/`mask`/SIMD-lane params per compute op (Gap 4)
- handoff list with producer/consumer components (serialization split)

Components (spec §2.1): `Cube, Vector, Scalar` (compute);
`MTE-GM (GM→{L1,L0A/B,UB}), MTE-L1 (L1→L0A/B), MTE-UB (UB→{OUT,L1})`.

**Eligibility oracle** (Gap 1 input): from TTIR/Linalg, the set of units each
op *could* run on (matmul+FP16/INT8 → Cube; element-wise/reduction → Vector;
type-incompatible → Scalar). Gap 1 = diff(eligibility, realized HIVM).

**Acceptance** (spec §A.3): `Σ O_prec` reconciles with the kernel's analytic
flop/byte count within 2%; the eligibility oracle correctly flags a
deliberately seeded i32-compare Scalar fallback.

### M4 — The Two Analytical Models (spec §A.4, Wk 5–7)
**Files**: `perfbound/model/grid_model.py`, `component_model.py`,
`serialization.py`, `bandwidth.py`. Pure functions, no I/O, no compilation.

- **Grid model** (Tier 1): Module 2 output + calibration → `T_grid_floor`,
  `busiest_core_id`. Formula spec §1.4:
  `T_grid_floor = T_total_work / (n_cores·occupancy·load_balance·I_binding)`,
  with `bytes_in` scaled by `redundancy(grid)` (=1 by default) in `I_binding`.
- **Component model** (Tier 2): Module 3 output + calibration → `I_c` per
  component via Eq. 4 weighted-harmonic mean
  `I_comp = Σ O_prec / Σ (O_prec / P_prec)`, then
  `T_core_floor = max_c(O_c / I_c)`.
- **Serialization split** (`serialization.py`): for each handoff classify
  mandatory (consumer needs producer's data AND producer/consumer on
  different components exchanging only via memory, i.e. Cube↔Vector through
  GM/L2) vs avoidable. Sum mandatory minimum costs into
  `T_serial_irreducible`. **The split MUST err toward "avoidable"** — a
  non-mandatory handoff wrongly counted would overstate time and break the
  bound (spec §4.0).

**Acceptance** (spec §A.4): on a hand-computed kernel, every intermediate
(`I_c`, `T_core_floor`, `T_serial_irreducible`) matches a spreadsheet to
3 significant figures.

### M5 — Bound Combiner & Attribution (spec §A.5, Wk 7)
**Files**: `perfbound/combine/bound_combiner.py`, `report.py`,
`two_limit.py`.

```
T_bound = max(T_grid_floor, T_core_floor) + T_serial_irreducible
binding = argmax(grid_floor vs each component) + which tier
gaps = {gap1, gap2, gap3, gap4} from Module 3 deltas
```
Composition is **max** (two independent lower bounds on the same wall-clock
time), with `+ T_serial_irreducible` attaching to the Tier-2 term (spec §4.0).

**Five-way attribution** (the separate attribution axis, spec §3 / §4.2),
distinct from the six roofline components:
1. **grid** (Tier 1, the fifth/higher axis) — realized grid worse than the
   optimal partition the bound assumes
2. **Gap 1** wrong-unit placement (R-axis, placement) — eligibility vs realized
3. **Gap 2** coalescing / transfer efficiency (E-axis, MTE) —
   `E_MTE = bytes / (T_MTE · BW_peak)`
4. **Gap 3** avoidable inter-unit serialization (R-axis, MTE) — the avoidable
   complement of the mandatory handoffs in T_serial_irreducible
5. **Gap 4** intra-unit execution efficiency (E-axis, compute) — repeat/mask
Gap2 vs Gap4 disambiguation keys off component TYPE (MTE→Gap2, compute→Gap4),
never off the R/E value alone (spec Part D risk).

**Deliverable**: per-kernel report (text + JSON via `report.py`): bound,
binding tier/component, five-way attribution, two-limit gap, single
recommended action (fix grid / fix DSL types / merge transfers / repeat / add
ping-pong).

### M6 — Validation Harness (spec §A.6, Wk 6–9, parallel, NOT in model)
**Files**: `perfbound/validate/harness.py`, `counterfactual.py`.

The only place compilation + execution happen; the model never calls this.
Drives the `remote-bench-910b3` skill (see §7). Per validation kernel:
compile via bishengir, run with output verification vs reference (correctness
first), profile with msprof, then check **(a)** `T_bound ≥ T_measured`
(soundness — must hold) and **(b)** binding component matches prediction;
record tightness `T_measured / T_bound`.

**Counterfactual** (`counterfactual.py`): hand-edit HIVM (raise `repeat`,
insert ping-pong), recompile, verify correctness, measure delta — confirms a
gap's quantified value matches measured improvement (validates attribution,
separate from validating the bound).

**Acceptance**: soundness holds on the validation set (binary); tightness
recorded; on ≥1 seeded kernel the predicted gap matches the measured
gap-removal within 20% (spec Exp 3 target).

### A.7 — Two-Limit Computation (spec §A.7, Wk 8)
**File**: `perfbound/combine/two_limit.py`.
```
T_bound_HIVM = bound with avoidable gaps analytically relaxed to zero
               (NOT by editing), subject to register/buffer/capacity/
               divisibility constraints
T_bound_DSL  = bound over HIVM bishengir actually emits (realized structural
               constraints)
gap = (T_bound_DSL − T_bound_HIVM) → compiler headroom;
      (T_measured − T_bound_DSL)   → kernel-author headroom
```
T_bound_HIVM must be computed under hardware-legality limits (the same
constraints M2/M3 enforce), not an affine-tiling-only idealization.

---

## 4. What to Port From tilesim (specific items — extract, do NOT graft)

Source repo: `/mnt/d/work/git/tilesim`. Manual extraction into `perfbound/`.

| Port target | tilesim source | Into | Adaptation |
|-------------|----------------|------|------------|
| Sustained bandwidth curves (core_num-indexed) | `core/config/arc_config/910B1/bandwidth_910B1.csv` | `perfbound/data/bandwidth_910b3.csv` | Use as the FORMAT and 910B1 fallback; **values must be re-measured on 910B3** (§6). The `(src_mem,dst_mem,core_num,pkt_size,bandwidth)` schema is exactly the sustained-rate table the spec demands. |
| Bandwidth lookup + interpolation | `core/config/arc_spec.py` (`lookup_bw`), `core/backend/engineering_costmodel/core/aicore_costmodel.py` (`scipy.interp1d` usage) | `perfbound/model/bandwidth.py` | Reimplement `lookup_bw(src, dst, core_num)` with interp over core_num. Drop the L2-reuse / cache-hit logic (redundancy=1 default). |
| Cube time model (repeat-based) | `aicore_costmodel.py::time_cube`, `time_mte1`, `time_fixpipe`, `time_mte2_aic` | `perfbound/model/component_model.py` | Use the `ceil(M/m)·ceil(K/k)·ceil(N/n)/freq · repeat_cycles` structure for `O_cube`. Strip the `delta` fudge constants; replace peak rates with measured `P_cube`. |
| Vector op cycle table | `core/backend/engineering_costmodel/core/vector_op.py`, `arc_config/910B1/vec_cycle_910B1.csv` | `perfbound/extract/op_classifier.py` + calib | Per-op vector cost ordering. Cross-check against the existing `vector_op_cycles_per_vec_instruction` in `ascend_910b.json` (exp=4,log=3,...). |
| `DType` sizes, `MemLoc` enum, op-type classifiers | `core/common/entity.py` (referenced by arc_spec) | `perfbound/extract/op_classifier.py` | Extract only the enums + classifier maps needed for op→(component,precision). Do not import the package. |
| `pkt_param` small-packet bandwidth coefficients | `arc_config/910B1/910B1.yaml` (`pkt_param`) | `perfbound/calibration/constants.py` | Seeds `η_amortization(size)` (Gap 2). Re-measure on 910B3. |
| Op-model structure reference (NOT code) | `core/backend/engineering_costmodel/op_model/{matmul,flash_attention,softmax,layer_norm_v3,...}.py` | reference only | Use as a checklist of which ops the suite must cover; do not port the cycle-accurate bodies (they over-model for a *bound*). |
| arc_spec dataclasses pattern | `core/config/arc_spec.py` (`CoreConfig`, `CubeConfig`, `VecConfig`) | `perfbound/calibration/constants.py` | Mirror the dataclass shape for the calib JSON loader; adapt fields to 910B3. |

**Do NOT port**: the L0–L3 model-level switching (`basic_cost_api.py`
`ModelLevel`), the optimizer/tiling search (`optim/`, `optimizer_algo/`),
the cache simulator, or any op-model that runs a schedule — all of these are
estimate/search machinery, antithetical to a single-shot analytical bound.

---

## 4b. Reuse vs Rebuild — vTriton's existing C++ analysis stack

The task lists vTriton's own C++ analyses under "What exists". These overlap
directly with work M3/M4 would otherwise build fresh, and reusing them is
*more* consistent with the §2 seam ("C++ emits, Python consumes") than a
from-scratch Python re-walk of `.npuir.mlir`. Decision per file:

| Existing C++ | What it already provides (verified) | Decision |
|--------------|--------------------------------------|----------|
| `lib/AscendModel/Analysis/PipelineAnalysis.cpp` + `.h` | A real `DependencyGraph` with producer→consumer edges (`addDependency`, `getDependencies`, `reverseEdges`), `getTopologicalOrder`, `hasCycle`; `PipelineOp` carrying `hwUnit`, `bytes`, `flops`, `loopMultiplier`; per-`HWUnit` pipelines. | **REUSE as emitter.** M4's mandatory/avoidable serialization split needs exactly this producer→consumer graph crossing component boundaries. Add a JSON-emit path (op list + edges + hwUnit + bytes/flops) consumed by `perfbound/model/serialization.py`. Do NOT rebuild the dep graph in Python. |
| `lib/AscendModel/Analysis/HIVMAnalysis.cpp` (114KB) | Already classifies ops into pipes (`HIVMPipe::{MTE1,CubeMTE2,MTE3,Cube,...}`), tracks barriers/sync (`isBarrier`, `isSyncOp`, `barrierPipes`, `syncCoreType`), and pulls startup latencies, bytes, FLOPs per op. | **REUSE as emitter** for M3's per-component `O_prec`, transfer sizes, and the handoff/barrier list. Extend its dump to emit per-op `{pipe→component, precision, bytes, transfer_size, unit_assignment, is_barrier}` as JSON. Python `hivm_extractor.py` becomes a thin consumer + classifier, not an MLIR re-walker. |
| `lib/AscendModel/Transforms/EstimateCycles.cpp` | Per-op cycle estimation. | **REUSE for cross-check only.** Its cycle estimates are an *estimate*, not a bound; use them to validate the M3 op→component/precision mapping (does every op the dump tags get a component?), NOT as `I_c`. |
| `lib/AscendModel/Analysis/RooflineAnalysis.cpp` | Single-level FLOP/byte roofline. | **KEEP unchanged** as the Exp 4b "sustained roofline" baseline (§2). Do not extend into components. |

Net effect on §5: M3 and M4's "build fresh" shrinks to the *analytical bound
math* on top of C++-emitted structural data. The dependency graph, pipe/
component classification, and barrier/sync detection are reused, not rebuilt.
This is the single most important consequence of grounding the plan in the
existing stack: most of the structural extraction already exists in C++.

---

## 5. What to Build Fresh (no tilesim AND no existing-vTriton equivalent)

1. **Grid tier entirely** (M2 + grid_model) — tilesim is single-core; there is
   no occupancy / load-balance / busiest-core / redundancy modeling anywhere.
   This is the spec's central contribution and must be built from scratch.
2. **DSL extractor** (M2) — symbolic execution of `tl.program_id` → tile
   affine map, idiom templates, hardware-legality rejection. No tilesim
   analogue (tilesim consumes already-tiled ops).
3. **Mandatory vs avoidable serialization *classification*** (M4
   `serialization.py`) — the property that makes this a *bound* not an
   estimate. The producer→consumer graph itself is REUSED from
   `PipelineAnalysis::DependencyGraph` (§4b); what is built fresh is the
   per-edge mandatory-vs-avoidable predicate (cross-component + data-dependent
   via GM/L2) and the min-cost sum into `T_serial_irreducible`. tilesim and
   the existing C++ both model *realized* serialization, never this split.
4. **Weighted-harmonic component ideal `I_c`** (Eq. 4) as a *bound* (max over
   components, perfect overlap assumed) — consumes the C++-emitted `O_prec`
   (§4b) but the harmonic-mean ideal + perfect-overlap floor is fresh; tilesim
   and EstimateCycles sum/schedule component times, we take the floor.
5. **Two-limit computation** (A.7) — T_bound_HIVM vs T_bound_DSL. No analogue.
6. **Five-way attribution + single recommended action** (M5) — the U=E×R gap
   decomposition plus grid axis. tilesim has no attribution output.
7. **Soundness-checking validation harness** (M6) — wraps `remote-bench-910b3`;
   checks the inequality, not a fit.
8. **910B3 config + calibration JSON** (M1) — no 910B3 spec exists in tilesim
   (confirmed: only 910B1/910B4 present) or vTriton.

---

## 6. Calibration Provenance — `I_c` sources (no datasheet peaks)

**Constraint**: every `I_c` must come from sustained *measured* rates. The
existing `configs/ascend_910b.json` is the starting calibration point; below
is which constants are trustworthy seeds vs datasheet peaks needing microbench
replacement on 910B3. Create `configs/ascend_910b3.json` from `ascend_910b.json`
and tag each field.

| Component / constant | Current value (`ascend_910b.json`) | Status | Action |
|----------------------|-------------------------------------|--------|--------|
| Cube `P_cube[fp16]` | 320 TFLOPS | **datasheet peak — suspect** | Re-measure sustained via large-GEMM steady-state microbench (M1 P0) |
| Cube `P_cube[int8]` | 640 TFLOPS | datasheet peak | Re-measure (M1 P0) |
| Vector `P_vector` | 20 TFLOPS fp16 + per-op cycle table (exp=4,...) | **per-op table is calibrated (trust as seed)**; 20 TFLOPS aggregate is peak | Keep per-op cycles as seed; re-measure aggregate sustained (M1 P0) |
| MTE-GM `BW[gm→{l1,ub}]` | 200 GB/s (round number) | **datasheet peak — suspect** | Re-measure aligned DMA sweep; build core_num curve like 910B1 csv (M1 P0) |
| MTE-L1 `BW[l1→l0]` | 400 GB/s | datasheet peak | Re-measure (M1 P0) |
| MTE-UB / fixpipe `BW[l0c→gm,ub→gm]` | 200 GB/s | datasheet peak | Re-measure (M1 P0) |
| `BW_hbm_sustained` (20-core) | 1.6 TB/s aggregate | **datasheet peak — critical** | Re-measure 20 cores under simultaneous load (M1 P0) — this is the GM→UB bottleneck the spec says dominates large models |
| Startup latencies (vector=35, mte2=50, ...) | measured from flash-attn profiling | **trust as seed** | Carry into 910B3 calib; re-validate variance |
| `mandatory_handoff_cost` | not present | **missing — must measure** | Minimal MatMul→Vector, separate from K-scaling (M1 P0) |
| `pipe_barrier` 7500 cyc/iter | calibrated | seed for avoidable-serial (Gap 3), NOT for the bound | Keep for attribution, not for T_serial_irreducible |
| `η_alignment`, `η_amortization` | `pkt_param` in 910B1 yaml only | **missing for 910B3** | Measure (M1 P1) |
| `L2_residency_bytes` | not present | **deferred P2** | redundancy(grid)=1 until Exp 7 proves stable |

Note: **910B3 ≠ 910B1**. tilesim's 910B1 csv is a format template and rough
fallback only; the 20 AIC / 40 AIV count and clock (1.85 GHz) in
`ascend_910b.json` already differ from 910B1's 24/48. All P0 rates must be
910B3-measured.

---

## 7. Remote-Bench Integration (M6 / validation)

The `remote-bench-910b3` skill
(`/mnt/d/work/git/vTriton/.claude/skills/remote-bench-910b3/SKILL.md`)
connects to **Module 6 only** — it is the compile+profile path, never called
by the model (M1–M5). It also serves M1 (running the calibration
microbenchmarks on real hardware).

Integration contract for `perfbound/validate/harness.py`:
1. Sync local checkout → remote `910B3` and profile via
   `scripts/remote_bench.py` (default compact mode). Run from repo root.
2. Remote runs under the `tlx` conda env with CANN env sourced
   (`source /usr/local/Ascend/cann/set_env.sh`) and
   `PYTHONPATH=/home/dyq/triton-ascend/python` prepended (per SKILL guardrails).
3. msprof emits `op_summary_*.csv` under `mindstudio_profiler_output/`
   (find via `rglob`). Parse the **`Op Name`** column (not Kernel/Task Name)
   and read **`Task Duration(us)`** for the target kernel → `T_measured_us`.
4. Harness computes soundness `T_bound ≤ T_measured` and tightness
   `T_measured / T_bound`; logs both with the synced local CSV as the artifact.
5. For M1 calibration, the same path runs each microbench ≥30× and feeds
   variance/CI into `calib_910b3_v1.json`.
6. For counterfactuals (`counterfactual.py`), sync hand-edited HIVM, recompile
   via bishengir, verify output equivalence, then re-profile through the same
   path — used only on the clean-compile + output-verified subset.

Guardrail: treat the local repo as source of truth; remote tree is replaced on
sync. Stop and report if remote/CANN/conda activation fails (no silent
fallback to system Python).

---

## 8. Sequencing / Timeline (mirrors spec §A.8)

```
Wk 1–2   M1 Calibration DB + microbench suite      (perfbound/calibration/)
Wk 2–3   M2 DSL / grid extractor                   (perfbound/extract/dsl_*)
Wk 3–5   M3 HIVM extractor + eligibility oracle     (perfbound/extract/hivm_*)
Wk 5–7   M4 grid + component analytical models      (perfbound/model/)
Wk 7     M5 combiner + five-way attribution         (perfbound/combine/)
Wk 6–9   M6 validation harness (parallel)           (perfbound/validate/)
Wk 8     A.7 two-limit (T_bound_HIVM vs T_bound_DSL)(perfbound/combine/two_limit.py)
Wk 9–12  Part B experiments; recalibrate where bound is loose
```

Hard dependency chain (cannot reorder):
`M1 → M2 → M3 → M4 → M5`. M6 runs parallel from Wk 6 (needs M5 stubs for the
inequality but can build the remote-bench plumbing earlier against M1).

**Day-1 spike (before M2/M3 begin)**: locate the actual dump-emission code and
a sample artifact. M2 (TTIR `tt.get_program_id` + pointer arithmetic) and M3
(HIVM `.npuir.mlir`) both rest on an unverified input contract. Concretely:
(i) find where the compile-only TTIR and HIVM dumps are written (the recent
`Share Triton DSL dump capture` / `Enable HIVM IR dump` commits), (ii) produce
one sample dump for a known kernel, (iii) confirm whether
`PipelineAnalysis`/`HIVMAnalysis` already emit (or can cheaply emit) the JSON
M3/M4 need per §4b. This spike de-risks the whole chain and decides how much
of the C++ emitter work is required.

Critical-path note: the §4b C++ JSON-emit paths (dependency graph from
PipelineAnalysis; per-op component/precision/transfer/barrier from HIVMAnalysis)
must land before M3/M4 consume them — schedule this emitter work in Wk 2–3
alongside M2.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Bound violated (`T_bound > T_measured`, unsound) | Exp 1 catches it; sustained (not peak) `I_c` keeps it achievable; localize to a too-high `I_c` (recalibrate lower) or a missed mandatory handoff (move avoidable→irreducible). §6 flags every peak. |
| `ascend_910b.json` peaks silently reused for 910B3 | §6 table tags each constant; M1 P0 gate blocks any datasheet peak from entering `I_c`. 910B3 ≠ 910B1 ≠ ascend_910b defaults. |
| Bound too loose to guide | Exp 2 on optimized kernels; loose ⇒ a missing achievable mechanism, add analytically (never a fudge constant — strip tilesim `delta` fudges on port). |
| Grid affine map not recoverable for exotic kernels | Template common idioms first (`grid_idioms.py`), symbolic-execution fallback, report coverage honestly. |
| Existing HIVM/TTIR dumps lack per-op metadata M3 needs | Audit dumps Wk 2; add a thin C++ annotation pass only if a field is missing — keep it metadata-only, no analysis in C++. |
| Gap 2 vs Gap 4 conflated (both high-R low-E) | Key off component TYPE (MTE→Gap2, compute→Gap4) in M3/M5, not the R/E value. |
| `redundancy(grid)` / L2 model wrong | Default OFF (redundancy=1, conservative — counts more traffic, never less). Enable only after Exp 7 shows `L2_residency` stable; report as flagged second-order estimate. |
| Two-limit claim on non-equivalent kernels | Restrict to clean-compile + output-verified subset; compute T_bound_HIVM under register/buffer/capacity/divisibility limits. |
| Wholesale tilesim graft creep | Port only the §4 itemized constants/curves/classifiers; explicit do-NOT-port list (search/scheduler/cache-sim) enforced in review. |
| Mandatory/avoidable misclassification raises the floor unsoundly | Split errs toward "avoidable" by construction (spec §4.0); a handoff enters T_serial_irreducible only if provably cross-component + data-dependent. |

---

## 10. Verification Steps

Per-module acceptance is in §3 (each `Acceptance` line). System-level
verification = Part B experiments, validated via `remote-bench-910b3`:

1. **M1**: every P0 constant <5% run-to-run variance over ≥30 runs;
   `mandatory_handoff_cost` separable from K-scaling. (CI recorded in JSON.)
2. **M2**: 10 reference kernels — `tile_assignment`, `occupancy`,
   `load_balance` match hand calculation; illegal tilings rejected.
3. **M3**: `Σ O_prec` reconciles to analytic flop/byte within 2%; seeded
   i32-compare Scalar fallback flagged by the oracle.
4. **M4**: hand-computed kernel — `I_c`, `T_core_floor`,
   `T_serial_irreducible` match a spreadsheet to 3 sig figs (pytest golden).
5. **M5**: five-way attribution sums consistently; worked example from spec §7
   (MatMul→Softmax, M=2048, BLOCK_M=128 → occupancy 0.80, grid binds)
   reproduced exactly as a golden test.
6. **M6 (remote-bench)**:
   - **Soundness (Exp 1)**: `T_bound ≤ T_measured` on 100% of the suite
     (binary; any violation = bug). `T_measured` from msprof
     `op_summary_*.csv` `Task Duration(us)`.
   - **Tightness (Exp 2)**: median `T_measured/T_bound < 1.20` on optimized
     kernels.
   - **Attribution (Exp 3)**: seeded kernels, gap-id accuracy >90%,
     quantification error <20% vs measured gap-removal (HIVM counterfactual
     via remote-bench).
   - **Two-tier ablation (Exp 4)** and **baselines (Exp 4b)**: two-tier beats
     sustained-roofline (the kept legacy `RooflineAnalysis.cpp`) and ECM-style
     on MAE and cause-id.
7. Run `tests/perfbound/` (pytest) on every commit; golden-number tests gate
   M4/M5 regressions independently of hardware.

---

## 11. Open Questions

Tracked in `/mnt/d/work/git/vTriton/.omc/plans/open-questions.md`. The genuine
unknowns (not user-preference questions): 910B3 sustained-rate calibration
values, whether existing HIVM/TTIR dumps already carry per-op precision/
transfer/unit metadata, bishengir Python-binding availability for M3, and the
exact remote 910B3 microbench harness for M1.
