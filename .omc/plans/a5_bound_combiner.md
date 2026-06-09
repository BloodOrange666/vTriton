# A.5 Plan — Module 5: Bound Combiner, Attribution, Two-Limit + Real-Kernel Milestone (rev. 2)

> On approval, save the canonical copy to `.omc/plans/a5_bound_combiner.md`
> (consistent with `a0_`–`a4_` naming) and create `.omc/plans/a5_progress.md` on completion.
>
> **Rev. 2** incorporates a review pass that found 8 issues (3 of them bugs in *existing*
> code). The corrected scope: bound-composition soundness, inter-program (wave) work scaling,
> two-limit-by-recomputation, Gap 1 Scalar detection, the grid-gap and Gap-3 formulas, and a
> non-vacuous milestone.

---

## Context

A.4 delivered the two analytical floors (`compute_grid_floor`, `compute_component_floor`,
`T_serial_irreducible`) and the `compute_bounds` driver. **A.5 (Module 5)** is the spec's
**bound combiner + five-way attribution + per-kernel report** (`implementation_and_paper_plan.md`
§A.5; `performance_bound_model.md` §3, §4), plus the analytical half of **A.7 two-limit**.

Much of the A.5 *scaffold already exists* (`bound_combiner.combine`, `Attribution`,
`_compute_gap1/2/4`, `report.KernelReport`, `two_limit.TwoLimitResult`) — **but the review showed
the scaffold is not just incomplete, it is in places incorrect.** The combiner composition can
violate the lower-bound (soundness) property; neither floor accounts for kernels with more
programs than cores; Gap 1 cannot detect the canonical Scalar fallback; `grid_gap`/Gap-3 are
unimplemented *and* the formulas I first proposed were wrong; two-limit is a stub *and* the
"subtract gaps" idea double-relaxes the model. And the whole surface is untested.

**Milestone (user-chosen):** validate the full stack on the real FLA kernel
`test/chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2.py`, **fully automated**:
`kernel.py → (wheel compile, NPU-mocked on CPU) → kernel.npuir.mlir → (tritonsim-hivm) → des.json → extract → A.5 report`,
**requiring the C++ HIVM parser fix** (user-chosen). The built `tritonsim-hivm` currently parses
**no** HIVM (`unsupported memory space Attribute`, `Dialect 'hivm' not found for 'hivm.hir.set_ctrl'`)
on both the fresh compile and the committed fixture — the source of the 3 known xfails.

**Intended outcome:** a *sound* bound, correctly scaled to the real launch grid, a genuinely
five-way attribution, a correctly-formulated analytical two-limit, a working typed HIVM parser, a
green A.5 test suite, and a defensible end-to-end report on a real kernel.

---

## Verified facts (this planning session)

- **Compile works without NPU hardware:** `tritonsim-hivm --triton-script <kernel.py> --python <env>` activates `[compile-only] NPU device mocking` (CPU) and dumps `kernel_001.npuir.mlir`; lowering succeeds on the tiny kernel.
- **Triton-Ascend env:** `/home/shane/miniconda3/envs/vtriton-verify/bin/python` (has the `ascend` backend). Stock base `python3` does not.
- **bishengir:** header present (`…/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVM.h`); **static lib `libBiShengIRHIVMDialect.a` NOT built** (triton-ascend native build dir absent). CMake auto-detect therefore left `TRITONSIM_HAS_BISHENGIR_HIVM=OFF` despite `TRITONSIM_ENABLE_BISHENGIR_HIVM=ON`. Parser falls to the `#ifndef` text path (`HIVMAnalysis.cpp:834`, `parseSourceString` + `allowUnregisteredDialects` `:2736`) → the observed errors.
- **No wave factor in the model:** `compute_bounds`/`compute_component_floor`/`compute_grid_floor` use **one program's** work; grid divides by `n_cores`; no `total_programs`/`waves` multiplier exists (`grid_model.py:96`). chunk_kda is 4096 programs on 20 cores ≈ 205 waves.
- **Extractor double-emits handoffs:** for each cross-component dependency it emits an *immediate* and a *canonical* edge (`hivm_extractor.py:329-334`) — must be deduped before summing Gap-3 cost.

---

## Scope decisions

- **Fix the parser (in scope).** Route 1: build & link the typed bishengir HIVM dialect against vTriton's `thirdparty/llvm-project`. Route 2 (if ABI/build intractable): extend the text fallback parser. (Change #1.)
- **A.7 = analytical, by recomputation.** `T_bound_DSL` and `T_bound_HIVM` are *two bounds computed from two structural constraint sets* — **not** arithmetic on diagnostic gap values. (Change #7.)
- **Do NOT build the multiplicative §4.2 attribution** — only meaningful against a measured time (M6). Keep fractions-of-`T_bound` for dominant-gap selection; document the divergence.

---

## Changes

### 1. Fix the C++ HIVM parser  *(prerequisite for the automated milestone)*
**Files:** `CMakeLists.txt`, `lib/AscendModel/Analysis/{HIVMAnalysis.cpp,CMakeLists.txt}`; build of `…/AscendNPU-IR/bishengir`.
- **Route 1 (primary):** build `libBiShengIRHIVMDialect.a` (+ deps) **against vTriton's `thirdparty/llvm-project`** (same LLVM commit → no ABI mismatch); point CMake at it so `TRITONSIM_HAS_BISHENGIR_HIVM` flips ON; rebuild. Native typed ingestion then parses `set_ctrl`/`address_space`.
- **Route 2 (fallback):** extend the `#ifndef` text path — register a minimal local `hivm` dialect / relax the memory-space attribute + `index`↔`i64` handling so `parseSourceString` ingests `hivm.hir.*` without fatal errors.
- **Acceptance:** `tritonsim-hivm --npuir-file test/hivm_add_kernel.npuir.mlir` exits 0 with a non-empty `operations` array; the 3 CLI xfails flip to pass (remove the xfail-on-failure shims once green).

### 2. **(NEW — F2/F3) Bound composition + inter-program (wave) scaling — soundness core**
**Files:** `perfbound/combine/bound_combiner.py` (`combine`), `perfbound/model/bounds.py` (`compute_bounds`), `perfbound/model/grid_model.py`, possibly `perfbound/model/component_model.py`.

**2a. Composition (F3).** Change `T_bound = max(T_grid_floor, T_core_floor) + T_serial_irreducible`
to **`T_bound = max(T_grid_floor, T_core_floor + T_serial_irreducible)`**.
Rationale: serialization is intra-core (spec §4.0/§4.1 prose: "+T_serial attaches to the Tier-2 term").
`max(a,b)+c ≥ max(a, b+c)` for `c≥0`, so the old form can *overstate* a lower bound → risk
`T_bound > T_measured` (unsound). The new form is the tightest provable lower bound and matches the
prose. (Note the spec's §4.1 formula box writes the additive form — internally inconsistent with its
own prose; resolve in favor of soundness + prose.) Update `bound_combiner.py:143-146` and the
binding-tier logic; add a soundness test asserting `T_bound = max(grid, core+serial)`.

**2b. Wave scaling (F2).** The extract is **one program**; the busiest core runs
`waves = ceil(total_programs / n_cores)` programs. Apply this factor consistently:
- **Tier 2:** busiest-core component work `O_c = waves × per_program_O_c` → `T_core_floor` scales by `waves`.
- **Tier 1:** chip work `total_work = total_programs × per_program_work`; `T_grid_floor = total_work / (n_cores·occ·lb·I)` then naturally carries the per-core `waves` factor.
- Thread `total_programs` (and `n_cores`) from the real launch grid into `compute_bounds`; default `waves=1` preserves all existing single-wave tests. Add a wave-scaling test (e.g. 40 programs / 20 cores → 2× both floors).

### 3. Gap 1 — detect the canonical Scalar fallback (F4)
**File:** `perfbound/combine/bound_combiner.py` (`_compute_gap1`, `:311-346`).
- **Remove the `if op.component == Component.SCALAR: continue` skip** (`:315`). A Scalar op whose semantic-eligibility set includes Vector/Cube (e.g. the seeded i32-compare) *is* the canonical Gap-1 mis-placement (spec §3 Gap 1; `implementation_and_paper_plan.md:126`). Keep skipping only fixed-assignment MTE ops.
- Quantify the Scalar→Vector mis-placement cost via the eligible unit's rate (the time it *would* take on Vector vs the Scalar path); since Scalar `t_c=0` today, source the cost from the op's work × the eligible-unit rate. Diagnostic-only (over-count safe).
- Add a seeded-i32-compare unit test asserting Gap 1 > 0 and is dominant.

### 4. Gap 3 — avoidable-serial cost, deduped (F6)
**Files:** `perfbound/model/serialization.py` (`:201-207`) and/or `bound_combiner.py`.
- **Dedup first:** the extractor emits immediate + canonical edges per dependency (`hivm_extractor.py:329-334`); key avoidable handoffs by their underlying dependency (or `(producer,consumer)` like the mandatory-edge dedup) so each is counted once.
- Cost per distinct avoidable handoff via the **same `memory.lookup_bw` path `_compute_gap2` uses** (`bytes/BW`). Document this as a **conservative upper estimate** of the stall — true Gap 3 is only the *non-overlapped* portion; an overlap model is a later refinement. Diagnostic-only, never enters `T_bound`, so over-estimation cannot break conservatism.
- Tests: deduped avoidable handoff → single non-zero cost; no double-count from the immediate/canonical pair; zero without calibration.

### 5. Grid gap — populate the fifth axis (F5, corrected formula)
**File:** `perfbound/combine/bound_combiner.py` (`combine`).
- `grid_gap_us = T_grid_floor × (1 − occupancy·load_balance)` — the realized floor minus the ideal-grid floor (`T_ideal = T_grid_floor · occ·lb`). (Corrected from the rev-1 `(1/(occ·lb) − 1)`, which over-counted by a `1/(occ·lb)` factor.)
- Source occ/lb from `GridBound`; perfect grid → 0. Test occ<1 / lb<1 → exact hand value; occ=lb=1 → 0.

### 6. Report — at-bound state (F8)
**File:** `perfbound/combine/report.py` (`from_bound`, `dominant_gap` consumer).
- When the largest gap fraction is below a small ε (all gaps ≈ 0), emit **"at component bound — no actionable software gap (consider algorithmic redesign: fusion/precision/less traffic)"** per spec §3 "Not a Gap — Component Bound", instead of defaulting to the grid recommendation (current `dominant_gap()` returns the first/zero entry → spurious "fix grid"). Add an `at_bound` branch + test.

### 7. Two-limit (A.7) — by recomputation, not subtraction (F1)
**Files:** `perfbound/combine/two_limit.py`, `perfbound/model/bounds.py`.
- **`T_bound_DSL`** = `T_bound` from the **realized** extract (Changes #2–#5 applied).
- **`T_bound_HIVM`** = `T_bound` recomputed from an **idealized extract** built by relaxing the *avoidable* structural constraints — re-assign Gap-1 mis-placed ops to their eligible unit (recompute per-component `O_c`) and remove avoidable handoffs from the serialization set (they no longer serialize). Run the **same `compute_bounds`/`combine`** on this idealized extract. This is a recomputation from a second legal constraint set (spec §A.7), *not* `T_bound_DSL − gaps`.
- `compiler_headroom = T_bound_DSL − T_bound_HIVM` (≥ 0 by construction since relaxation only lowers floors); `author_headroom = None` until M6 supplies `T_measured`. Remove the `NotImplementedError`; wire into `KernelReport.from_bound(result, two_limit=…)`.
- Tests: idealizing a Gap-1 kernel lowers `T_bound_HIVM` below `T_bound_DSL`; a kernel already at the bound → `compiler_headroom ≈ 0`.

### 8. End-to-end driver + CLI (F7 — consistent API)
**File (new):** `perfbound/combine/run_report.py` with `__main__`.
- Two explicit entry points, no conflation: `report_from_npuir(npuir_path, grid, calib_db, hardware_config)` (runs `extract_from_npuir` → tool → des → extract) **and** `report_from_desgraph(des_json, grid, calib_db)` (consumes an existing des.json via `extract_hivm`).
- **Separate the two configs:** `hardware_config` (`configs/ascend_910b.json`) is for the **C++ tool** stage only; the **Python model** stage uses the **calibration DB** (`load_default_calib_db()`). The CLI must pass each to the right stage.
- Populate the grid tier from the **real launch grid** (`total_programs`, `n_cores`, occupancy/load_balance, `waves`) — not the `occupancy=1.0, n_cores=20` placeholder.
- CLI: `python -m perfbound.combine.run_report --desgraph /tmp/kda_des.json --grid 128,32` → `to_text()` + `to_json()`.

### 9. Tests — the largest gap (zero A.5 coverage today)
**Files (new):** `test_combine.py`, `test_report.py`, `test_two_limit.py`; extend `test_serialization.py`.
- Covers every change above: composition `max(grid, core+serial)` (2a), wave scaling (2b), Gap-1 Scalar (3), Gap-3 dedup (4), grid-gap formula (5), at-bound state (6), two-limit recomputation (7), report round-trip + recommendation mapping.

### 10. Real-kernel milestone — non-vacuous acceptance (F7)
**File (new):** `tests/perfbound/test_chunk_kda_milestone.py`.
- Compile via `tritonsim-hivm --triton-script test/chunk_kda_*.py --python <vtriton-verify> --hardware-config configs/ascend_910b.json --des-graph-file /tmp/kda_des.json` (constexprs H=32,K=128,V=128,BT=64,BK=32,BV=32, TRANSPOSE_STATE=False, IS_VARLEN; grid (128,32); resolve `--script-arg`/`--entry-arg` empirically — the autotune wrapper is the friction point). Then `report_from_desgraph(...)`.
- **Non-vacuous assertions (not mere positivity):**
  (a) `Σ O_prec` reconciles with a **hand-derived analytic flop/byte estimate** for chunk_kda (count `tl.dot`s × M·N·K from the dims; bf16 load/store bytes) within a stated tolerance (spec A.3 acceptance uses 2%; allow a looser sanity band here);
  (b) the **binding component** matches the hand prediction (Cube- vs MTE-bound from the analytic ratio);
  (c) `T_bound_us` is within a stated factor of the hand estimate (with wave scaling applied — sanity, not 3-sig-fig);
  (d) `T_serial_irreducible > 0` (real Cube↔Vector handoffs: `tl.dot`→`exp2`→`tl.dot`).
- **When the tool/env is present but the run fails → `xfail` with the error** (track the gap), not silent `skip`. `skip` only when the binary/env is genuinely absent.
- Commit a reference report (`test/chunk_kda_report.json`) as a regression anchor.

---

## Open items this stage closes

| Item | Source | Closed by |
|------|--------|-----------|
| Bound composition can be unsound (`max+serial` vs `max(grid, core+serial)`) | review F3; spec §4.0 prose | Change #2a |
| No inter-program/wave scaling (both floors understate on grid≫cores) | review F2; `grid_model.py:96` | Change #2b |
| Gap 1 cannot detect canonical Scalar fallback | review F4; `bound_combiner.py:315` | Change #3 |
| Gap 3 cost stubbed to 0; double-counts immediate/canonical handoffs | review F6; `serialization.py:207`, `hivm_extractor.py:329` | Change #4 |
| `grid_gap_us` never computed (and rev-1 formula wrong) | review F5 | Change #5 |
| Zero-gap report recommends "fix grid" | review F8; `report.py:113` | Change #6 |
| `two_limit` NotImplementedError; subtraction double-relaxes | review F1; `two_limit.py:67` | Change #7 |
| C++ HIVM parser parses no HIVM (3 CLI xfails) | smoke test | Change #1 |
| No A.5 test coverage; milestone could pass vacuously | review F7 | Changes #9, #10 |

**Do NOT claim closed:** A.7 measured author-headroom (M6/msprof); multiplicative §4.2 attribution (M6); the Gap-3 overlap model (only the conservative upper estimate now); Group-V seeded-gap *validation* (B.4); L2-cache BW model; Scalar `t_c=0` rate divergence (carried from A.4 — Change #3 detects the *placement* but the Scalar rate itself stays 0).

---

## Verification

```bash
cd /mnt/d/work/git/vTriton
# Change #1 — fixture must now parse:
./build/bin/tritonsim-hivm --npuir-file test/hivm_add_kernel.npuir.mlir \
  --des-graph-file /tmp/x.json --hardware-config configs/ascend_910b.json && \
  python3 -c "import json;print('ops',len(json.load(open('/tmp/x.json'))['operations']))"

# Changes #2-#9 — A.5 Python suite (incl. soundness + wave-scaling + gap semantics):
python3 -m pytest tests/perfbound/test_combine.py tests/perfbound/test_report.py \
  tests/perfbound/test_two_limit.py tests/perfbound/test_serialization.py -q
python3 -m pytest tests/perfbound/ -q   # full suite; 3 CLI xfails should now PASS

# Change #10 — automated stack on the real kernel:
PY=/home/shane/miniconda3/envs/vtriton-verify/bin/python
./build/bin/tritonsim-hivm --triton-script test/chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2.py \
  --python "$PY" --hardware-config configs/ascend_910b.json \
  --des-graph-file /tmp/kda_des.json --scheduler des
python3 -m perfbound.combine.run_report --desgraph /tmp/kda_des.json --grid 128,32
```

**Acceptance gate:** (1) fixture parses (xfails flip to pass); (2) A.5 suite green with the
composition-soundness, wave-scaling, Gap-1-Scalar, Gap-3-dedup, grid-gap, at-bound, and
two-limit-recomputation tests all asserting *hand values* (not positivity); (3) chunk_kda runs
fully-automated to a report whose `Σ O_prec`, binding component, and `T_bound` magnitude
reconcile with a hand-derived analytic estimate.

---

## Risks

| Risk | Mitigation |
|------|------------|
| bishengir HIVM lib ABI mismatch vs vTriton LLVM | Build bishengir against `thirdparty/llvm-project`; else Route 2 (text-parser fix, no link) |
| Wave scaling mis-derives per-core program count for ragged/varlen grids (chunk_kda is varlen) | Derive `waves` from the realized `tile_assignment`/busiest-core count from M2 when available; fall back to `ceil(total_programs/n_cores)`; test both |
| Composition change `max(grid, core+serial)` regresses A.4 golden numbers | A.4 goldens are single-wave with grid not binding; assert they're unchanged, add the new composition test separately |
| Two-limit idealized-extract construction is subtle (which constraints are "avoidable") | Default to relaxing only Gap-1 placement + avoidable handoffs (unambiguous); leave Gap-2/4 efficiency in `T_bound_HIVM`; document the choice |
| chunk_kda compile fails through `@triton.autotune`/`@heuristics` | Smoke the tiny kernel's `--triton-script` first; resolve binding flags empirically; compile is proven to reach HIVM dump |
| Milestone occupancy=1.0 → no Tier-1 exercise | Validate grid-gap/wave via unit tests (Change #9); milestone validates Tier-2+serial+gaps+two-limit |
