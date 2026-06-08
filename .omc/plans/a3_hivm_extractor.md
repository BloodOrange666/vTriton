# A.3 Implementation Plan - HIVM Extractor (M3 Tier 2 Input)

**Stage**: A.3 / Module 3 of the two-tier analytical performance bound model
**Spec**: `.omc/specs/implementation_and_paper_plan.md` A.3; `.omc/specs/performance_bound_model.md` Tier 2 and Section 5
**Created**: 2026-06-08
**Prerequisites**: A.1 measured calibration complete; A.2 DSL extractor complete
**Primary files**: `perfbound/extract/hivm_extractor.py`, `perfbound/extract/op_classifier.py`, `perfbound/extract/eligibility_oracle.py`, `lib/AscendModel/Analysis/HIVMAnalysis.cpp`, `lib/AscendModel/Transforms/HIVMAnalysisPass.cpp`

## Requirements Summary

A.3 must turn one core's HIVM program, selected by the A.2 `GridInfo.busiest_core_id`, into the Tier 2 structural quantities consumed by A.4. Per `.omc/specs/implementation_and_paper_plan.md:109`, A.3 reads the HIVM of one core's program and extracts per-component quantities. The required outputs are listed in `.omc/specs/implementation_and_paper_plan.md:114`: `O_prec[component]`, transfer sizes and alignments, realized unit assignment, repeat/mask/SIMD-lane parameters, and handoff records.

The performance model depends on this split because Tier 2 is the per-core component floor for the busiest core (`.omc/specs/performance_bound_model.md:128`). A.3 must not compute the final harmonic-mean rates or final bound; those are A.4/A.5 responsibilities (`.omc/specs/implementation_and_paper_plan.md:135`, `.omc/specs/implementation_and_paper_plan.md:151`). A.3's job is reliable extraction plus semantic eligibility for Gap 1.

A.2 is complete and provides the Tier 1 input pipeline: `TTIR -> ExtractTTIRInfoPass -> GridInfo` with 63/63 tests passing (`.omc/plans/a2_progress.md:77`, `.omc/plans/a2_progress.md:130`). A.3 should reuse that AST-based extraction posture and avoid regex/text scraping in Python.

## Current State and Gaps

Existing scaffold:

- `perfbound/extract/hivm_extractor.py:33` defines `OpRecord`, `HandoffRecord`, and `HIVMExtract`.
- `perfbound/extract/op_classifier.py:14` defines the six component classes: `cube`, `vector`, `scalar`, `mte_gm`, `mte_l1`, `mte_ub`.
- `perfbound/extract/eligibility_oracle.py:58` has a category/precision eligibility API.
- `include/AscendModel/Analysis/HIVMAnalysis.h:47` already tracks HIVM op metadata including pipe, duration, bytes, elements, src/dst spaces, elem type, buffers, and dependencies.
- `lib/AscendModel/Analysis/HIVMAnalysis.cpp:3092` emits DES graph JSON, and `tools/tritonsim-hivm/tritonsim-hivm.cpp:92` exposes `--des-graph-file`.

Blocking gaps to close before A.3 can be marked complete:

- `perfbound/extract/hivm_extractor.py:112` consumes `data.get("nodes", [])`, but `HIVMAnalysisReport::emitDESGraph()` emits `"operations"` at `lib/AscendModel/Analysis/HIVMAnalysis.cpp:3134`. The primary C++ emitter and Python consumer currently disagree.
- `HIVMAnalysisPass` exposes report and Perfetto output only (`include/AscendModel/Transforms/Passes.td:156`); it does not expose a DES graph option even though the CLI tool does. This makes `tritonsim-opt --analyze-hivm` less useful for A.3 tests.
- `PipelineAnalysisPass` writes `pipeline_dep_graph.json` to the current working directory (`lib/AscendModel/Transforms/PipelineAnalysisPass.cpp:439`), which is not reproducible for a Python extractor.
- `HIVMExtract.transfer_sizes` and `transfer_alignments` exist (`perfbound/extract/hivm_extractor.py:89`) but are not populated in `extract_hivm()`.
- `extract_hivm()` aggregates `o_prec` with `op.elements` for every component (`perfbound/extract/hivm_extractor.py:220`), but MTE components must be byte-counted while compute components are op/element-counted.
- The eligibility oracle is not yet connected to a TTIR/Linalg semantic extraction path. The spec requires the oracle to derive which unit an op could run on from semantics, then compare that with realized HIVM assignment (`.omc/specs/implementation_and_paper_plan.md:126`).

## Acceptance Criteria

### AC-1: C++/Python DES graph schema compatibility

Given `test/hivm_add_kernel.npuir.mlir`, running either:

```bash
./build/bin/tritonsim-hivm --npuir-file test/hivm_add_kernel.npuir.mlir --des-graph-file /tmp/hivm_add_des.json
```

or the new `tritonsim-opt --analyze-hivm="des-graph-file=/tmp/hivm_add_des.json"` path emits JSON with:

- `schema_version`
- `operations` as a non-empty array
- per-op `id`, `name`, `pipe`, `duration`, `bytes`, `elements`, `loop_multiplier`, `depends_on`, `src_space`, `dst_space`, `elem_type`

`load_hivm_desgraph()` parses this JSON into non-empty `OpRecord` objects without using regex.

### AC-2: `O_prec` reconciliation within 2 percent

For at least these fixtures:

- `test/hivm_add_kernel.npuir.mlir`
- `test/hivm_mixed_cv_kernel.npuir.mlir`

`sum(O_prec)` reconciles with hand-derived analytic op/byte counts within 2 percent, matching the A.3 spec acceptance in `.omc/specs/implementation_and_paper_plan.md:131`.

Compute components must aggregate operations/elements with loop multipliers. MTE components must aggregate bytes with loop multipliers.

### AC-3: Transfer metadata populated

For every MTE op in the DES graph:

- `transfer_sizes[component]` contains the byte size.
- `transfer_alignments[component]` contains a conservative alignment bucket or explicit `0`/`None` when the emitter cannot infer alignment.
- `src_space` and `dst_space` are normalized to the memory-space vocabulary used by the model (`gm`, `ub`, `l1`, `l0a`, `l0b`, `l0c`).

### AC-4: Handoff list is structurally correct

`extract_hivm()` must build `HandoffRecord` entries for cross-component producer/consumer edges using C++ dependency metadata. On `test/hivm_mixed_cv_kernel.npuir.mlir`, the Cube-to-Vector path must produce at least one handoff with `producer_component == cube` and `consumer_component == vector`.

Mandatory vs avoidable classification remains A.4, but A.3 must provide enough producer/consumer component and byte metadata for `perfbound/model/serialization.py`.

### AC-5: Eligibility oracle flags seeded i32 compare fallback

Add a deliberately seeded fixture where an i32 compare is realized as Scalar. The semantic eligibility oracle must flag the Scalar fallback case required by `.omc/specs/implementation_and_paper_plan.md:131`.

Unknown semantic categories must stay conservative: include more eligible units rather than creating false Gap 1 positives, matching `perfbound/extract/eligibility_oracle.py:15`.

### AC-6: A.2 compatibility and extractor hygiene

All existing A.1/A.2 perfbound tests continue to pass, including the 63-test A.2 suite recorded in `.omc/plans/a2_progress.md:77`. No `re.finditer`, `re.search`, or `re.match` is introduced under `perfbound/extract/`.

### AC-7: Deterministic output paths

No C++ pass writes `pipeline_dep_graph.json` or trace artifacts into the caller's current working directory unless the user explicitly requested that path. Tests must write all generated JSON into `tmp_path`.

### AC-8: Progress artifact

After implementation, add `.omc/plans/a3_progress.md` with exact commands run, test counts, known gaps, and whether A.4 can proceed.

## Implementation Steps

### Step 1 - Lock the schema with failing tests

Add focused tests before changing implementation:

- `tests/perfbound/test_hivm_extractor.py`
  - fixture DES JSON with an `operations` array, not `nodes`
  - verifies parse of `id`, `name`, `pipe`, `elem_type`, `src_space`, `dst_space`, `depends_on`, `loop_multiplier`
  - verifies MTE `o_prec` uses bytes and compute `o_prec` uses elements/flops
  - verifies `transfer_sizes` and `transfer_alignments` are populated
- `tests/perfbound/test_eligibility_oracle.py`
  - matmul FP16/BF16/INT8 -> Cube
  - elementwise/reduction -> Vector
  - i32 compare -> Scalar fallback flag
  - unknown category remains conservative
- `tests/perfbound/test_hivm_cli_integration.py`
  - skipped when `build/bin/tritonsim-hivm` or `build/bin/tritonsim-opt` is absent
  - runs the CLI into `tmp_path`, parses JSON, and checks non-empty operations

This step turns the known mismatch at `perfbound/extract/hivm_extractor.py:112` into a regression test.

### Step 2 - Stabilize the C++ DES graph schema

Update `HIVMAnalysisReport::emitDESGraph()` in `lib/AscendModel/Analysis/HIVMAnalysis.cpp:3092`:

- emit a top-level `schema_version`, for example `"a3_hivm_des_v1"`
- keep `operations` as the canonical key
- include `start_cycle` and `end_cycle` because `HIVMOp` already tracks them in `include/AscendModel/Analysis/HIVMAnalysis.h:61`
- keep `depends_on`, `read_buffers`, `write_buffers`, versions, `src_space`, `dst_space`, and `elem_type`
- use `llvm::json` or a small escaping helper so op names and buffer names cannot produce malformed JSON

Then extend `HIVMAnalysisPass`:

- add `desGraphFile` option in `include/AscendModel/Transforms/Passes.td:156`
- write `report.emitDESGraph()` from `lib/AscendModel/Transforms/HIVMAnalysisPass.cpp:65`
- preserve existing report/perfetto behavior

For `PipelineAnalysisPass`, either add an explicit `dependency-graph-file` option or stop using it as an A.3 input. If retained, replace the hard-coded `pipeline_dep_graph.json` write at `lib/AscendModel/Transforms/PipelineAnalysisPass.cpp:439` with an option defaulting to empty.

### Step 3 - Fix Python DES graph ingestion and aggregation

Update `perfbound/extract/hivm_extractor.py`:

- parse `operations` as canonical and accept `nodes` only as a legacy fallback
- validate required fields and raise a clear `ValueError` for malformed JSON
- store `start_cycle`, `end_cycle`, `src_space`, `dst_space`, `elem_type`, `read/write_buffers`, and dependency ids
- compute work by component:
  - Cube/Vector/Scalar: `flops` when present, otherwise `elements`, multiplied by `loop_multiplier`
  - MTE-GM/MTE-L1/MTE-UB: `bytes_transferred`, multiplied by `loop_multiplier`
- populate `o_prec`, `total_flops`, `total_bytes`, `transfer_sizes`, `transfer_alignments`, and `unit_assignment`
- normalize pipe names using `PIPE_TO_COMPONENT` and `HW_UNIT_TO_COMPONENT` in `perfbound/extract/op_classifier.py:32`

Keep compatibility with existing A.4 tests in `tests/perfbound/test_component_model.py`.

### Step 4 - Build the A.3 extraction runner

Add a thin runner API, preferably in `perfbound/extract/hivm_runner.py`:

```python
extract_from_npuir(
    npuir_path: Path,
    *,
    tool: Path | None = None,
    hardware_config: Path | None = None,
    arg_bindings: dict[str, int] | None = None,
    scheduler: str = "des",
) -> HIVMExtract
```

The runner should:

- call `tritonsim-hivm --npuir-file ... --des-graph-file tmp/des.json` when available
- support `tritonsim-opt --analyze-hivm="... des-graph-file=..."` after Step 2
- always write JSON into a caller-provided or temporary directory
- return `HIVMExtract`
- include the selected `busiest_core_id`/program-id binding surface needed to connect A.2 `GridInfo` to one-core HIVM extraction

Do not add network, remote profiling, or compile-run search behavior. The model remains analytical; measurement is only A.1 calibration and A.6 validation (`.omc/specs/performance_bound_model.md:11`).

### Step 5 - Implement semantic eligibility input

Extend the A.2-style AST extraction path instead of parsing TTIR text:

- add semantic op records to `ExtractTTIRInfoPass` or create a narrowly named semantic pass
- record per semantic op: stable id if available, op name/category, precision, shape/elements, and source location when present
- add `perfbound/extract/semantic_extractor.py` to convert that JSON into semantic records
- update `eligibility_oracle.py` so Gap 1 checks compare semantic eligibility against realized `unit_assignment`

The oracle should preserve current conservative behavior for unknowns. It should not declare a Gap 1 unless the semantic category and precision make the eligible unit set clear.

### Step 6 - Reconciliation and fixture coverage

Add `perfbound/extract/reconcile.py` or equivalent helper with:

- `reconcile_extract(extract, expected) -> ReconciliationReport`
- per-component absolute and relative error
- failure when any required component exceeds 2 percent error

Use hand-derived expected values for:

- vector add fixture: two GM-to-UB loads, one Vector add, one UB-to-GM store
- mixed Cube/Vector fixture: Cube loads, `mmadL1`, `fixpipe`, vector load/add, store
- seeded i32 compare Scalar fallback fixture

The test should verify both aggregate `O_prec` and individual transfer metadata, not just non-zero counts.

## Open Items to Close in A.3

A.3 should explicitly close the extractor-related open items below. Items outside the extractor boundary, such as L2 hit-rate modeling and validation-kernel-suite selection, remain deferred to later stages.

| Item | Source | Why A.3 owns it | Closure condition |
|------|--------|-----------------|-------------------|
| End-to-end C++ JSON verification | `.omc/plans/open-questions.md:51`, `.omc/plans/a0_progress.md:74` | A.3 depends on real DES graph JSON from `.npuir.mlir`, not only hand-written Python fixtures | A CLI integration test runs `tritonsim-hivm --npuir-file ... --des-graph-file tmp/des.json`, parses the result, and reconciles extracted work within 2 percent |
| DES graph schema mismatch | `perfbound/extract/hivm_extractor.py:112`, `lib/AscendModel/Analysis/HIVMAnalysis.cpp:3134` | Python currently reads `nodes`, while C++ emits `operations`; this can produce an empty extract | `load_hivm_desgraph()` accepts canonical `operations`, tests fail on missing required fields, and legacy `nodes` support is clearly marked |
| Deterministic C++ JSON output paths | `lib/AscendModel/Transforms/PipelineAnalysisPass.cpp:439` | A.3 tests must not depend on or pollute the caller's current directory | DES/dependency graph outputs are written only to explicit paths or temporary directories |
| Metadata completeness for Tier 2 | `.omc/specs/implementation_and_paper_plan.md:114`, `.omc/plans/a0_progress.md:13` | A.3 must deliver transfer metadata, realized assignment, repeat/mask defaults, and handoffs for A.4/A.5 | `HIVMExtract` has populated `o_prec`, `transfer_sizes`, `transfer_alignments`, `unit_assignment`, and `handoffs` on vector and mixed Cube/Vector fixtures |
| MTE byte-vs-element aggregation bug | `perfbound/extract/hivm_extractor.py:220` | Counting MTE as elements corrupts the memory component floor used downstream | AC-2 tests prove MTE components use bytes and compute components use ops/elements |
| Gap 1 semantic eligibility input | `.omc/specs/implementation_and_paper_plan.md:126`, `perfbound/extract/eligibility_oracle.py:58` | A.3 owns the semantic eligibility oracle; later stages only consume its result | A seeded i32 compare Scalar fallback is flagged, while unknown categories remain conservative |
| Gap 4 field availability contract | `.omc/plans/a0_progress.md:105`, `.omc/plans/a0_progress.md:106` | Repeat/mask data is required as A.3 output even if the C++ emitter cannot infer it yet | DES graph emits repeat/mask when available; Python defaults are explicit and tested as conservative no-gap defaults |

### Step 7 - Documentation and handoff to A.4

Update:

- `perfbound/extract/__init__.py` with the public A.3 APIs
- optional `perfbound/extract/README.md` with DES schema and example commands
- `.omc/plans/a3_progress.md` after implementation
- `.omc/plans/open-questions.md` only if implementation discovers an actual unresolved semantic mapping issue

The A.4 handoff should state that the extractor now provides:

- per-component `O_prec`
- MTE transfer sizes and alignments
- realized unit assignments
- repeat/mask fields with explicit defaults or emitted values
- handoff records with producer/consumer components
- semantic eligibility records sufficient for Gap 1

## Verification Commands

Run these locally after implementation:

```bash
python3 -m py_compile \
  perfbound/extract/hivm_extractor.py \
  perfbound/extract/op_classifier.py \
  perfbound/extract/eligibility_oracle.py

python3 -m pytest \
  tests/perfbound/test_hivm_extractor.py \
  tests/perfbound/test_eligibility_oracle.py \
  tests/perfbound/test_hivm_cli_integration.py \
  tests/perfbound/test_component_model.py \
  tests/perfbound/test_calibration_wiring.py \
  -q

python3 -m pytest tests/perfbound -q
```

If the C++ build exists, also run:

```bash
cmake --build build

./build/bin/tritonsim-hivm \
  --npuir-file test/hivm_add_kernel.npuir.mlir \
  --des-graph-file /tmp/vtriton_a3_hivm_add_des.json

python3 -m pytest tests/perfbound/test_hivm_cli_integration.py -q
```

For extractor hygiene:

```bash
rg -n "re\\.finditer|re\\.search|re\\.match" perfbound/extract
```

Expected result: no matches.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| DES graph schema drift between `tritonsim-hivm` and `tritonsim-opt` | Python extractor silently drops ops | Add schema version, canonical `operations` key, and fixture tests for both emit paths |
| Manual JSON construction emits invalid JSON for unusual op/buffer names | Flaky extractor and hard-to-debug parse failures | Use `llvm::json` or explicit escaping in `emitDESGraph()` |
| MTE work counted as elements instead of bytes | A.4 memory floor becomes wrong and may break bound tightness | Add AC-2/AC-3 tests with hand-derived byte totals |
| Eligibility oracle over-flags unknown ops | False Gap 1 attribution | Keep unknown categories conservative and test that behavior |
| Hard-coded `pipeline_dep_graph.json` pollutes cwd | Non-deterministic tests and stale-file bugs | Require explicit output paths or stop using PipelineAnalysis JSON for A.3 |
| A.3 absorbs A.4 model math | Scope creep and mixed acceptance | Treat A.3 outputs as data only; keep harmonic mean, `T_core_floor`, and serialization split tests in A.4 |

## Out of Scope

- Re-running A.1 calibration or changing `calib_910b3_v1.json`
- Computing `I_c`, `T_core_floor`, `T_serial_irreducible`, or final `T_bound` beyond compatibility tests
- A.6 remote validation against measured kernels
- Optimizing schedules, changing tiling, or performing compile-run search
- Any dependency on the cloned Ascend C samples unless needed as reference for C++/Ascend tooling style

## Definition of Done

- [ ] `load_hivm_desgraph()` consumes current C++ DES graph JSON with `operations`
- [ ] `extract_hivm()` returns non-empty operations, populated `o_prec`, transfer metadata, unit assignments, and handoffs
- [ ] `sum(O_prec)` reconciles within 2 percent on vector and mixed Cube/Vector fixtures
- [ ] i32 compare Scalar fallback is flagged by the eligibility oracle
- [ ] C++ emitters use explicit output paths and do not write stale cwd artifacts
- [ ] `python3 -m pytest tests/perfbound -q` passes
- [ ] `rg -n "re\\.finditer|re\\.search|re\\.match" perfbound/extract` returns no matches
- [ ] `.omc/plans/a3_progress.md` records final evidence and A.4 handoff status
