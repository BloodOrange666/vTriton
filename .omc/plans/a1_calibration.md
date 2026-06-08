# A.1 Implementation Plan — Calibration Database (M1)

**Stage**: A.1 of the two-tier analytical performance bound model  
**Spec**: `.omc/specs/performance_bound_model.md §A.1`  
**Parent plan**: `.omc/plans/a0_perf_bound_model.md §3 (Module 1)`  
**Target hardware**: Ascend 910B3 NPU (remote server)  
**Deliverable**: `calib_910b3_v1.json` + CCE microbench suite

---

## 1. Requirements Summary

Populate the calibration database so that `I_c` values are non-zero, measured,
and provably conservative. Measurement is the **only** place hardware numbers
enter the model — every constant must carry value, CI, source, and n_runs.

**Constants to produce (priority order):**

| Constant | Unit | Priority | Notes |
|----------|------|----------|-------|
| `P_cube[fp16]` — sustained Cube FLOPS | TFLOPS | **P0** | Dominant in matmul workloads |
| `P_cube[int8]` | TFLOPS | **P0** | Quantized inference |
| `P_cube[bf16]` | TFLOPS | **P0** | Training workloads |
| `P_cube[fp32]` | TFLOPS | P1 | Fallback; can derive from fp16 |
| `P_vector[add/mul/max/min]` — sustained | GFLOPS | **P0** | Common elementwise ops |
| `P_vector[exp/log/sqrt/rsqrt]` — sustained | GFLOPS | **P0** | Transcendental ops |
| `P_vector[tanh/sigmoid/gelu]` | GFLOPS | P1 | Activation functions |
| `BW[gm→ub]` at 1–20 cores | GB/s per core_num | **P0** | MTE_GM primary path |
| `BW[ub→gm]` at 1–20 cores | GB/s per core_num | **P0** | Store path |
| `BW[gm→l1]` | GB/s | **P0** | Cube data load |
| `BW[l1→l0a]` | GB/s | **P0** | L1 to Cube input |
| `BW[gm→l0b]` | GB/s | P1 | Weight pre-load |
| `mandatory_handoff_cost[l0c→gm+gm→ub]` | cycles | **P0** | Cube→Vector pipeline boundary |
| `clock_rate` | MHz | **P0** | Confirm 1850 MHz vs datasheet |
| `η_amortization(size)` — pkt size curves | dimensionless | P1 | Small-transfer overhead |
| `P_scalar` | Gop/s | P1 | i32 compare/branch |

**Acceptance criteria (all P0 constants):**
- Run-to-run variance < 5% (CV = std/mean)
- ≥ 30 independent runs per constant
- `mandatory_handoff_cost` passes K-scaling test (vary K, confirm T = intercept + slope·K)
- Every constant stored with: `value`, `ci_95`, `source="cce_microbench"`, `n_runs`

---

## 2. Acceptance Criteria

### AC-1: Measurement quality (per P0 constant)
- `n_runs ≥ 30` for every P0 constant (enforced in `fit_constants.py`)
- Coefficient of variation CV = std/mean < 5% for every P0 constant
- `ci_95 / value < 0.025` (i.e., 95% CI half-width < 2.5% of measured value)
- Source field = `"cce_microbench"` for all P0 constants (no datasheet seeds in I_c)

### AC-2: Clock rate confirmed
- `clock_rate` measured from CCE kernel cycle counter or msprof metadata
- Confirmed value within ±1% of 1850 MHz, OR documented deviation stored in `notes` field

### AC-3: Mandatory handoff isolation
- Linear fit T(K) = α + β·K across K ∈ {64, 128, 256, 512, 1024, 2048}
- R² > 0.99 on the fit
- `mandatory_handoff_cost` = α (intercept in cycles), stored as a `CalibrationConstant`

### AC-4: Conservative soundness check (tilesim anchor)
- For each P0 Cube constant: `measured_910B3 ≤ tilesim_910B4 × (1850/1650) × 1.05`
  - If violated: flag in validation report, re-measure before populating `calib_910b3_v1.json`
- For each P0 BW constant (single-core): within 2× of 910B4 single-core value (any larger difference requires explanation in notes)
- `validate_vs_tilesim.py` exits 0 when all checks pass; non-zero exit blocks Step 5

### AC-5: Data completeness
- `calib_910b3_v1.json` loads without error via `CalibrationDB.load()`
- All 10 P0 constants have `value > 0`
- All 4 BW path tables (`gm→ub`, `ub→gm`, `gm→l1`, `l1→l0a`) have ≥ 5 core_num data points
- `bandwidth_910b3.csv` matches tilesim CSV schema (src_mem, dst_mem, core_num, pkt_size, mode, bandwidth(GB/s))

### AC-6: Model integration
- `T_core_floor` is non-zero for a synthetic Cube-bound extract after loading `calib_910b3_v1.json`
- `python -m pytest tests/perfbound/ -v` passes (all existing + new calibration load test)
- New test `test_calib_db_loads_without_zero_p0` passes:
  ```python
  db = CalibrationDB.load("perfbound/calibration/data/calib_910b3_v1.json")
  assert db.cube.peak_fp16.value > 0
  assert db.cube.peak_fp16.n_runs >= 30
  assert db.cube.peak_fp16.ci_95 / db.cube.peak_fp16.value < 0.05
  ```

### AC-7: Reproducibility
- `perfbound/calibration/microbench/README.md` documents: toolchain version, compile command, run command, expected output format
- `run_benchmarks.sh` is idempotent: re-running overwrites output CSVs, does not corrupt state
- A second engineer following `README.md` can reproduce all P0 constants within 10% on the same 910B3 unit

---

## 3. Cross-Validation Against Tilesim (No Hardware Required)

Tilesim has **no 910B3 config** — 910B3 is the only 20 AIC/40 AIV chip without
one. However, two tilesim constants can serve as **validation anchors**:

| What | Tilesim source | Expected relationship |
|------|---------------|----------------------|
| `vec_cycle` intrinsic table | `arc_config/910B1/vec_cycle_910B1.csv` | 910B3 shares the same Vector ISA generation as 910B1/910B4; intrinsic latencies (computing_cycles, head_cycles, interval_cycles) should match ±10% after clock-rate normalization |
| Cube throughput ratios | 910B4: FP16=271.4 TFLOPS, 20 AIC @ 1650 MHz | 910B3 @ 1850 MHz should scale: `271.4 × 1850/1650 ≈ 304 TFLOPS`. Measured value should be ≤ this (bound is conservative, any excess = measurement artifact) |
| BW curve shape | `bandwidth_910B4.csv` | Relative shape (single-core peak, multi-core saturation) should match qualitatively; absolute values differ |

**Validation rule**: if a measured 910B3 `I_c` exceeds the clock-scaled 910B4
value by > 5%, treat it as a measurement error and re-run. Measured rates ≤
clock-scaled 910B4 are conservative and accepted.

---

## 4. File Structure (Deliverables)

```
perfbound/calibration/
├── data/
│   ├── calib_910b3_v1.json          # Populated CalibrationDB JSON
│   ├── bandwidth_910b3.csv          # BW curve table (tilesim format)
│   └── vec_cycle_910b3.csv          # Vector intrinsic cycle table
├── microbench/
│   ├── README.md                    # How to compile and run on remote 910B3
│   ├── cube_peak_fp16.cce           # Cube FP16 peak benchmark
│   ├── cube_peak_int8.cce           # Cube INT8 peak benchmark
│   ├── cube_peak_bf16.cce           # Cube BF16 peak benchmark
│   ├── vector_peak_elemwise_add.cce # Vector ADD throughput
│   ├── vector_peak_elemwise_mul.cce # Vector MUL throughput
│   ├── vector_peak_elemwise_max.cce # Vector MAX throughput
│   ├── vector_peak_elemwise_min.cce # Vector MIN throughput
│   ├── vector_peak_transcendental.cce # EXP/LOG/SQRT/RSQRT
│   ├── mte_gm_to_ub.cce             # GM→UB bandwidth sweep
│   ├── mte_ub_to_gm.cce             # UB→GM bandwidth sweep
│   ├── mte_gm_to_l1.cce             # GM→L1 bandwidth sweep
│   ├── mte_l1_to_l0a.cce            # L1→L0A bandwidth sweep
│   └── mandatory_handoff.cce        # Cube→Vector handoff isolation
├── scripts/
│   ├── run_benchmarks.sh            # Orchestrate all benchmarks on remote 910B3
│   ├── fit_constants.py             # Extract constants from msprof CSV output
│   └── validate_vs_tilesim.py       # Cross-validation script
└── __init__.py  (already exists)
configs/
└── ascend_910b3.json                # Hardware config (scaffold from 910B4)
```

---

## 5. CCE Microbenchmark Design

**Language**: AscendC (`kernel_operator.h`). See memory note: CCE is the
appropriate language for Ascend micro-benchmarks.

### 4.1 Cube Peak (`cube_peak_fp16.cce`)

**Goal**: Achieve maximum sustained Cube FP16 throughput by keeping L0A/L0B
pre-loaded (GM→L1→L0A/B in double-buffer), and measuring only Cube
execution cycles over K iterations.

```
Template:
  - M=128, N=128, K=4096 (large K to amortize startup)
  - Outer loop: repeat 30× to fill msprof time window
  - Inner loop: MATMUL(L0A[M×k], L0B[k×N]) → L0C[M×N], k=32 per step
  - DMA pre-fetch ahead of Cube to hide memory latency
  - Measure: msprof op_summary → cube_time_us
  - I_c = (2 × M × N × K) / cube_time_us  [FLOPS/μs = TFLOPS × 10⁶]
```

**Repeat for**: INT8 (2× FLOPS per cycle vs FP16), BF16 (same as FP16), FP32.

### 4.2 Vector Peak (`vector_peak_elemwise_{add,mul,max,min}.cce`)

**Goal**: Max sustained Vector throughput for each op class.

```
Template (per op):
  - Buffer: 256 elements × FP16 = 512B (fits UB easily)
  - Loop count: N_iter = 10000 repeats
  - Op: VADD(UB[256], UB[256]) → UB[256], then measure
  - Measure: msprof vector_time_us
  - I_c = (N_iter × 256) / vector_time_us  [elements/μs]
```

**Ops to sweep**: VADD, VMUL, VMAX, VMIN, VEXP, VLOG, VSQRT, VRSQRT.
Use FP16 as primary precision; add FP32 for transcendentals.

### 4.3 GM→UB Bandwidth (`mte_gm_to_ub.cce`)

**Goal**: Sustained bandwidth for aligned large transfers, swept over core_num.

```
Template:
  - Transfer size: 256 KiB (large enough to overflow L2 over repeated transfers)
  - Source: GM buffer (pre-allocated, cache-cold each iter via stride pattern)
  - Dest: UB
  - Loop: 2048 transfers per measurement (discard warmup in extraction)
  - Vary: run with 1, 2, 4, 8, 12, 16, 20 cores active
  - Measure per-core BW = transfer_bytes / mte_time_us
```

**Run also for**: UB→GM (store direction), GM→L1, L1→L0A.

### 4.4 Mandatory Handoff Cost (`mandatory_handoff.cce`)

**Goal**: Isolate the pipeline serialization overhead at Cube→Vector boundary.

```
Template:
  - Kernel: Cube matmul (M×N×K) → stores L0C to GM → Vector reads from GM
  - Vary K: [128, 256, 384, 512, 1024, 2048]
  - Fit: T_total(K) = α + β × K
    α = mandatory_handoff_cost (cycles)
    β = per-K compute slope
  - Acceptance: R² > 0.99 for the linear fit
```

---

## 6. Implementation Steps

### Step 0 — Config scaffold (no hardware)

**File**: `configs/ascend_910b3.json`  
Clone from `configs/ascend_910b.json`, update:
- `clock_freq: 1850` (confirm vs 910B4's 1650 — 910B3 is the higher-clocked variant)
- `cube_core_num: 20`, `vec_core_num: 40`, `cv_bind: true`
- `bandwidth: "bandwidth_910b3.csv"` (placeholder until Step 5)
- `vec_intrinsics: "vec_cycle_910b3.csv"` (placeholder until Step 4)

Wire `perfbound/extract/grid_idioms.py` capacity checks to read from this config
instead of hardcoded 256 KB / 1 MB. (`grid_idioms.py:33`, `grid_idioms.py:66`)

**File**: `perfbound/calibration/data/vec_cycle_910b3.csv`  
Seed with 910B1's table (`tilesim/core/config/arc_config/910B1/vec_cycle_910B1.csv`)
as an initial estimate. Mark every row's `source` as `"tilesim_910B1_seed"` in the
CalibrationDB until 910B3-measured values replace them.

**Acceptance**: `python -m pytest tests/perfbound/ -v` still 11/11 passing after
wiring the config reads.

---

### Step 1 — CCE kernel suite

Write all kernels in `perfbound/calibration/microbench/`. Each kernel:
- Includes a header comment with: expected output metric, formula for extraction,
  and the msprof field to read
- Runs ≥ 30 iterations internally (warm-up 5, measurement 30, discard top/bottom 5%)
- Writes a `RESULT:` line to stdout for the collection script to parse

**Kernels**: 10 files as listed in §3.

**Acceptance**: kernels compile without error on 910B3 CCE toolchain.

---

### Step 2 — Run script

**File**: `perfbound/calibration/scripts/run_benchmarks.sh`

```bash
#!/usr/bin/env bash
# Runs all CCE microbench kernels on remote 910B3 server.
# Usage: REMOTE_HOST=user@910b3-server bash run_benchmarks.sh [output_dir]
# Output: one CSV per kernel in output_dir/
KERNELS=(cube_peak_fp16 cube_peak_int8 cube_peak_bf16
         vector_peak_elemwise_add vector_peak_elemwise_mul
         vector_peak_elemwise_max vector_peak_elemwise_min
         vector_peak_transcendental
         mte_gm_to_ub mte_ub_to_gm mte_gm_to_l1 mte_l1_to_l0a
         mandatory_handoff)
```

Each kernel: run through `cce_remote_bench.py` with 45 raw repeats so MTE trimming retains at least 30 samples.

**Acceptance**: all kernels produce output CSVs; no kernel exits with error.

---

### Step 3 — Constant extraction

**File**: `perfbound/calibration/scripts/fit_constants.py`

Reads msprof op_summary CSVs, computes:
- Compute I_c: `mean(steady_state_runs)`, `ci_95 = 1.96 × std / sqrt(n)`
- BW curves: per-(src,dst,core_num) mean BW; output `bandwidth_910b3.csv`
- Handoff cost: `scipy.stats.linregress(K_values, T_values)` → intercept = cost
- CV check: reject any constant with CV > 5% and log a warning

Output: `calib_910b3_v1.json` with all `CalibrationConstant` entries.

**Acceptance**:
- All P0 constants present with `n_runs ≥ 30`
- All P0 CV < 5%
- `mandatory_handoff_cost` linear fit R² > 0.99

---

### Step 4 — Tilesim cross-validation

**File**: `perfbound/calibration/scripts/validate_vs_tilesim.py`

Checks:
1. **vec_cycle**: for each intrinsic in `vec_cycle_910b3.csv`, compare
   `computing_cycles` to `vec_cycle_910B1.csv` after clock normalization
   (× 1650/1850 for 910B3→910B4 scale). Flag if delta > 10%.
2. **Cube throughput**: `measured_910B3_fp16_TFLOPS ≤ clock_scaled_910B4_fp16_TFLOPS × 1.05`.
   (5% margin for measurement noise; > 5% = re-measure.)
3. **BW ratios**: `BW_910B3_gm_ub_1core / BW_910B4_gm_ub_1core` — flag if > 1.3×
   (unexpected; would need investigation).

Output: printed validation report; non-zero exit if any P0 constant fails.

**Acceptance**: all P0 validation checks pass (or produce a flagged report for
human review when expected differences are explained).

---

### Step 5 — Load into CalibrationDB

Update `perfbound/calibration/constants.py` or a new
`perfbound/calibration/calib_loader.py` so that:
```python
db = CalibrationDB.load("perfbound/calibration/data/calib_910b3_v1.json")
assert db.cube.peak_fp16.value > 0
```

Update `perfbound/model/component_model.py` and `bandwidth.py` to use these
values (currently `I_c = 0.0` stubs).

Add test:
```python
# tests/perfbound/test_calibration_load.py
def test_calib_db_loads_without_zero_p0():
    db = CalibrationDB.load("perfbound/calibration/data/calib_910b3_v1.json")
    assert db.cube.peak_fp16.value > 0
    assert db.cube.peak_fp16.n_runs >= 30
    assert db.cube.peak_fp16.ci_95 / db.cube.peak_fp16.value < 0.05
```

**Acceptance**: `python -m pytest tests/perfbound/ -v` all passing including new
calibration load test.

---

## 7. Dependency Map

```
Step 0 (config scaffold)     ←── no hardware, do now
    │
    ▼
Step 1 (CCE kernels)         ←── CCE toolchain on dev machine to compile-check
    │
    ▼
Step 2 (run on 910B3)        ←── requires remote 910B3 access
    │
    ▼
Step 3 (fit constants)       ←── requires Step 2 output CSVs
    │
    ├──▶ Step 4 (tilesim validate)   ←── parallel with Step 5
    │
    ▼
Step 5 (load into model)     ←── unblocks T_core_floor > 0
```

Steps 0 and 1 can proceed immediately without hardware.
Steps 2–5 require remote 910B3 access.

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| 910B3 clock not 1850 MHz | All cube I_c values wrong | Step 0 adds clock verification as first CCE kernel output |
| CCE toolchain version mismatch on remote | Kernels fail to compile | Pin toolchain version in `run_benchmarks.sh`; document in `README.md` |
| Measured `P_cube[fp16]` > clock-scaled 910B4 value | Unsound bound if used | Validation script (Step 4) flags this; cap at 910B4 clock-scaled value if >5% over |
| CV > 5% on BW constants | Noisy calibration | Increase n_repeat to 100; check for thermal throttling; run at steady-state temp |
| mandatory_handoff R² < 0.99 | Handoff cost unreliable | Add NOP padding between Cube/Vector stages to isolate the transition; try different K range |
| Remote server unavailable | Steps 2–5 blocked | Keep model runnable with `I_c = 0.0` stubs (current behavior); partial progress on Steps 0–1 |

---

## 9. Not in A.1 Scope

- `L2_residency_bytes` (P2, deferred per spec)
- `η_alignment(stride)` (P1, deferred to A.7 tightness work)
- Calibration for other 910B variants (B1, B4) — not needed for this paper
- Multi-card / HCCL bandwidth (out of scope for single-kernel model)

---

## 10. Definition of Done

- [ ] `configs/ascend_910b3.json` created
- [ ] 10 CCE benchmark kernels written and compile-checked
- [ ] All kernels run on remote 910B3 (≥30 runs each)
- [ ] All P0 constants: CV < 5%, CI populated
- [ ] `calib_910b3_v1.json` written with full provenance
- [ ] `bandwidth_910b3.csv` populated (matches tilesim CSV schema)
- [ ] `vec_cycle_910b3.csv` measured or validated from tilesim seed
- [ ] Tilesim cross-validation passes (or flags are explained)
- [ ] `T_core_floor` is non-zero for a real Cube workload
- [ ] `python -m pytest tests/perfbound/ -v` all passing
