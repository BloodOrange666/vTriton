# A.1 Remediation Plan — Post-Review Fixes

**Date**: 2026-06-07  
**Source**: Code review of A.1 implementation (bench_output/ + microbench kernels)  
**Language**: Migrated to **AscendC** (`kernel_operator.h`) — CCE reference too sparse (decided 2026-06-07)  
**Status**: Remediation complete; full 910B3 rerun passed on 2026-06-07

## Completion Evidence

- Bug A fixed with two launcher calls for MTE kernels: an unprofiled warmup launch
  uses `--mte-start 0 --mte-iters 768`, and the profiled msprof wrapper uses
  `--mte-start 768 --mte-iters 1280`.
- Bug B fixed in `UbToGm()` by advancing the GM source window with
  `srcGm[i * kMteElements + offset]`.
- Bug C fixed with one mixed `mandatory_handoff` kernel. Generated remote CANN
  metadata now emits both `mandatory_handoff_mix_aic_section` and
  `mandatory_handoff_mix_aiv_section` with `F_TYPE_MIX_TASK_RATION = 1,1`.
- Bug D fixed by replacing the duplicate `K=64` point with `K=384`.
- Full remote verification:
  - `python3 perfbound/calibration/scripts/cce_remote_bench.py --host 910B3 --skip-direct-compile --n-repeat 45 ... --output-dir /tmp/vtriton_a1_full_20260607`
  - Result: `Kernels compiled: 13/13`, `Benchmarks run: 18/18`, `CSV files synced: 18`.
- Extraction:
  - `python3 perfbound/calibration/scripts/fit_constants.py perfbound/calibration/bench_output perfbound/calibration/data/calib_910b3_v1.json`
  - Result: all P0 constants pass `n_runs>=30`, `CV<5%`; handoff fit `R²=0.9998`.
- Validation:
  - `python3 perfbound/calibration/scripts/validate_vs_tilesim.py perfbound/calibration/data/calib_910b3_v1.json`
  - Result: all validation checks passed; bandwidth ratio warning remains informational.

---

## Blocking Fixes

### Fix 1 — BW kernels: buffer must overflow L2 (CRITICAL — soundness)

**Root cause**: srcGm buffer = 1.6 MB << L2 = 192 MB. Reads are L2-served → 400+ GB/s (implausible). Real HBM expected ~70–90 GB/s.

**Fix in `vt_microbench_common.h`**:
```cpp
// OLD
constexpr uint32_t kMteElements = 8192;   // 16 KB per transfer
constexpr uint32_t kMteRepeat   = 100;    // total = 1.6 MB (fits in L2)

// NEW
constexpr uint32_t kMteElements = 131072; // 256 KB per transfer (near UB limit)
constexpr uint32_t kMteRepeat   = 2048;   // total = 512 MB (> L2 = 192 MB)
constexpr uint32_t kMteWarmup   = 768;    // first 192 MB: L2 flush phase (skipped)
constexpr uint32_t kMteMeasured = 1280;   // last 320 MB: cold HBM reads (measured)
```

Launcher/run structure:
```cpp
// Kernel receives a start/count window.
op.Init(src, dst, startIter, iterCount);
```

The runner enforces the split:
```bash
# Warmup outside msprof
vt_a1_bench_launcher --kernel mte_* --repeat 1 --mte-start 0 --mte-iters 768

# Measured inside msprof
vt_a1_bench_launcher --kernel mte_* --repeat 45 --mte-start 768 --mte-iters 1280
```

`fit_constants.py` formula update:
```python
transfer_bytes = 131072 * 2   # 256 KB × sizeof(half)
N_measured     = 1280          # cold HBM phase only
total_bytes    = transfer_bytes * N_measured
```

**Expected outcome**: BW drops from ~400 GB/s to ~70–90 GB/s (HBM range). Soundness restored.

---

### Fix 2 — `mandatory_handoff` kernel: implement actual Cube→Vector handoff

**Root cause**: `MandatoryHandoffKernel` is AIV-only vector Add. No Cube, no K-sweep, no per-K CSV files. slope=0 with R²=1 is an artifact of all K values using the same CSV.

**Fix**: Mixed AIC+AIV kernel (`KERNEL_TYPE_MIX_AIC_1_1`):
1. AIC block 0: `Mmad(L0A[128×K], L0B[K×128]) → L0C` + `Fixpipe(L0C → cGm)`
2. `SyncAll()` — the handoff barrier
3. AIV block 0: `DataCopy(cGm → UB)` + minimal vector op

Kernel signature:
```cpp
extern "C" __global__ __aicore__ void mandatory_handoff(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t K)
```

Run script adds per-K invocations:
```bash
for K in 128 256 384 512 1024 2048; do
    python3 perfbound/calibration/scripts/cce_remote_bench.py \
        --host 910B3 --kernels mandatory_handoff \
        --mandatory-k-values "${K}" --skip-direct-compile
done
```

`fit_constants.py` already handles per-K CSVs — no change needed there.

**Expected outcome**: Linear T(K) = α + β·K with meaningful β. R² > 0.99 with non-zero slope becomes a real quality check. α = true handoff cost (expected ~5000–20000 cycles).

---

### Fix 3 — `validate_vs_tilesim.py`: correct 910B4 reference BW

**File**: `perfbound/calibration/scripts/validate_vs_tilesim.py:60`

```python
# OLD — wrong datasheet guess, makes ratio check too permissive
BW_910B4_GM_UB_GBPS = 200.0  # Datasheet peak (sustained will be lower)

# NEW — actual tilesim measured value (bandwidth_910B4.csv, GM,UB,core_num=1)
BW_910B4_GM_UB_GBPS = 64.36  # Tilesim sustained measured (bandwidth_910B4.csv)
```

After Fix 1, measured 910B3 BW should be ~70–90 GB/s → ratio ≈ 1.1–1.4. The 1.3× threshold is appropriate.

---

### Fix 4 — `extract_mte_bandwidth`: flip trimming direction

**File**: `perfbound/calibration/scripts/fit_constants.py:249–250`

```python
# OLD — keeps slowest 2/3 (longest durations = lowest BW, backwards)
durations = sorted_durations[len(sorted_durations) // 3:]

# NEW — keeps fastest 2/3 (drops slow cold-start outliers, correct)
durations = sorted_durations[:2 * len(sorted_durations) // 3]
```

Also fix n_runs to report actual sample count used:
```python
# OLD
n_runs=len(raw_durations),    # misleading: reports 30, computes from 20
# NEW
n_runs=len(durations),         # honest: count of samples used in mean
```

---

## Minor Fixes (bundle with above)

- **Vector op isolation**: Split `VectorElemwiseKernel` into 4 separate kernels (one per op: Add, Mul, Max, Min) so individual `P_vector_*` values are independently measured, not all identical fallback values.
- **AC-1 test threshold**: Tighten `test_calib_db_loads_without_zero_p0` to `ci_95/value < 0.025` (matches AC-1's 2.5% criterion; currently 5%).
- **A.1 plan §5 language note**: Replace "CCE (not AscendC, not hand-HIVM)" with "AscendC (`kernel_operator.h`)".

---

## Execution Order

```
Fix 3 — one-liner, no hardware         ←── now
Fix 4 — one-liner, no hardware         ←── now
Fix 1 — kernel change, compile-check locally, re-run on 910B3
Fix 2 — kernel rewrite + per-K script, re-run on 910B3
Minor fixes — bundle with Fix 1/2
Re-run validate_vs_tilesim.py          ←── after new bench_output CSVs
```

---

## Open Item — L2 Cache Bandwidth Model (future work)

**Status**: Deferred — tracked here, not blocking A.1.

**Background**: Fixing Fix 1 measures pure HBM bandwidth. However real kernels have partial L2 reuse depending on:
- Tile working set size relative to L2 (192 MB on 910B3)
- Access pattern (sequential vs strided vs random)
- Number of active cores sharing L2

Using only HBM bandwidth is conservative (lower bound on BW → higher T_mte → conservative T_bound). But for tightness, if a kernel's working set fits in L2, using HBM BW overstates T_mte.

**Planned model**:
- Predict L2 cache hit rate `h ∈ [0, 1]` from tile working set size and access pattern
- Blended effective bandwidth: `BW_eff = h × BW_l2 + (1 - h) × BW_hbm`
- BW_l2 already appears in `configs/ascend_910b3.json` (bandwidth_gbps = 3200 GB/s for L2→UB path)
- Model `h` as a function of `working_set_bytes / L2_size_bytes`, clipped to [0, 1]

**Where it fits**:
- Adds a new calibration constant: `BW_l2_sustained` (measured separately with small-buffer benchmark)
- Adds `predict_l2_hit_rate(working_set_bytes, access_pattern)` helper to `perfbound/model/bandwidth.py`
- Changes `lookup_bw()` from a table lookup to `BW_eff(h)` computation
- Validation: compare predicted `BW_eff` vs tilesim's measured per-core curves (which implicitly include L2 effects)

**Dependency**: Requires Fix 1 (clean HBM measurement) as the lower anchor. BW_l2 measured with a buffer << L2 (the current-style small-buffer bench, repurposed).

**Deferred to**: A.5 or A.7 tightness pass (not A.1 scope).
