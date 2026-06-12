# Author Headroom Gap Analysis: Decomposing the T_measured − T_bound Gap

**Status**: Research complete (v2 — evidence-led rewrite), spec for implementation
**Date**: 2026-06-12
**Kernel**: `chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2` (bf16 fused attention backward)
**Hardware**: Ascend 910B3, 20 AI Cores

> **v2 note.** The first draft (v1) decomposed author headroom into five new gap
> dimensions (HBM contention, AIC-AIV dequant serialization, instruction overlap,
> hardware/cache residual, wave quantization) whose magnitudes were imported from
> W4A16 / MatMul literature. When those estimates are checked against the data
> that is *already present* in the msprof CSV and the DES graph, **four of the
> five are refuted for this kernel** and the dominant cause — unmodeled **scalar**
> cost — was missing entirely. v2 rewrites §0–§6 around what the two inputs
> actually show. The prototype that produced these numbers is
> `scripts/component_attribution_prototype.py`.

---

## 0. Problem Statement

The existing 5-gap model (grid, Gap-1 … Gap-4) attributes performance loss
*within* T_bound. For chunk_kda:

```
T_bound    = 46,110 µs   (Tier-2 DES structural bound)
T_measured = 104,326 µs  (msprof on 910B3)
Author headroom = 58,216 µs   (55.8% of T_measured)
```

The 5-gap attribution covers only 2.73% of T_bound, so the headroom looks like a
monolithic unknown. **It is not.** The msprof op-summary CSV carries per-engine
busy ratios, and the DES graph carries per-pipe structural busy time. Read
together they give a direct, quantified answer: **the kernel is scalar-bound, and
the model does not price scalar work.**

**Question answered:** *Can author headroom be analyzed from msprof + HIVM IR?*
**Yes** — for this kernel class it decomposes cleanly, and it is dominated by a
single, fixable cause rather than a spread of microarchitectural effects.

---

## 1. Key Finding: The Headroom Is Unmodeled Scalar Cost

Two independent inputs are read separately and **agree on the dominant engine**:

| Input | Signal | Scalar share |
|-------|--------|--------------|
| msprof (measured) | `aiv_scalar_ratio` × aiv_time + `aic_scalar_ratio` × aic_time | **91.6% of T_measured** (95,525 µs) |
| HIVM/DES (structural) | Σ `duration·loop_multiplier` over `PIPE_S` ops | **72.7% of structural busy** (597 ops) |

Measured scalar time alone (**95,525 µs**) is **2.07× the entire bound**
(46,110 µs). No combination of memory/compute-engine effects can explain the gap
when scalar is 9/10ths of the measured execution.

**Why the bound misses it.** `compute_component_floor` models the scalar pipe at
the *full vector throughput* as a deliberate soundness fallback, because scalar
was never directly calibrated:

- `perfbound/model/component_model.py:290` — *"Without real Scalar calibration,
  use Vector/20 as a conservative proxy"* — and the proxy is further relaxed to
  the full Vector rate at:
- `perfbound/calibration/constants.py:252` (`get_scalar_throughput_ops_per_us`) —
  returns the measured Vector rate unless `scalar_throughput_measured` is True.

Modeling scalar at vector SIMD speed makes the scalar floor ~two orders of
magnitude too low, so the bound binds on **vector** (per §2.1) and the real scalar
cost reappears downstream as "author headroom."

**Consequence for the taxonomy.** Most of this 58,216 µs is **model headroom**,
not author headroom: it is closed by *calibrating the scalar pipe and adding a
scalar floor to the bound* (the still-open **US-SB-007**), which raises T_bound
toward T_measured. It is not closed by the kernel author writing different Triton.
Calling the whole 55.8% "author headroom" mislabels the cause.

---

## 2. Existing Analysis Results (End-to-End Run)

### 2.1 Bound Analysis

| Metric | Value |
|--------|-------|
| T_bound (Tier-2 DES) | 46,109.91 µs |
| T_bound_HIVM (idealized) | 46,064.92 µs |
| T_measured | 104,326.00 µs |
| Tightness | 2.26× |
| Soundness | PASS ✓ (T_bound ≤ T_measured) |
| Binding tier | component |
| Binding component | vector  ← *should be scalar; see §1* |
| T_grid_floor | 46,064.92 µs |
| T_core_floor | 46,109.91 µs |
| T_serial_irreducible | 0.0000 µs |

### 2.2 Component Attribution (NEW — from msprof + DES, not estimated)

Produced by `scripts/component_attribution_prototype.py`. Engine-time =
(core busy time) × (that engine's msprof ratio).

**[1] msprof MEASURED engine-time** (where the hardware spent time):

| Engine | µs | % of core busy |
|--------|-----|----------------|
| **scalar (AIV)** | **87,935** | **84.5%** |
| scalar (AIC) | 7,590 | 7.3% |
| vector (AIV) | 4,579 | 4.4% |
| mte2-load (AIC) | 2,703 | 2.6% |
| fixpipe (AIC) | 2,183 | 2.1% |
| mte2-load (AIV) | 1,977 | 1.9% |
| mte3-store (AIV) | 1,145 | 1.1% |
| mte1 (AIC) | 832 | 0.8% |
| mac-cube (AIC) | 624 | 0.6% |
| **scalar total** | **95,525** | **91.6% of T_measured** |

Side facts: `aic_icache_miss_rate = aiv_icache_miss_rate = 0.0` (not a cache
problem); `cube_utilization = 99.6%` while `aic_mac_ratio = 0.006` (cube is
*occupied but stalled* — waiting, not computing).

**[2] HIVM STRUCTURAL busy per PIPE** (where the model puts time):

| Pipe | n ops | µs | % of structural busy |
|------|------:|-----:|---------------------:|
| **PIPE_S (scalar)** | 597 | 128.0 | **72.7%** |
| PIPE_V (vector) | 254 | 17.5 | 9.9% |
| PIPE_MTE2_V | 62 | 9.6 | 5.5% |
| PIPE_MTE3 | 70 | 6.8 | 3.9% |
| PIPE_FIX | 41 | 6.4 | 3.6% |
| PIPE_ALL | 54 | 3.0 | 1.7% |
| PIPE_MTE2_C | 59 | 2.8 | 1.6% |
| PIPE_M (cube) | 32 | 1.2 | 0.7% |
| PIPE_MTE1 | 19 | 0.6 | 0.4% |
| PIPE_UNKNOWN | 190 | 0.0 | 0.0% |

Both views rank scalar #1 by a wide margin. The structural share (72.7%) is lower
than measured (91.6%) precisely because the model under-prices each scalar op
(`duration` is often 1 cycle) — the residual is the scalar-calibration gap.

### 2.3 Two-Limit Analysis

| Limit | Value (µs) | % of T_measured |
|-------|-----------|-----------------|
| T_bound_HIVM | 46,064.92 | 44.2% |
| T_bound_DSL | 46,109.91 | 44.2% |
| T_measured | 104,326.00 | 100% |
| Compiler headroom | 44.99 | 0.04% |
| Author headroom | 58,216.09 | 55.8% |

### 2.4 DES Graph Summary

Total ops 1,378. Pipe counts: PIPE_S 597, PIPE_V 254, PIPE_UNKNOWN 190,
PIPE_MTE3 70, PIPE_MTE2_V 62, PIPE_MTE2_C 59, PIPE_ALL 54, PIPE_FIX 41,
PIPE_M 32, PIPE_MTE1 19. Ops with repeat>1: 271 (min 8, max 64, median 32).
`schedule_truncated = False`.

---

## 3. Headroom Decomposition (evidence-led)

Gaps are ordered by what the data supports, with v1's literature gaps demoted to
their measured magnitude.

### Gap-S: Unmodeled Scalar Cost — **DOMINANT**

**Evidence**: msprof `scalar` = 91.6% of T_measured (95,525 µs); DES `PIPE_S` =
72.7% of structural busy (597 ops). Bound models scalar at vector rate
(component_model.py:290, constants.py:252).

**What it measures**: the gap between real scalar-pipe time and the bound's
(near-zero) scalar floor. Scalar ops here are address arithmetic, loop/index
control, and per-tile setup that the AIV scalar unit executes serially.

**How to extract** (no new hardware): scalar floor from DES =
Σ_{PIPE_S} cycles · *calibrated* scalar cost; compare to msprof scalar engine-time
to validate. The fix is **US-SB-007**: replace the vector-rate fallback with a
measured CCE scalar rate, set `scalar_throughput_measured = True`, and let the
scalar floor enter `max(...)` in the core floor.

**Magnitude**: ~50,000–60,000 µs — i.e. essentially the entire author headroom.
Closing it tightens the bound, it does not change the kernel.

### Gap-E: Wave Quantization — small, structural, belongs in the grid floor

**Evidence**: grid = 4096 programs / 20 cores ⇒ 205 waves, last wave 16/20.
`grid_model.py` uses `occupancy = min(G,n_cores)/n_cores = 1.0` for G≫cores, so it
captures the average but not the last-wave tail.

**Magnitude (its own formula)**: `(1 − 16/20)·T_core_floor / n_waves =
0.2·46,110/205 ≈ 45 µs`. (v1 estimated 500–1,500 µs — ~10–30× its own formula;
corrected here.) This belongs **inside** T_grid_floor, not in author headroom.

### Gap-C: Instruction Overlap — plausible, needs calibration

**Evidence**: Gap-3 (avoidable serialization) is a conservative upper bound; real
hardware overlaps independent sub-ops. Genuine, but **not** derivable from
msprof-aggregate + HIVM alone — it needs per-pipe overlap factors from
microbenchmarks. Keep as a calibration item, not a quantified headroom bucket.

### Gap-A: HBM Bandwidth Contention — **refuted for this kernel**

v1 estimate 9,000–17,000 µs (15–30%). Measured total MTE activity here is
`aiv_mte2 0.019 + aic_mte2 0.026 + mte3 0.011 + mte1 0.008 ≈ 4.7%` (~7,600 µs).
Contention cannot be 15–30% of headroom when *all* memory-engine activity is
~4.7%. The 20–40% figure is from MatMul+weight-load (W4A16) kernels; chunk_kda is
scalar-bound. **Drop for this kernel class**; revisit only for memory-bound MIX
kernels.

### Gap-B: AIC-AIV Pipeline Serialization — **refuted for this kernel**

v1 estimate 6,000–12,000 µs. Premised on cube→HBM→vector dequant round-trips. But
`aic_mac_ratio = 0.006` — there is almost no cube MatMul, and `cube_utilization
99.6%` with mac 0.6% means the cube is *idle-but-allocated*, not serialized
through dequant. The W4A16 mechanism does not apply to this bf16 kernel.
Structural PIPE_M→PIPE_V chain *counts* may still be reported, but no µs estimate
is defensible here. **Drop for this kernel class.**

### Gap-D: Hardware-Level Residual — keep, but small and renamed

Honest catch-all = `T_measured − T_bound(after Gap-S) − Gap-C − Gap-E`. Note
`icache_miss_rate = 0.0` directly refutes the "cache behavior" content named in
v1's rationale. Once Gap-S is priced into the bound, this residual is expected to
be small.

---

## 4. Estimated Headroom Decomposition (corrected)

```
T_measured (104,326 µs)
├── T_bound (46,110 µs)  ── structural bound (currently mis-binds on vector)
│     ├── gap4_intra_unit (1,185 µs = 2.57%)
│     ├── gap1_wrong_unit (72 µs = 0.16%)
│     └── model core (44,853 µs)
│
└── Author headroom (58,216 µs = 55.8%)
      ├── Gap-S  Unmodeled scalar cost      ~50,000–60,000 µs   (DOMINANT; = US-SB-007)
      ├── Gap-C  Instruction overlap         (needs microbench calibration)
      ├── Gap-D  Hardware residual           small once Gap-S is priced
      ├── Gap-E  Wave quantization           ~45 µs (belongs in grid floor)
      ├── Gap-A  HBM contention              refuted (~7,600 µs total MTE; not 9–17k)
      └── Gap-B  AIC-AIV dequant serial      refuted (aic_mac 0.6%; mechanism absent)
```

The honest one-line decomposition: **author headroom ≈ Gap-S (unmodeled scalar) +
a small residual.** Everything else is either tiny, refuted, or belongs in the
bound.

---

## 5. Data Availability Matrix (corrected)

| Data | Available? | Location |
|------|-----------|----------|
| Grid dims + n_cores | ✅ | grid=(128,32), n_cores=20 |
| DES dependency edges | ✅ | kda_des.json `depends_on` |
| Pipe assignment per op | ✅ | kda_des.json `pipe` (PIPE_S = 597) |
| Per-pipe structural busy | ✅ | Σ `duration·loop_multiplier` |
| **msprof per-engine ratios** | ✅ | `aiv_scalar_ratio`, `aiv_vec_ratio`, `aic_*_ratio` — **populated for the MIX_AIC kernel row** |
| **msprof icache miss rate** | ✅ | `aic/aiv_icache_miss_rate` (= 0.0 here) |
| Cube utilization | ✅ | `cube_utilization(%)` |
| Per-core execution timeline | ❌ | Not in msprof aggregate |
| Cache miss *penalty* (cycles) | ❌ | Only miss-rate exposed |
| Per-wave timing | ❌ | Not in msprof aggregate |

> **Correction to v1.** v1 stated msprof "only extracts aggregate task duration."
> That describes the *current parser*, not the data: the CSV has per-engine ratios
> and icache rates. The limitation is in parsing, not availability.

> **Scope caveat (updated after implementation).** Per-engine ratios are
> populated for the chunk_kda **MIX_AIC** row *and* for the **AI_VECTOR_CORE**
> rows of softmax / layernorm / rmsnorm. The method generalises across both task
> types; `parse_engine_attribution` returns `populated=False` only when the
> matched row genuinely lacks ratio columns (e.g. vector_add's captured CSV has
> no populated AI-core row). A per-row populated-field check is built in.
>
> **Scalar dominance is pervasive, not chunk_kda-specific.** Measured scalar
> share across the committed fixtures: chunk_kda **0.85**, rmsnorm **0.69**,
> softmax **0.51**, layernorm **0.47** — scalar is the dominant engine in every
> populated kernel. This makes **US-SB-007 (scalar calibration) the single
> highest-leverage model fix for the entire validation set**, not a one-kernel
> patch.

---

## 6. Implementation Roadmap (reprioritized)

| Phase | Component | Effort | Why |
|-------|-----------|--------|-----|
| **P0** | **US-SB-007 scalar calibration → scalar floor in bound** | (HW) | Closes ~the entire headroom; both inputs agree it is the cause |
| **P0** | Promote `component_attribution_prototype.py` → `_attribute_by_component` in `bound_combiner.py`; surface in `report.py` | 3 hr | Turns headroom into a per-engine table from real data |
| **P1** | Fix binding-component reporting so scalar can bind once calibrated | 1 hr | §2.1 currently mis-binds vector |
| **P2** | Gap-E wave-quant term folded into `grid_model.py` floor (~45 µs) | 1 hr | Structural; move out of "headroom" |
| **P2** | Gap-C overlap factor (needs microbench) | 4 hr | Only real remaining model item |
| **P3** | Gap-D residual bucket | 1 hr | Named remainder after Gap-S/C/E |
| ~~—~~ | ~~Gap-A HBM contention, Gap-B AIC-AIV serial~~ | — | Refuted for this kernel class; do not implement on chunk_kda |

---

## 7. Refuted Claims (Lessons Learned)

| Claim | Why Refuted |
|-------|-------------|
| Author headroom decomposable via existing 5-gap model | Gaps are WITHIN T_bound, not the headroom |
| HBM contention is 15–30% of headroom (Gap-A) | Measured total MTE activity ≈ 4.7%; refuted by this kernel's msprof |
| AIC-AIV dequant serialization is 10–20% (Gap-B) | `aic_mac_ratio = 0.006`; cube idle-but-allocated; W4A16 mechanism absent |
| msprof exposes only aggregate task duration | CSV has per-engine ratios + icache rates (parser limitation, not data) |
| Cache behavior drives the residual (Gap-D content) | `icache_miss_rate = 0.0` |
| Wave quantization is 500–1,500 µs (Gap-E v1) | Its own formula gives ~45 µs |
| Author headroom is mostly "author" inefficiency | ~entire gap is unmodeled scalar = bound looseness (model headroom) |

---

## 8. Limitations & Open Questions

1. **The bound under-prices scalar by design.** Gap-S is closed by US-SB-007
   (direct CCE scalar measurement); until then the bound stays loose but sound.

2. **MIX_AIC vs simple kernels.** Component attribution is proven on the MIX_AIC
   chunk_kda row; DSA_SQE/AI_VECTOR_CORE kernels need a populated-field check
   before the same method applies.

3. **Overlap (Gap-C) and per-core variance** remain invisible to msprof-aggregate;
   they need microbench calibration / per-core traces respectively.

4. **Counterfactual validation still blocked** — DES-JSON edits cannot be
   recompiled; Gap-S is validated by *agreement of two independent inputs* rather
   than by a hardware counterfactual.

5. **structural vs measured scalar share** (72.7% vs 91.6%) is itself the
   calibration signal — closing it is the US-SB-007 deliverable.

---

## 9. Verified Sources

| Source | Used for | Access |
|--------|----------|--------|
| `chunk_kda_op_summary.csv` (MIX_AIC row) | §1/§2.2 measured engine ratios | `.omc/research/hw_runs/` |
| `kda_des.json` | §2.2 structural per-pipe busy | `.omc/research/hw_runs/` |
| `scripts/component_attribution_prototype.py` | reproduces all §1/§2.2 numbers | repo |
| `perfbound/model/component_model.py:288–306` | scalar floor uses vector rate | codebase |
| `perfbound/calibration/constants.py:252` | scalar-rate soundness fallback | codebase |
| `perfbound/combine/two_limit.py` | author/compiler headroom defs | codebase |
| RESULTS.md §8 (US-SB-007) | scalar calibration is the open item | `.omc/research/hw_runs/` |
