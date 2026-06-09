# A.5 Progress — Bound Combiner, Attribution, Two-Limit

## Status: Code complete, gaps #3–#7 remediated, gaps #1–#2 open

### Review Round 1 — 8 findings (pre-remediation)
All 8 fixed in the initial A.5 implementation.

### Review Round 2 — 7 gaps (remediation targets)

| Gap | Severity | Status | Notes |
|-----|----------|--------|-------|
| #1 | HIGH | ❌ Open | C++ HIVM parser fix (bishengir build or text-parser extension) |
| #2 | HIGH | ❌ Open | Real-kernel milestone (blocked by #1) |
| #3 | MED-HIGH | ✅ Fixed | Scalar rate added (Vector/20 proxy). **See caveat below.** |
| #4 | LOW-MED | ✅ Fixed | Two-limit test now proves strict HIVM < DSL |
| #5 | LOW | ✅ Fixed | report_from_npuir passes --python to tritonsim-hivm |
| #6 | LOW | ✅ Fixed | Wave scaling uses `replace()` (no mutation); docstring corrected |
| #7 | sign-off | ✅ Fixed | Spec updated: §4.1 formula box + example + implementation_plan.md all use `max(grid, core+serial)` |

### ⚠️ Soundness caveat — Scalar Vector/20 proxy rate (Gap #3)

The Scalar rate lives in `compute_component_floor` (component_model.py:288-302), so it feeds
the **headline T_bound**, not just two-limit. For a genuinely Scalar-bound kernel, T_bound
rides on the Vector/20 guess. Since Vector/20 *assumes* Scalar is 20× slower than Vector
(pessimistic), it can **overestimate** Scalar time, potentially making T_bound > T_measured —
violating the §4.0 conservatism theorem.

**Mitigating factors:**
- Scalar rarely binds (canonical case is Cube or MTE bound)
- Aligns with spec §2.1 (Scalar as first-class component)
- The old t_c=0 was also wrong (underestimate → unsound for a *lower* bound)

**Action needed:** Calibrate real Scalar throughput in M6/B.4 before trusting as a lower
bound. If real Scalar is faster than Vector/20, the proxy is conservative (sound); if slower,
the bound could exceed T_measured on Scalar-bound kernels.

### Gap math invariant (wave scaling)

Wave scaling scales `t_core_floor_us` and `per_component_us` by `waves` but intentionally
**does not** scale `total_ops`/`total_bytes`. This keeps gap helpers correct:

```
gap = comp_time_waves × (op_work_1prog / total_ops_1prog)
    = (comp_time_1 × waves) × (op_1 / total_1)
    = waves × (comp_time_1 × op_1 / total_1)   ← correctly waves-scaled
```

If total_ops were also scaled, the waves factors would cancel and gaps would be 1-wave
magnitude regardless of actual wave count — wrong for multi-wave kernels like chunk_kda
(~205 waves).

### Test suite
- 174 passed, 3 xfailed (CLI tests — blocked by Gap #1)
- Two-limit test proves strict HIVM < DSL with Scalar→Vector idealization

### Files modified (this remediation)
1. perfbound/calibration/constants.py — Scalar rate in VectorConfig
2. perfbound/model/component_model.py — Scalar component uses non-zero rate
3. perfbound/model/bounds.py — Wave scaling uses replace(), no total_ops scaling
4. perfbound/combine/bound_combiner.py — Spec divergence doc, docstring fix
5. perfbound/combine/run_report.py — --python flag wired
6. tests/perfbound/test_two_limit.py — Strict HIVM < DSL test
7. tests/perfbound/test_hivm_cli_integration.py — xfail markers restored
