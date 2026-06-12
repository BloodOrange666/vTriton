# M6 — msprof CSV Parser for Validation
#
# Parses msprof op_summary CSVs to extract kernel timing measurements.
# Used by the validation harness to compare T_measured vs T_bound.
#
# Source spec: .omc/plans/a6_validation_harness.md §2

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path
from typing import NamedTuple, Optional

# Import from fit_constants (do not copy)
sys.path.insert(0, str(Path(__file__).parent.parent / "calibration" / "scripts"))
from fit_constants import MSProfRow, read_msprof_csv


# Task-type values that real msprof emits for AI compute-core execution.
# A Triton/HIVM kernel shows up as one of these depending on whether it is
# cube-only (AI_CORE/AICORE), mixed (MIX_AIC cube-dominant / MIX_AIV
# vector-dominant), or pure vector (AI_VECTOR_CORE / AIV).  The chunk_kda
# kernel profiles as MIX_AIC — recognising only "AI_CORE" silently drops it.
#
# Exact (set) membership, not substring: msprof task_type is a discrete enum,
# and substring matching on the short "AIV" token would misclassify any future
# composite type that happens to contain it (e.g. a hypothetical
# "HCCL_AIV_TASK").
_AICORE_TASK_TYPES = frozenset(
    {"AI_CORE", "AICORE", "MIX_AIC", "MIX_AIV", "AI_VECTOR_CORE", "AIV"}
)


def _is_aicore_task(task_type: str) -> bool:
    """True if an (already upper-cased, stripped) task_type is AI compute-core.

    Shared by the timing parser and the component-duration aggregator so the
    two cannot drift — a divergence first surfaced on real hardware, where the
    timing filter dropped MIX_AIC rows the component mapper kept.
    """
    return task_type in _AICORE_TASK_TYPES


class TimingResult(NamedTuple):
    """Parsed kernel timing from msprof CSV."""
    t_us: float                      # median duration in microseconds
    n_invocations: int               # valid invocations used in median
    n_warmup_discarded: int          # warmup invocations discarded


def parse_kernel_time_us(
    csv_path: Path,
    op_name_filter: str | None = None,
    n_warmup: int = 1,
) -> TimingResult:
    """Parse kernel timing from msprof op_summary CSV.

    Algorithm:
    1. Load rows via read_msprof_csv(csv_path).
    2. Filter to AiCore rows: task_type contains "AI_CORE" (case-insensitive);
       fall back to op_type if task_type is empty (old CANN CSV).
    3. If op_name_filter given: keep rows where op_name contains the filter
       (exact normalized match, not unrestricted substring).
    4. Raise ValueError if no rows remain.
    5. Sort by start_time_us. Group sequential rows into invocations:
       each invocation = one or more AiCore rows that started within a tight
       time window (gap threshold: 10× median row duration). Wall-clock latency
       per invocation = max duration_us across concurrent rows for that invocation
       (parallel device tasks should not be summed).
    6. Discard the first n_warmup invocations explicitly.
    7. Raise ValueError if fewer than 1 valid invocation remains.
    8. Return statistics.median(per_invocation_durations).

    Args:
        csv_path: Path to op_summary CSV.
        op_name_filter: Optional op name substring to filter (case-insensitive).
        n_warmup: Number of initial invocations to discard (default: 1).

    Returns:
        TimingResult with median duration, invocation count, and warmup count.

    Raises:
        ValueError: No matching rows, or insufficient invocations after warmup.
        OSError: CSV file not found or unreadable.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise OSError(f"CSV file not found: {csv_path}")

    rows = read_msprof_csv(csv_path)

    # Filter to AiCore rows
    aicore_rows = []
    for row in rows:
        # Skip malformed rows (NaN, zero duration)
        if not row.duration_us or math.isnan(row.duration_us) or row.duration_us <= 0:
            print(f"Warning: Skipping row with invalid duration: {row.duration_us}", file=sys.stderr)
            continue

        # Filter by task_type (prefer) or op_type (fallback)
        task_type = row.task_type.strip().upper() if row.task_type else ""
        if not task_type:
            # Old CANN CSV: fall back to op_type
            task_type = row.op_type.strip().upper() if row.op_type else ""

        if _is_aicore_task(task_type):
            # Apply op_name filter if specified
            if op_name_filter:
                if op_name_filter.lower() in row.op_name.lower():
                    aicore_rows.append(row)
            else:
                aicore_rows.append(row)

    if not aicore_rows:
        raise ValueError(f"No AiCore rows found in {csv_path} (filter={op_name_filter})")

    # Sort by start_time_us
    aicore_rows.sort(key=lambda r: r.start_time_us)

    # Group into invocations using gap detection.
    # NOTE: gap_threshold = 10× median row duration is a heuristic.  Known
    # failure mode: tightly-pipelined real traces where the inter-invocation
    # gap is comparable to per-row duration will merge adjacent invocations
    # into one, undercounting n_invocations and inflating per-invocation
    # latency (max across merged rows).  Acceptable for A.6.1; a more robust
    # detector (e.g. adaptive threshold from bimodal gap distribution) is
    # future work.
    durations = [r.duration_us for r in aicore_rows]
    median_duration = statistics.median(durations)
    gap_threshold = 10.0 * median_duration

    invocations = []
    current_invocation = [aicore_rows[0]]

    for i in range(1, len(aicore_rows)):
        row = aicore_rows[i]
        prev_row = aicore_rows[i - 1]
        time_gap = row.start_time_us - prev_row.start_time_us

        if time_gap > gap_threshold:
            # New invocation
            invocations.append(current_invocation)
            current_invocation = [row]
        else:
            # Same invocation
            current_invocation.append(row)

    invocations.append(current_invocation)

    # Discard warmup invocations
    if n_warmup > 0 and len(invocations) > n_warmup:
        invocations = invocations[n_warmup:]
        n_warmup_discarded = n_warmup
    else:
        n_warmup_discarded = 0

    if not invocations:
        raise ValueError(f"No valid invocations remaining after warmup={n_warmup} in {csv_path}")

    # Compute wall-clock latency per invocation (max duration across concurrent rows)
    per_invocation_durations = []
    for inv in invocations:
        max_duration = max(r.duration_us for r in inv)
        per_invocation_durations.append(max_duration)

    # Return median
    t_us = statistics.median(per_invocation_durations)

    return TimingResult(
        t_us=t_us,
        n_invocations=len(per_invocation_durations),
        n_warmup_discarded=n_warmup_discarded,
    )


def parse_component_durations(
    csv_path: Path,
    op_name_filter: str | None = None,
) -> dict[str, float]:
    """Return total duration per task-type category from all rows.

    Categories: 'aicore', 'mte', 'aicpu', 'other'.

    Map CSV task_type values (via _is_aicore_task):
    - AI_CORE/AICORE/MIX_AIC/MIX_AIV/AI_VECTOR_CORE/AIV → 'aicore'
    - MTE* → 'mte'
    - AI_CPU/AiCPU → 'aicpu'
    - else → 'other'

    Args:
        csv_path: Path to op_summary CSV.
        op_name_filter: Optional op name substring to filter (case-insensitive).
            When provided, only rows whose op_name contains the filter are
            included in the totals.  This is essential for multi-kernel CSVs
            where the dominant component must be computed per-kernel.

    Returns:
        Dict mapping category → total duration in microseconds.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise OSError(f"CSV file not found: {csv_path}")

    rows = read_msprof_csv(csv_path)

    totals = {"aicore": 0.0, "mte": 0.0, "aicpu": 0.0, "other": 0.0}

    for row in rows:
        if not row.duration_us or math.isnan(row.duration_us) or row.duration_us <= 0:
            continue

        # Apply op_name filter if specified
        if op_name_filter:
            if op_name_filter.lower() not in row.op_name.lower():
                continue

        task_type = row.task_type.strip().upper() if row.task_type else ""
        if not task_type:
            task_type = row.op_type.strip().upper() if row.op_type else ""

        if _is_aicore_task(task_type):
            totals["aicore"] += row.duration_us
        elif task_type.startswith("MTE"):
            totals["mte"] += row.duration_us
        elif "AI_CPU" in task_type or "AICPU" in task_type:
            totals["aicpu"] += row.duration_us
        else:
            totals["other"] += row.duration_us

    return totals


class EngineAttribution(NamedTuple):
    """Measured per-engine time split for one kernel row (author-headroom A.8).

    msprof reports, per AI-core invocation, the fraction of AIV (vector core)
    and AIC (cube core) busy time spent in each micro-engine (scalar / vector /
    mte / mac / fixpipe).  Multiplying those ratios by the core busy time gives
    the engine-time breakdown of where the hardware actually spent its cycles —
    the empirical counterpart to the HIVM structural per-pipe split.

    ``populated`` is False when the matched row lacks per-engine ratios (e.g. a
    DSA_SQE / pure AI_VECTOR_CORE kernel whose ratio columns are ``N/A``); in
    that case only ``t_measured_us`` is meaningful.
    """
    engine_us: dict[str, float]
    t_measured_us: float
    aiv_time_us: float
    aic_time_us: float
    scalar_us: float
    scalar_frac: float            # scalar_us / t_measured_us (blended AIV+AIC, dominance display)
    aiv_scalar_frac: float        # scalar(AIV) / aiv_time_us (same-core, for Gap-OVL pts)
    dominant_engine: str
    icache_miss: tuple[float, float]   # (aic, aiv)
    cube_util_pct: float
    populated: bool


# Per-engine ratio columns, keyed by the human label used in the breakdown.
# (label, core, ratio_column) — core picks aiv_time vs aicore_time as the base.
_ENGINE_RATIO_COLUMNS = [
    ("scalar (AIV)", "aiv", "aiv_scalar_ratio"),
    ("vector (AIV)", "aiv", "aiv_vec_ratio"),
    ("mte2-load (AIV)", "aiv", "aiv_mte2_ratio"),
    ("mte3-store (AIV)", "aiv", "aiv_mte3_ratio"),
    ("mac-cube (AIC)", "aic", "aic_mac_ratio"),
    ("scalar (AIC)", "aic", "aic_scalar_ratio"),
    ("mte1 (AIC)", "aic", "aic_mte1_ratio"),
    ("mte2-load (AIC)", "aic", "aic_mte2_ratio"),
    ("fixpipe (AIC)", "aic", "aic_fixpipe_ratio"),
]


def _ffloat(value: str | None) -> float:
    """Parse a CSV cell to float; 'N/A'/blank/garbage → NaN."""
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def parse_engine_attribution(
    csv_path: Path | str,
    op_name_filter: str | None = None,
) -> Optional[EngineAttribution]:
    """Extract the measured per-engine time split for a kernel from msprof.

    Picks the longest-duration AI-core row matching ``op_name_filter`` (the
    kernel itself, not its setup ops) and converts its per-engine ratios into
    absolute engine-time.  Uses the raw CSV columns directly because the shared
    ``MSProfRow`` does not carry the ratio fields.

    Args:
        csv_path: Path to an msprof op_summary CSV.
        op_name_filter: Substring matched against ``Op Name`` (case-insensitive).

    Returns:
        EngineAttribution, or None if no matching AI-core row exists.  When the
        row exists but lacks ratio columns, the result has ``populated=False``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise OSError(f"CSV file not found: {csv_path}")

    with open(csv_path) as f:
        reader = csv.DictReader(line for line in f if not line.strip().startswith("#"))
        candidates = []
        for line in reader:
            name = (line.get("Op Name") or line.get("op_name") or "")
            if op_name_filter and op_name_filter.lower() not in name.lower():
                continue
            ttype = (line.get("Task Type") or line.get("task_type") or "").strip().upper()
            if not _is_aicore_task(ttype):
                continue
            dur = _ffloat(line.get("Task Duration(us)") or line.get("duration(us)"))
            if math.isnan(dur) or dur <= 0:
                continue
            candidates.append((dur, line))

    if not candidates:
        return None

    _, row = max(candidates, key=lambda c: c[0])

    t = _ffloat(row.get("Task Duration(us)"))
    aiv = _ffloat(row.get("aiv_time(us)"))
    aic = _ffloat(row.get("aicore_time(us)"))
    base = {"aiv": aiv, "aic": aic}

    engine_us: dict[str, float] = {}
    populated = not (math.isnan(aiv) or math.isnan(aic))
    for label, core, col in _ENGINE_RATIO_COLUMNS:
        ratio = _ffloat(row.get(col))
        b = base[core]
        if math.isnan(ratio) or math.isnan(b):
            populated = False
            continue
        engine_us[label] = ratio * b

    scalar_us = engine_us.get("scalar (AIV)", 0.0) + engine_us.get("scalar (AIC)", 0.0)
    scalar_frac = scalar_us / t if t and not math.isnan(t) else 0.0
    # Same-core scalar share: scalar(AIV) over the AIV core's own busy time.
    # This is the apples-to-apples counterpart to the DES schedule's
    # critical-path exposed-control fraction (both normalize to one core's
    # timeline), and is what Gap-OVL subtracts.  scalar_frac (blended over
    # wall-clock) is kept only for the dominance display.
    aiv_scalar_us = engine_us.get("scalar (AIV)", 0.0)
    aiv_scalar_frac = aiv_scalar_us / aiv if aiv and not math.isnan(aiv) else 0.0
    dominant = max(engine_us, key=engine_us.get) if engine_us else ""

    return EngineAttribution(
        engine_us=engine_us,
        t_measured_us=t,
        aiv_time_us=aiv,
        aic_time_us=aic,
        scalar_us=scalar_us,
        scalar_frac=scalar_frac,
        aiv_scalar_frac=aiv_scalar_frac,
        dominant_engine=dominant,
        icache_miss=(_ffloat(row.get("aic_icache_miss_rate")),
                     _ffloat(row.get("aiv_icache_miss_rate"))),
        cube_util_pct=_ffloat(row.get("cube_utilization(%)")),
        populated=populated,
    )
