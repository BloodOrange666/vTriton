# M5 — Per-Kernel Report (text + JSON)
#
# Deliverable: bound, binding tier/component, five-way attribution,
# two-limit gap, single recommended action.
#
# Source spec: .omc/specs/performance_bound_model.md §A.5

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .bound_combiner import BoundResult, BindingTier
from .two_limit import TwoLimitResult


_RECOMMENDATIONS = {
    "grid": "Fix grid partitioning — increase occupancy or load balance",
    "gap1_wrong_unit": "Fix DSL types — move ops to eligible unit",
    "gap2_coalescing": "Merge transfers — increase transfer size to reduce amortization",
    "gap3_avoidable_serial": "Add ping-pong buffer to overlap this handoff",
    "gap4_intra_unit_exec": "Increase SIMD repeat/mask utilization",
}

# Threshold below which all gaps are considered negligible ("at bound")
_AT_BOUND_EPS = 1e-4


@dataclass
class KernelReport:
    """Complete per-kernel performance bound report."""
    kernel_name: str

    # Bound
    t_bound_us: float
    binding_tier: str
    binding_component: Optional[str] = None

    # Decomposed
    t_grid_floor_us: float = 0.0
    t_core_floor_us: float = 0.0
    t_serial_irreducible_us: float = 0.0

    # Two-limit (A.7)
    t_bound_hivm_us: Optional[float] = None
    t_measured_us: Optional[float] = None
    compiler_headroom_us: Optional[float] = None
    author_headroom_us: Optional[float] = None

    # Attribution (five-way, fractions of T_bound)
    attribution: dict[str, float] = field(default_factory=dict)

    # Recommendation
    recommended_action: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "kernel_name": self.kernel_name,
            "t_bound_us": self.t_bound_us,
            "binding_tier": self.binding_tier,
            "binding_component": self.binding_component,
            "t_grid_floor_us": self.t_grid_floor_us,
            "t_core_floor_us": self.t_core_floor_us,
            "t_serial_irreducible_us": self.t_serial_irreducible_us,
            "t_bound_hivm_us": self.t_bound_hivm_us,
            "t_measured_us": self.t_measured_us,
            "compiler_headroom_us": self.compiler_headroom_us,
            "author_headroom_us": self.author_headroom_us,
            "attribution": self.attribution,
            "recommended_action": self.recommended_action,
        }

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize to JSON string, optionally writing to a file."""
        text = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).write_text(text)
        return text

    def to_text(self) -> str:
        """Human-readable text report."""
        lines = [
            f"=== Performance Bound Report: {self.kernel_name} ===",
            f"",
            f"T_bound:   {self.t_bound_us:.2f} us",
            f"  Tier 1 (grid):      {self.t_grid_floor_us:.2f} us",
            f"  Tier 2 (component): {self.t_core_floor_us:.2f} us",
            f"  Serial irreducible: {self.t_serial_irreducible_us:.2f} us",
            f"",
            f"Binding: {self.binding_tier}",
        ]
        if self.binding_component:
            lines.append(f"  Component: {self.binding_component}")

        lines.append(f"")
        lines.append(f"Attribution (fraction of T_bound):")
        for gap_name, frac in sorted(self.attribution.items(), key=lambda x: -x[1]):
            lines.append(f"  {gap_name}: {frac:.3f}")

        if self.t_bound_hivm_us is not None:
            lines.append(f"")
            lines.append(f"Two-Limit (A.7):")
            lines.append(f"  T_bound_HIVM:       {self.t_bound_hivm_us:.2f} us")
            if self.compiler_headroom_us is not None:
                lines.append(f"  Compiler headroom:  {self.compiler_headroom_us:.2f} us")
            if self.author_headroom_us is not None:
                lines.append(f"  Author headroom:    {self.author_headroom_us:.2f} us")

        lines.append(f"")
        lines.append(f"Recommended action: {self.recommended_action}")

        return "\n".join(lines)

    @classmethod
    def from_bound(cls, result: BoundResult,
                   two_limit: Optional[TwoLimitResult] = None) -> "KernelReport":
        """Create a report from a BoundResult."""
        dominant_name, dominant_frac = result.attribution.dominant_gap()

        # At-bound detection: when all gap fractions are below ε, the kernel
        # is at its analytical bound — no actionable software gap remains.
        total_gap_frac = (
            result.attribution.grid_gap_frac
            + result.attribution.gap1_frac
            + result.attribution.gap2_frac
            + result.attribution.gap3_frac
            + result.attribution.gap4_frac
        )
        if total_gap_frac < _AT_BOUND_EPS:
            action = (
                "At component bound — no actionable software gap "
                "(consider algorithmic redesign: fusion/precision/less traffic)"
            )
        else:
            action = _RECOMMENDATIONS.get(dominant_name, "Profile to identify bottleneck")

        return cls(
            kernel_name=result.kernel_name,
            t_bound_us=result.t_bound_us,
            binding_tier=result.binding_tier.value,
            binding_component=result.binding_component.value if result.binding_component else None,
            t_grid_floor_us=result.t_grid_floor_us,
            t_core_floor_us=result.t_core_floor_us,
            t_serial_irreducible_us=result.t_serial_irreducible_us,
            t_bound_hivm_us=two_limit.t_bound_hivm_us if two_limit else None,
            t_measured_us=two_limit.t_measured_us if two_limit else None,
            compiler_headroom_us=two_limit.compiler_headroom_us if two_limit else None,
            author_headroom_us=two_limit.author_headroom_us if two_limit else None,
            attribution={
                "grid": result.attribution.grid_gap_frac,
                "gap1_wrong_unit": result.attribution.gap1_frac,
                "gap2_coalescing": result.attribution.gap2_frac,
                "gap3_avoidable_serial": result.attribution.gap3_frac,
                "gap4_intra_unit_exec": result.attribution.gap4_frac,
            },
            recommended_action=action,
        )
