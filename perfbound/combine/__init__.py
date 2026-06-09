# M5 — Bound Combiner + Five-Way Attribution
#
# T_bound = max(T_grid_floor, T_core_floor + T_serial_irreducible)
#
# Five-way attribution (separate from 6 roofline components):
#   1. grid        — realized grid vs optimal partition
#   2. Gap 1       — wrong-unit placement (eligibility vs realized)
#   3. Gap 2       — coalescing / transfer efficiency (MTE)
#   4. Gap 3       — avoidable inter-unit serialization (MTE)
#   5. Gap 4       — intra-unit execution efficiency (compute)
#
# Two-limit computation (A.7):
#   T_bound_HIVM — analytically relaxed, hardware-legal
#   T_bound_DSL  — realized bishengir structural constraints
#   gap → compiler headroom vs kernel-author headroom
