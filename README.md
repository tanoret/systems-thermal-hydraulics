# systems-th (v0.2)

A **steady-state** thermal-hydraulics **network solver** (Newton + finite-difference Jacobian) inspired by TESPy’s
architecture, aimed at **systems-like** reactor concepts:

- subcooled water downcomer → boiling core (target exit void fraction) → orifice → chimney → separator
- vapor branch to turbine → condenser → feedwater pump → returns to downcomer/mixer

This is **not** TESPy and does not depend on TESPy’s code; it reuses the high-level idea:
**connections carry state; components contribute equations; a network solver assembles and solves the system**.

## What’s new in v0.2
- IAPWS97 wrapper expanded with **transport properties** (μ, k, cp, σ) and **LRU caching**
- Two-phase pressure-drop modeling:
  - Default: **Homogeneous Equilibrium Model (HEM)**
  - Optional: **Chisholm / Lockhart–Martinelli-style two-phase multiplier**
  - Includes **acceleration** pressure drop term and gravity
- Cleaner `CoreChannel` equations (no redundant constraints) and support for spacer-grid losses
- New correlation modules: `systems_th.correlations.pressure_drop` and `systems_th.correlations.heat_transfer`

## Install
```bash
pip install -e .
```

## Run example
```bash
python examples/systems_loop.py
```

## State definition (connections)
Each connection holds primary variables:
- mass flow `m` [kg/s]
- pressure `p` [Pa]
- specific enthalpy `h` [J/kg]

Derived two-phase quantities are computed from (p,h) via IAPWS97:
- quality `x`
- void fraction `alpha` (HEM)
- mixture density

## Current scope / roadmap
- Steady-state only (transients + point kinetics are planned)
- Heat-transfer correlations are included as utilities; the solver currently enforces energy via `Q` inputs/unknowns
- Next production hardening steps:
  - validated pin-bundle friction/spacer-grid correlations
  - non-condensables support in condenser/separator
  - dedicated controller components (pump speed, valve position, power control)

## License
MIT (see LICENSE).
