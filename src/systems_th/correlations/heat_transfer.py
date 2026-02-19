from __future__ import annotations

import math


def htc_dittus_boelter(G: float, D: float, mu: float, cp: float, k: float, n: float = 0.4) -> float:
    """Single-phase turbulent internal convection (Dittus–Boelter).

    Parameters
    ----------
    G:
        Mass flux [kg/m^2/s]
    D:
        Hydraulic diameter [m]
    mu:
        Dynamic viscosity [Pa*s]
    cp:
        Heat capacity [J/kg/K]
    k:
        Thermal conductivity [W/m/K]
    n:
        Exponent: ~0.4 for heating, ~0.3 for cooling

    Returns
    -------
    h:
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Valid (roughly): Re > 10^4, 0.7 < Pr < 160, L/D > 10.
    """
    if D <= 0 or mu <= 0 or k <= 0 or cp <= 0:
        return 0.0
    Re = abs(G * D / mu)
    Pr = cp * mu / k
    if Re < 2300:
        # Laminar fallback (constant wall temperature, fully developed): Nu=3.66
        Nu = 3.66
    else:
        Nu = 0.023 * (Re ** 0.8) * (Pr ** n)
    return Nu * k / D


# Placeholders for future validated 2-phase HTC correlations.
# Intentionally not used in solver equations in v0.2; these are utilities for post-processing.
def htc_two_phase_placeholder() -> float:
    """Placeholder for two-phase HTC correlations (flow boiling/condensation)."""
    raise NotImplementedError(
        "Two-phase HTC correlations are not yet implemented in v0.2. "
        "Use htc_dittus_boelter for single-phase or extend this module with validated correlations."
    )
