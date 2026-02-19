from __future__ import annotations

import math


def haaland_friction_factor(Re: float, eps_rel: float) -> float:
    """Haaland explicit approximation for Darcy friction factor."""
    Re = abs(Re)
    if Re <= 0.0:
        return 0.0
    if Re < 2300.0:
        return 64.0 / Re
    return (-1.8 * math.log10((eps_rel / 3.7) ** 1.11 + 6.9 / Re)) ** -2
