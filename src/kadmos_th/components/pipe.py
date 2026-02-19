from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import math

from .base import Component
from ..equation import Equation
from ..correlations.pressure_drop import dp_pipe


@dataclass
class Pipe(Component):
    """A 1-in/1-out pipe with friction + form loss + gravity + acceleration.

    This is a steady-state integral momentum model (0D component). It is **two-phase aware**
    via mixture properties computed from (p,h) using IAPWS97 (HEM).

    Parameters
    ----------
    L:
        Length [m]
    D:
        Hydraulic diameter [m]
    A:
        Flow area [m^2]. If None, uses pi*D^2/4.
    eps:
        Absolute roughness [m]
    K:
        Lumped form loss coefficient [-]
    dz:
        Elevation change [m] (positive = outlet higher than inlet)
    Q:
        Heat added to the fluid [W] (positive adds enthalpy). Default 0 (adiabatic).
    two_phase_friction:
        "homogeneous" (default) or "chisholm".
    include_acceleration:
        Include acceleration dp term based on inlet/outlet densities.
    """

    L: float = 1.0
    D: float = 0.1
    A: Optional[float] = None
    eps: float = 1e-5
    K: float = 0.0
    dz: float = 0.0
    Q: float = 0.0
    two_phase_friction: str = "homogeneous"
    include_acceleration: bool = True

    def equations(self, props) -> List[Equation]:
        inc = self._req_in("in")
        out = self._req_out("out")

        m = inc.m.value
        p_in = inc.p.value
        h_in = inc.h.value

        # Mass
        eqs: List[Equation] = [
            Equation(f"{self.name}.mass", out.m.value - m, scale=max(1.0, abs(m))),
        ]

        # Energy: h_out = h_in + Q/m (adiabatic if Q=0)
        dh = self.Q / m if abs(m) > 1e-9 else 0.0
        h_out_target = h_in + dh
        eqs.append(Equation(f"{self.name}.energy", out.h.value - h_out_target, scale=max(1e5, abs(h_out_target))))

        # Momentum: p_in - p_out = Δp
        p_out = out.p.value
        h_out = out.h.value

        dp = dp_pipe(
            m_dot=m,
            p_in=p_in,
            h_in=h_in,
            p_out=p_out,
            h_out=h_out,
            props=props,
            L=self.L,
            D=self.D,
            eps=self.eps,
            K=self.K,
            dz=self.dz,
            A=self.A,
            two_phase_friction=self.two_phase_friction,
            include_acceleration=self.include_acceleration,
            include_gravity=True,
        )
        eqs.append(Equation(f"{self.name}.dp", (p_in - p_out) - dp.total, scale=max(1e5, abs(p_in))))

        return eqs
