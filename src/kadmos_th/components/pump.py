from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .base import Component
from ..equation import Equation


@dataclass
class Pump(Component):
    """Pump model with efficiency.

    Options
    -------
    - Provide p_out (absolute outlet pressure) or dp (pressure rise).

    Energy:
      h_out = h_in + (p_out - p_in)/(rho_in * eta)
    """

    eta: float = 0.8
    p_out: Optional[float] = None
    dp: Optional[float] = None

    def equations(self, props) -> List[Equation]:
        inc = self._req_in("in")
        out = self._req_out("out")

        m = inc.m.value
        p_in = inc.p.value
        h_in = inc.h.value

        if self.p_out is None and self.dp is None:
            raise ValueError(f"{self.name}: specify p_out or dp")

        p_out = self.p_out if self.p_out is not None else (p_in + float(self.dp))

        rho_in = props.rho_ph(p_in, h_in)
        dh = (p_out - p_in) / max(1e-9, rho_in * self.eta)
        h_out = h_in + dh

        eqs: List[Equation] = []
        eqs.append(Equation(f"{self.name}.mass", out.m.value - m, scale=max(1.0, abs(m))))
        eqs.append(Equation(f"{self.name}.p_out", out.p.value - p_out, scale=max(1e5, abs(p_out))))
        eqs.append(Equation(f"{self.name}.energy", out.h.value - h_out, scale=max(1e5, abs(h_out))))
        return eqs

    def shaft_power(self) -> float:
        inc = self._req_in("in")
        out = self._req_out("out")
        return inc.m.value * (out.h.value - inc.h.value)
