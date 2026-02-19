from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .base import Component
from ..equation import Equation


@dataclass
class Turbine(Component):
    """Turbine with isentropic efficiency.

    Specify either:
      - p_out (absolute outlet pressure), or
      - pr (pressure ratio, p_out = pr*p_in)

    Energy:
      h_out = h_in - eta_is*(h_in - h_isentropic(p_out, s_in))
    """

    eta_is: float = 0.85
    p_out: Optional[float] = None
    pr: Optional[float] = None

    def equations(self, props) -> List[Equation]:
        inc = self._req_in("in")
        out = self._req_out("out")

        m = inc.m.value
        p_in = inc.p.value
        h_in = inc.h.value

        if self.p_out is None and self.pr is None:
            raise ValueError(f"{self.name}: specify p_out or pr")

        p_out = self.p_out if self.p_out is not None else float(self.pr) * p_in

        s_in = props.s_ph(p_in, h_in)
        h_is = props.h_ps(p_out, s_in)
        h_out = h_in - self.eta_is * (h_in - h_is)

        eqs: List[Equation] = []
        eqs.append(Equation(f"{self.name}.mass", out.m.value - m, scale=max(1.0, abs(m))))
        eqs.append(Equation(f"{self.name}.p_out", out.p.value - p_out, scale=max(1e5, abs(p_out))))
        eqs.append(Equation(f"{self.name}.energy", out.h.value - h_out, scale=max(1e5, abs(h_out))))
        return eqs
