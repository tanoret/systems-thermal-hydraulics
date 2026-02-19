from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .base import Component
from ..equation import Equation


@dataclass
class Condenser(Component):
    """Condenser enforcing outlet to specified quality (default saturated liquid).

    This is a thermodynamic boundary-like component (no UA/HTC in v0.2).
    Heat rejected is a result: Q_dot = m*(h_in - h_out).
    """

    dp: float = 0.0
    x_out: float = 0.0
    p_out: Optional[float] = None  # if set, fixes outlet pressure; otherwise uses p_in - dp

    def equations(self, props) -> List[Equation]:
        inc = self._req_in("in")
        out = self._req_out("out")

        m = inc.m.value
        p_in = inc.p.value

        p_out = self.p_out if self.p_out is not None else (p_in - self.dp)
        h_out = props.h_px(p_out, self.x_out)

        eqs: List[Equation] = []
        eqs.append(Equation(f"{self.name}.mass", out.m.value - m, scale=max(1.0, abs(m))))
        eqs.append(Equation(f"{self.name}.p_out", out.p.value - p_out, scale=max(1e5, abs(p_out))))
        eqs.append(Equation(f"{self.name}.h_out", out.h.value - h_out, scale=max(1e5, abs(h_out))))
        return eqs

    def heat_rejected(self) -> float:
        inc = self._req_in("in")
        out = self._req_out("out")
        return inc.m.value * (inc.h.value - out.h.value)
