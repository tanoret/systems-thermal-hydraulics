from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .base import Component
from ..equation import Equation


@dataclass
class Heater(Component):
    """Simple heater enforcing outlet temperature or outlet enthalpy."""

    dp: float = 0.0
    T_out: Optional[float] = None
    h_out: Optional[float] = None

    def equations(self, props) -> List[Equation]:
        inc = self._req_in("in")
        out = self._req_out("out")

        if self.T_out is None and self.h_out is None:
            raise ValueError(f"{self.name}: specify T_out or h_out")

        m = inc.m.value
        p_in = inc.p.value

        p_out = p_in - self.dp
        h_target = self.h_out if self.h_out is not None else props.h_pT(p_out, float(self.T_out))

        eqs: List[Equation] = []
        eqs.append(Equation(f"{self.name}.mass", out.m.value - m, scale=max(1.0, abs(m))))
        eqs.append(Equation(f"{self.name}.p_out", out.p.value - p_out, scale=max(1e5, abs(p_out))))
        eqs.append(Equation(f"{self.name}.h_out", out.h.value - float(h_target), scale=max(1e5, abs(h_target))))
        return eqs

    def heat_added(self) -> float:
        inc = self._req_in("in")
        out = self._req_out("out")
        return inc.m.value * (out.h.value - inc.h.value)
