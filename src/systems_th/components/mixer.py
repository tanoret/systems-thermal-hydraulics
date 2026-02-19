from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base import Component
from ..equation import Equation


@dataclass
class Mixer(Component):
    """Mix multiple inlet streams into one outlet with pressure equalization."""

    def equations(self, props) -> List[Equation]:
        out = self._req_out("out")
        if len(self.inlets) < 2:
            raise ValueError(f"{self.name}: Mixer needs at least two inlets")

        eqs: List[Equation] = []

        for port, inc in self.inlets.items():
            eqs.append(
                Equation(
                    f"{self.name}.p_eq_{port}",
                    out.p.value - inc.p.value,
                    scale=max(1e5, abs(out.p.value)),
                )
            )

        m_sum = sum(inc.m.value for inc in self.inlets.values())
        e_sum = sum(inc.m.value * inc.h.value for inc in self.inlets.values())

        eqs.append(Equation(f"{self.name}.mass", out.m.value - m_sum, scale=max(1.0, abs(m_sum))))
        eqs.append(Equation(f"{self.name}.energy", out.m.value * out.h.value - e_sum, scale=max(1e6, abs(e_sum))))
        return eqs
