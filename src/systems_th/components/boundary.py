from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from .base import Component
from ..equation import Equation


@dataclass
class Source(Component):
    """Boundary condition source: sets outlet state if values are provided."""

    m_dot: Optional[float] = None
    p: Optional[float] = None
    h: Optional[float] = None

    def equations(self, props) -> List[Equation]:
        out = self._req_out("out")
        eqs: List[Equation] = []
        if self.m_dot is not None:
            eqs.append(Equation(f"{self.name}.m_out", out.m.value - self.m_dot, scale=max(1.0, abs(self.m_dot))))
        if self.p is not None:
            eqs.append(Equation(f"{self.name}.p_out", out.p.value - self.p, scale=max(1e5, abs(self.p))))
        if self.h is not None:
            eqs.append(Equation(f"{self.name}.h_out", out.h.value - self.h, scale=max(1e5, abs(self.h))))
        return eqs


@dataclass
class Sink(Component):
    """Boundary sink: can enforce pressure/enthalpy if provided."""

    p: Optional[float] = None
    h: Optional[float] = None

    def equations(self, props) -> List[Equation]:
        inc = self._req_in("in")
        eqs: List[Equation] = []
        if self.p is not None:
            eqs.append(Equation(f"{self.name}.p_in", inc.p.value - self.p, scale=max(1e5, abs(self.p))))
        if self.h is not None:
            eqs.append(Equation(f"{self.name}.h_in", inc.h.value - self.h, scale=max(1e5, abs(self.h))))
        return eqs
