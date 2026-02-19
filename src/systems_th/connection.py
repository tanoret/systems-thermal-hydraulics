from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .variable import Variable


@dataclass
class Connection:
    """A directed connection between component ports carrying water/steam state.

    Primary state variables:
      - m: mass flow [kg/s]
      - p: pressure [Pa]
      - h: specific enthalpy [J/kg]
    """

    name: str
    m: Variable
    p: Variable
    h: Variable

    @classmethod
    def create(
        cls,
        name: str,
        m_guess: float = 100.0,
        p_guess: float = 1e6,
        h_guess: float = 1e6,
        m_bounds: tuple[float, float] = (1e-6, 1e9),
        p_bounds: tuple[float, float] = (1e3, 1e9),
        h_bounds: tuple[float, float] = (1e3, 1e8),
    ) -> "Connection":
        return cls(
            name=name,
            m=Variable(f"{name}.m", float(m_guess), fixed=False, lower=m_bounds[0], upper=m_bounds[1]),
            p=Variable(f"{name}.p", float(p_guess), fixed=False, lower=p_bounds[0], upper=p_bounds[1]),
            h=Variable(f"{name}.h", float(h_guess), fixed=False, lower=h_bounds[0], upper=h_bounds[1]),
        )

    def fix(self, m: Optional[float] = None, p: Optional[float] = None, h: Optional[float] = None) -> None:
        if m is not None:
            self.m.fix(m)
        if p is not None:
            self.p.fix(p)
        if h is not None:
            self.h.fix(h)

    def guess(self, m: Optional[float] = None, p: Optional[float] = None, h: Optional[float] = None) -> None:
        if m is not None:
            self.m.value = float(m)
        if p is not None:
            self.p.value = float(p)
        if h is not None:
            self.h.value = float(h)

    def variables(self) -> list[Variable]:
        return [self.m, self.p, self.h]
