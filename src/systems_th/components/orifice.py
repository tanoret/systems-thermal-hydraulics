from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .base import Component
from ..equation import Equation


@dataclass
class OrificePlate(Component):
    """Isenthalpic orifice / flow restriction.

    Models a pressure drop with either:
      - direct K-loss: dp = K * G^2/(2 rho)
      - or a discharge coefficient Cd and area A: dp = (m/(Cd*A))^2 / (2 rho)

    For flashing, the isenthalpic drop naturally increases quality downstream if saturation is crossed.
    """

    K: Optional[float] = None
    Cd: Optional[float] = None
    A: Optional[float] = None
    dz: float = 0.0  # optional elevation change

    def equations(self, props) -> List[Equation]:
        inc = self._req_in("in")
        out = self._req_out("out")

        m = inc.m.value
        p_in = inc.p.value
        h_in = inc.h.value
        rho = props.rho_ph(p_in, h_in)

        if self.K is not None:
            if self.A is None:
                raise ValueError(f"{self.name}: A must be provided when using K-loss model")
            G = m / self.A
            dp = float(self.K) * (G ** 2) / (2.0 * rho)
        else:
            if self.Cd is None or self.A is None:
                raise ValueError(f"{self.name}: specify either K or (Cd and A)")
            v = m / (rho * self.A)
            dp = (v ** 2) * rho / (2.0 * (float(self.Cd) ** 2))

        dp_g = rho * 9.80665 * self.dz
        dp_total = dp + dp_g

        eqs: List[Equation] = []
        eqs.append(Equation(f"{self.name}.mass", out.m.value - m, scale=max(1.0, abs(m))))
        eqs.append(Equation(f"{self.name}.h_isenthalpic", out.h.value - h_in, scale=max(1e5, abs(h_in))))
        eqs.append(Equation(f"{self.name}.dp", (p_in - out.p.value) - dp_total, scale=max(1e5, abs(p_in))))
        return eqs
