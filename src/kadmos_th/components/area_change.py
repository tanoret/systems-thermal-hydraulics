from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base import Component
from ..equation import Equation


@dataclass
class AreaChange(Component):
    """Sudden area change (expansion/contraction), isenthalpic.

    Ports
    -----
    - inlet:  'in'
    - outlet: 'out'

    Model
    -----
    - mass continuity
    - isenthalpic (throttling): h_out = h_in
    - momentum:
        p_in - p_out = dp_form + dp_acc + dp_grav

      dp_acc uses different inlet/outlet areas:
        dp_acc = m^2 * (1/(rho_out*A_out^2) - 1/(rho_in*A_in^2))

      dp_form uses upstream velocity head by default:
        dp_form = K * (G_in^2)/(2*rho_in)

    Parameters
    ----------
    A_in, A_out:
        Areas [m^2].
    K:
        Loss coefficient.
    dz:
        Elevation change [m] (positive = outlet higher).
    """

    A_in: float = 1.0
    A_out: float = 1.0
    K: float = 0.0
    dz: float = 0.0

    def equations(self, props) -> List[Equation]:
        inc = self._req_in("in")
        out = self._req_out("out")

        m = inc.m.value
        p_in = inc.p.value
        h_in = inc.h.value
        p_out = out.p.value
        h_out = out.h.value

        rho_in = props.rho_ph(p_in, h_in)
        rho_out = props.rho_ph(p_out, h_out)

        # Form loss (use inlet velocity head)
        if self.A_in <= 0 or rho_in <= 0:
            dp_form = 0.0
        else:
            G_in = m / self.A_in
            dp_form = self.K * (G_in ** 2) / (2.0 * rho_in)

        # Acceleration term with area change
        if rho_in <= 0 or rho_out <= 0 or self.A_in <= 0 or self.A_out <= 0:
            dp_acc = 0.0
        else:
            dp_acc = (m ** 2) * (1.0 / (rho_out * self.A_out ** 2) - 1.0 / (rho_in * self.A_in ** 2))

        dp_grav = rho_in * 9.80665 * self.dz  # use inlet density for gravity term

        dp_total = dp_form + dp_acc + dp_grav

        eqs: List[Equation] = []
        eqs.append(Equation(f"{self.name}.mass", out.m.value - m, scale=max(1.0, abs(m))))
        eqs.append(Equation(f"{self.name}.h_isenthalpic", out.h.value - h_in, scale=max(1e5, abs(h_in))))
        eqs.append(Equation(f"{self.name}.dp", (p_in - p_out) - dp_total, scale=max(1e5, abs(p_in))))
        return eqs
