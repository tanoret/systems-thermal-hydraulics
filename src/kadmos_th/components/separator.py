from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base import Component
from ..equation import Equation


@dataclass
class Separator(Component):
    """Steam separator / phase splitter with two outlets.

    Ports
    -----
    - inlet:  'in'
    - vapor:  'vap'
    - liquid: 'liq'

    Model
    -----
    - common pressure drop from inlet to each outlet: p_out = p_in - dp
    - outlet qualities are *targets* representing separation efficiency:
        vap outlet quality = x_vap_target (close to 1)
        liq outlet quality = x_liq_target (close to 0)
    - mass + energy determine the split (unknown vapor/liquid mass flows).
    """

    dp: float = 0.0
    x_vap_target: float = 0.999
    x_liq_target: float = 0.001

    def equations(self, props) -> List[Equation]:
        inc = self._req_in("in")
        vap = self._req_out("vap")
        liq = self._req_out("liq")

        m_in = inc.m.value
        p_in = inc.p.value
        h_in = inc.h.value

        p_out = p_in - self.dp

        h_v = props.h_px(p_out, self.x_vap_target)
        h_l = props.h_px(p_out, self.x_liq_target)

        eqs: List[Equation] = []
        eqs.append(Equation(f"{self.name}.p_vap", vap.p.value - p_out, scale=max(1e5, abs(p_out))))
        eqs.append(Equation(f"{self.name}.p_liq", liq.p.value - p_out, scale=max(1e5, abs(p_out))))
        eqs.append(Equation(f"{self.name}.h_vap_target", vap.h.value - h_v, scale=max(1e5, abs(h_v))))
        eqs.append(Equation(f"{self.name}.h_liq_target", liq.h.value - h_l, scale=max(1e5, abs(h_l))))

        eqs.append(Equation(f"{self.name}.mass", m_in - (vap.m.value + liq.m.value), scale=max(1.0, abs(m_in))))
        eqs.append(
            Equation(
                f"{self.name}.energy",
                m_in * h_in - (vap.m.value * vap.h.value + liq.m.value * liq.h.value),
                scale=max(1e6, abs(m_in * h_in)),
            )
        )
        return eqs
