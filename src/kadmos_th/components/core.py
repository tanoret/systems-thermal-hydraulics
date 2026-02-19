from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .base import Component
from ..equation import Equation
from ..variable import Variable
from ..correlations.pressure_drop import dp_pipe


@dataclass
class CoreChannel(Component):
    """Boiling core channel: heat input + two-phase-aware pressure drop.

    Ports
    -----
    - inlet:  'in'
    - outlet: 'out'

    Modes
    -----
    1) Fixed power:
        call `set_power(Q)`.
    2) Solve for power to hit exit void fraction:
        call `set_exit_void_fraction(alpha_target)` and Q becomes a free variable.

    Pressure drop model:
    - base pipe-like dp (friction/form/gravity/acceleration)
    - additional bundle and spacer-grid form losses.

    Notes
    -----
    - Uses HEM void fraction derived from (p,h).
    - Correlations can be upgraded without changing the component interface.
    """

    # Geometry / hydraulics
    L: float = 4.0
    D: float = 0.08
    A: float = 0.30
    eps: float = 1e-5
    dz: float = 4.0

    # Losses
    K: float = 0.0            # base form losses
    K_bundle: float = 0.0     # pin bundle friction as lumped K (placeholder in v0.2)
    K_grid: float = 0.0       # per-spacer-grid K
    n_grids: int = 0

    # Two-phase friction model selection
    two_phase_friction: str = "homogeneous"
    include_acceleration: bool = True

    # Optional target exit void fraction (0..1). If set, Q becomes a solver variable unless fixed.
    alpha_out_target: Optional[float] = None

    # Internal variable for heat input [W]
    Q_var: Variable = field(default_factory=lambda: Variable("core.Q", 1e8, fixed=True, lower=-1e12, upper=1e12))

    def variables(self) -> List[Variable]:
        return [self.Q_var]

    def set_power(self, Q_w: float) -> None:
        self.Q_var.fix(float(Q_w))
        self.alpha_out_target = None

    def set_exit_void_fraction(self, alpha: float, Q_guess_w: float = 1e8) -> None:
        self.alpha_out_target = float(alpha)
        self.Q_var.value = float(Q_guess_w)
        self.Q_var.unfix()

    def equations(self, props) -> List[Equation]:
        inc = self._req_in("in")
        out = self._req_out("out")

        m = inc.m.value
        p_in = inc.p.value
        h_in = inc.h.value

        # Mass
        eqs: List[Equation] = [
            Equation(f"{self.name}.mass", out.m.value - m, scale=max(1.0, abs(m))),
        ]

        # Energy (power adds enthalpy)
        dh = self.Q_var.value / m if abs(m) > 1e-9 else 0.0
        h_out_target = h_in + dh
        eqs.append(Equation(f"{self.name}.energy", out.h.value - h_out_target, scale=max(1e5, abs(h_out_target))))

        # Momentum
        p_out = out.p.value
        h_out = out.h.value

        K_total = self.K + self.K_bundle + self.n_grids * self.K_grid

        dp = dp_pipe(
            m_dot=m,
            p_in=p_in,
            h_in=h_in,
            p_out=p_out,
            h_out=h_out,
            props=props,
            L=self.L,
            D=self.D,
            eps=self.eps,
            K=K_total,
            dz=self.dz,
            A=self.A,
            two_phase_friction=self.two_phase_friction,
            include_acceleration=self.include_acceleration,
            include_gravity=True,
        )

        eqs.append(Equation(f"{self.name}.dp", (p_in - p_out) - dp.total, scale=max(1e5, abs(p_in))))

        # Optional void fraction target at outlet
        if self.alpha_out_target is not None:
            alpha_out = props.void_fraction_ph(p_out, h_out)
            eqs.append(
                Equation(
                    f"{self.name}.alpha_out",
                    alpha_out - self.alpha_out_target,
                    scale=max(1e-2, abs(self.alpha_out_target)),
                )
            )

        return eqs
