from __future__ import annotations

from dataclasses import dataclass
import math

from .friction import haaland_friction_factor


@dataclass(frozen=True)
class DpBreakdown:
    dp_fric: float
    dp_form: float
    dp_grav: float
    dp_acc: float

    @property
    def total(self) -> float:
        return self.dp_fric + self.dp_form + self.dp_grav + self.dp_acc


def _area_from_D(D: float) -> float:
    return math.pi * (D ** 2) / 4.0


def _dp_form_loss(m_dot: float, rho: float, K: float, A: float) -> float:
    if A <= 0 or rho <= 0:
        return 0.0
    G = m_dot / A
    return K * (G ** 2) / (2.0 * rho)


def _dp_gravity(rho: float, dz: float) -> float:
    return rho * 9.80665 * dz


def _dp_acceleration_same_area(m_dot: float, A: float, rho_in: float, rho_out: float) -> float:
    if A <= 0:
        return 0.0
    G = m_dot / A
    # dp_acc = G^2(1/rho_out - 1/rho_in)
    if rho_in <= 0 or rho_out <= 0:
        return 0.0
    return (G ** 2) * (1.0 / rho_out - 1.0 / rho_in)


def _dp_friction_homogeneous(
    m_dot: float, rho_mix: float, mu_mix: float, L: float, D: float, eps: float, A: float
) -> float:
    if A <= 0 or D <= 0 or L <= 0 or rho_mix <= 0 or mu_mix <= 0:
        return 0.0
    G = m_dot / A
    # Re = rho*v*D/mu, with v=G/rho => Re = G*D/mu
    Re = abs(G * D / mu_mix)
    f = haaland_friction_factor(Re, eps / D)
    return f * (L / D) * (G ** 2) / (2.0 * rho_mix)


def _chisholm_C(Re_l0: float, Re_v0: float) -> float:
    # Common Chisholm constants based on laminar/turbulent classification
    l_lam = Re_l0 < 2000.0
    v_lam = Re_v0 < 2000.0
    if (not l_lam) and (not v_lam):
        return 20.0
    if l_lam and v_lam:
        return 5.0
    return 12.0


def _phi_l2_chisholm(x: float, rho_l: float, rho_v: float, mu_l: float, mu_v: float, Re_l0: float, Re_v0: float) -> float:
    # Lockhart-Martinelli parameter X_tt for turbulent-turbulent baseline
    x = min(max(x, 1e-8), 1.0 - 1e-8)
    if rho_l <= 0 or rho_v <= 0 or mu_l <= 0 or mu_v <= 0:
        return 1.0
    X_tt = ((1.0 - x) / x) ** 0.9 * (rho_v / rho_l) ** 0.5 * (mu_l / mu_v) ** 0.1
    C = _chisholm_C(Re_l0, Re_v0)
    return 1.0 + C / X_tt + 1.0 / (X_tt ** 2)


def _dp_friction_chisholm(
    m_dot: float,
    p_pa: float,
    h_jkg: float,
    props,
    L: float,
    D: float,
    eps: float,
    A: float,
) -> float:
    """Two-phase friction dp via Chisholm multiplier on liquid-only dp (dp_lo).

    Default behavior:
      - if x <= 0: return dp_lo (liquid)
      - if x >= 1: return dp_go (vapor), computed similarly
      - else: dp = phi_l^2 * dp_lo
    """
    if A <= 0 or D <= 0 or L <= 0:
        return 0.0

    x = props.quality_ph(p_pa, h_jkg)
    rho_l, rho_v = props.sat_rho_l_v(p_pa)
    mu_l, mu_v = props.sat_mu_l_v(p_pa)

    G = m_dot / A
    Re_l0 = abs(G * D / max(mu_l, 1e-12))
    Re_v0 = abs(G * D / max(mu_v, 1e-12))

    # liquid-only dp
    f_l0 = haaland_friction_factor(Re_l0, eps / D)
    dp_lo = f_l0 * (L / D) * (G ** 2) / (2.0 * max(rho_l, 1e-12))

    if x <= 0.0:
        return dp_lo
    if x >= 1.0:
        f_v0 = haaland_friction_factor(Re_v0, eps / D)
        dp_go = f_v0 * (L / D) * (G ** 2) / (2.0 * max(rho_v, 1e-12))
        return dp_go

    phi_l2 = _phi_l2_chisholm(x, rho_l, rho_v, mu_l, mu_v, Re_l0, Re_v0)
    return phi_l2 * dp_lo


def dp_pipe(
    m_dot: float,
    p_in: float,
    h_in: float,
    p_out: float,
    h_out: float,
    props,
    L: float,
    D: float,
    eps: float,
    K: float = 0.0,
    dz: float = 0.0,
    A: float | None = None,
    two_phase_friction: str = "homogeneous",
    include_acceleration: bool = True,
    include_gravity: bool = True,
) -> DpBreakdown:
    """Pressure drop breakdown for a 1D pipe-like element.

    Parameters
    ----------
    two_phase_friction:
        "homogeneous" (default) or "chisholm".
    """
    if A is None:
        A = _area_from_D(D)

    # Use mid-state for friction and form losses
    p_avg = 0.5 * (p_in + p_out)
    h_avg = 0.5 * (h_in + h_out)

    rho_avg = props.rho_ph(p_avg, h_avg)
    mu_avg = props.mu_ph(p_avg, h_avg)

    if two_phase_friction.lower() == "chisholm":
        dp_fric = _dp_friction_chisholm(m_dot, p_avg, h_avg, props, L, D, eps, A)
    else:
        dp_fric = _dp_friction_homogeneous(m_dot, rho_avg, mu_avg, L, D, eps, A)

    dp_form = _dp_form_loss(m_dot, rho_avg, K, A)

    dp_grav = _dp_gravity(rho_avg, dz) if include_gravity else 0.0

    if include_acceleration:
        rho_in = props.rho_ph(p_in, h_in)
        rho_out = props.rho_ph(p_out, h_out)
        dp_acc = _dp_acceleration_same_area(m_dot, A, rho_in, rho_out)
    else:
        dp_acc = 0.0

    return DpBreakdown(dp_fric=dp_fric, dp_form=dp_form, dp_grav=dp_grav, dp_acc=dp_acc)
