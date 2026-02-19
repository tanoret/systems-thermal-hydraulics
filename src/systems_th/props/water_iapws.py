from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, Tuple


class WaterProps(Protocol):
    """Water/steam properties interface (SI units)."""

    # Saturation helpers
    def sat_h_l_v(self, p_pa: float) -> tuple[float, float]: ...
    def sat_rho_l_v(self, p_pa: float) -> tuple[float, float]: ...
    def sat_mu_l_v(self, p_pa: float) -> tuple[float, float]: ...
    def sat_k_l_v(self, p_pa: float) -> tuple[float, float]: ...
    def sat_cp_l_v(self, p_pa: float) -> tuple[float, float]: ...
    def T_sat_p(self, p_pa: float) -> float: ...
    def sigma_sat_p(self, p_pa: float) -> float: ...

    # Thermodynamic state
    def h_pT(self, p_pa: float, T_k: float) -> float: ...
    def T_ph(self, p_pa: float, h_jkg: float) -> float: ...
    def s_ph(self, p_pa: float, h_jkg: float) -> float: ...
    def h_ps(self, p_pa: float, s_jkgK: float) -> float: ...
    def quality_ph(self, p_pa: float, h_jkg: float) -> float: ...
    def h_px(self, p_pa: float, x: float) -> float: ...

    # Transport + mixture (HEM)
    def rho_ph(self, p_pa: float, h_jkg: float) -> float: ...
    def rho_px(self, p_pa: float, x: float) -> float: ...
    def mu_ph(self, p_pa: float, h_jkg: float) -> float: ...
    def k_ph(self, p_pa: float, h_jkg: float) -> float: ...
    def cp_ph(self, p_pa: float, h_jkg: float) -> float: ...
    def void_fraction_ph(self, p_pa: float, h_jkg: float) -> float: ...


def _round(x: float, nd: int = 8) -> float:
    return round(float(x), nd)


@dataclass(frozen=True)
class WaterIAPWS:
    """IAPWS97 property wrapper using SI units.

    Units (IAPWS97 python package)
    ------------------------------
    - P [MPa], h [kJ/kg], s [kJ/kg/K], cp [kJ/kg/K]
    - μ [Pa·s], k [W/m/K], σ [N/m]

    This wrapper converts to/from SI (Pa, J/kg, J/kg-K).

    Notes
    -----
    Uses LRU caching on the expensive IAPWS97 state calls.
    """

    # Critical pressure of water [Pa] (IAPWS IF97)
    Pc_pa: float = 22.064e6
    # Molar mass [kg/mol]
    molar_mass: float = 0.018015268

    @staticmethod
    def _require_iapws():
        try:
            from iapws import IAPWS97  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "The 'iapws' package is required. Install with: pip install iapws"
            ) from e
        return IAPWS97

    @staticmethod
    def _pa_to_mpa(p_pa: float) -> float:
        return p_pa * 1e-6

    @staticmethod
    def _mpa_to_pa(p_mpa: float) -> float:
        return p_mpa * 1e6

    @staticmethod
    def _jkg_to_kjkg(h_jkg: float) -> float:
        return h_jkg * 1e-3

    @staticmethod
    def _kjkg_to_jkg(h_kjkg: float) -> float:
        return h_kjkg * 1e3

    @staticmethod
    def _jkgK_to_kjkgK(s_jkgK: float) -> float:
        return s_jkgK * 1e-3

    @staticmethod
    def _kjkgK_to_jkgK(s_kjkgK: float) -> float:
        return s_kjkgK * 1e3

    # -------------------------
    # Cached IAPWS state calls
    # -------------------------

    @staticmethod
    @lru_cache(maxsize=8192)
    def _state_px_cached(P_mpa: float, x: float):
        IAPWS97 = WaterIAPWS._require_iapws()
        return IAPWS97(P=_round(P_mpa, 8), x=_round(x, 8))

    @staticmethod
    @lru_cache(maxsize=8192)
    def _state_ph_cached(P_mpa: float, h_kjkg: float):
        IAPWS97 = WaterIAPWS._require_iapws()
        return IAPWS97(P=_round(P_mpa, 8), h=_round(h_kjkg, 6))

    @staticmethod
    @lru_cache(maxsize=8192)
    def _state_ps_cached(P_mpa: float, s_kjkgK: float):
        IAPWS97 = WaterIAPWS._require_iapws()
        return IAPWS97(P=_round(P_mpa, 8), s=_round(s_kjkgK, 7))

    @staticmethod
    @lru_cache(maxsize=8192)
    def _state_pT_cached(P_mpa: float, T_k: float):
        IAPWS97 = WaterIAPWS._require_iapws()
        return IAPWS97(P=_round(P_mpa, 8), T=_round(T_k, 6))

    # -------------------------
    # Saturation
    # -------------------------

    def T_sat_p(self, p_pa: float) -> float:
        P = self._pa_to_mpa(p_pa)
        # saturation line: use x=0.5
        w = self._state_px_cached(P, 0.5)
        return float(w.T)

    def sigma_sat_p(self, p_pa: float) -> float:
        P = self._pa_to_mpa(p_pa)
        w = self._state_px_cached(P, 0.5)
        return float(w.sigma)

    def sat_h_l_v(self, p_pa: float) -> tuple[float, float]:
        P = self._pa_to_mpa(p_pa)
        w_l = self._state_px_cached(P, 0.0)
        w_v = self._state_px_cached(P, 1.0)
        return (self._kjkg_to_jkg(w_l.h), self._kjkg_to_jkg(w_v.h))

    def sat_rho_l_v(self, p_pa: float) -> tuple[float, float]:
        P = self._pa_to_mpa(p_pa)
        w_l = self._state_px_cached(P, 0.0)
        w_v = self._state_px_cached(P, 1.0)
        return (float(w_l.rho), float(w_v.rho))

    def sat_mu_l_v(self, p_pa: float) -> tuple[float, float]:
        P = self._pa_to_mpa(p_pa)
        w_l = self._state_px_cached(P, 0.0)
        w_v = self._state_px_cached(P, 1.0)
        return (float(w_l.mu), float(w_v.mu))

    def sat_k_l_v(self, p_pa: float) -> tuple[float, float]:
        P = self._pa_to_mpa(p_pa)
        w_l = self._state_px_cached(P, 0.0)
        w_v = self._state_px_cached(P, 1.0)
        return (float(w_l.k), float(w_v.k))

    def sat_cp_l_v(self, p_pa: float) -> tuple[float, float]:
        P = self._pa_to_mpa(p_pa)
        w_l = self._state_px_cached(P, 0.0)
        w_v = self._state_px_cached(P, 1.0)
        # cp in kJ/kg/K
        return (float(w_l.cp) * 1e3, float(w_v.cp) * 1e3)

    # -------------------------
    # Thermodynamic state
    # -------------------------

    def h_pT(self, p_pa: float, T_k: float) -> float:
        P = self._pa_to_mpa(p_pa)
        w = self._state_pT_cached(P, T_k)
        return self._kjkg_to_jkg(w.h)

    def T_ph(self, p_pa: float, h_jkg: float) -> float:
        x = self.quality_ph(p_pa, h_jkg)
        if 0.0 < x < 1.0:
            return self.T_sat_p(p_pa)
        P = self._pa_to_mpa(p_pa)
        h = self._jkg_to_kjkg(h_jkg)
        w = self._state_ph_cached(P, h)
        return float(w.T)

    def s_ph(self, p_pa: float, h_jkg: float) -> float:
        P = self._pa_to_mpa(p_pa)
        h = self._jkg_to_kjkg(h_jkg)
        w = self._state_ph_cached(P, h)
        return self._kjkgK_to_jkgK(w.s)

    def h_ps(self, p_pa: float, s_jkgK: float) -> float:
        P = self._pa_to_mpa(p_pa)
        s = self._jkgK_to_kjkgK(s_jkgK)
        w = self._state_ps_cached(P, s)
        return self._kjkg_to_jkg(w.h)

    def quality_ph(self, p_pa: float, h_jkg: float) -> float:
        # Robust quality from sat enthalpies
        h_l, h_v = self.sat_h_l_v(p_pa)
        if h_jkg <= h_l:
            return 0.0
        if h_jkg >= h_v:
            return 1.0
        return max(0.0, min(1.0, (h_jkg - h_l) / (h_v - h_l)))

    def h_px(self, p_pa: float, x: float) -> float:
        x = max(0.0, min(1.0, float(x)))
        h_l, h_v = self.sat_h_l_v(p_pa)
        return (1.0 - x) * h_l + x * h_v

    # -------------------------
    # Transport + mixture (HEM)
    # -------------------------

    def rho_px(self, p_pa: float, x: float) -> float:
        x = max(0.0, min(1.0, float(x)))
        rho_l, rho_v = self.sat_rho_l_v(p_pa)
        inv = x / rho_v + (1.0 - x) / rho_l
        return 1.0 / inv

    def rho_ph(self, p_pa: float, h_jkg: float) -> float:
        x = self.quality_ph(p_pa, h_jkg)
        if 0.0 < x < 1.0:
            return self.rho_px(p_pa, x)
        P = self._pa_to_mpa(p_pa)
        h = self._jkg_to_kjkg(h_jkg)
        w = self._state_ph_cached(P, h)
        return float(w.rho)

    def void_fraction_ph(self, p_pa: float, h_jkg: float) -> float:
        x = self.quality_ph(p_pa, h_jkg)
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        rho_l, rho_v = self.sat_rho_l_v(p_pa)
        vg = x / rho_v
        vl = (1.0 - x) / rho_l
        return vg / (vg + vl)

    def mu_ph(self, p_pa: float, h_jkg: float) -> float:
        x = self.quality_ph(p_pa, h_jkg)
        if 0.0 < x < 1.0:
            mu_l, mu_v = self.sat_mu_l_v(p_pa)
            alpha = self.void_fraction_ph(p_pa, h_jkg)
            # Simple alpha-weighted mixture (smooth and bounded). Upgrade later if needed.
            return (1.0 - alpha) * mu_l + alpha * mu_v
        P = self._pa_to_mpa(p_pa)
        h = self._jkg_to_kjkg(h_jkg)
        w = self._state_ph_cached(P, h)
        return float(w.mu)

    def k_ph(self, p_pa: float, h_jkg: float) -> float:
        x = self.quality_ph(p_pa, h_jkg)
        if 0.0 < x < 1.0:
            k_l, k_v = self.sat_k_l_v(p_pa)
            alpha = self.void_fraction_ph(p_pa, h_jkg)
            return (1.0 - alpha) * k_l + alpha * k_v
        P = self._pa_to_mpa(p_pa)
        h = self._jkg_to_kjkg(h_jkg)
        w = self._state_ph_cached(P, h)
        return float(w.k)

    def cp_ph(self, p_pa: float, h_jkg: float) -> float:
        x = self.quality_ph(p_pa, h_jkg)
        if 0.0 < x < 1.0:
            # cp is not well-defined in 2-phase; return saturated-liquid cp for correlation use.
            cp_l, _ = self.sat_cp_l_v(p_pa)
            return cp_l
        P = self._pa_to_mpa(p_pa)
        h = self._jkg_to_kjkg(h_jkg)
        w = self._state_ph_cached(P, h)
        return float(w.cp) * 1e3
