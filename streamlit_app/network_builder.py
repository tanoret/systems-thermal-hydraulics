"""
network_builder.py
------------------
Builds the venturi-recirculation kadmos_th Network from a flat parameter
dictionary and exposes helpers for extracting post-solve state data.
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kadmos_th import Network
from kadmos_th.components import (
    Pipe, CoreChannel, OrificePlate, Separator,
    Turbine, Condenser, Pump, Heater, Mixer,
)
from kadmos_th.solver import SolveOptions, SolveResult
from kadmos_th.props import WaterIAPWS

# ── station sequence (label, connection key, branch) ──────────────────────────
STATION_SEQUENCE: list[tuple[str, str, str]] = [
    ("Mixer → DC upper",      "c_mix_dcup",   "Primary loop"),
    ("Venturi outlet",         "c_vent_dclow",  "Primary loop"),
    ("Core inlet",             "c_dclow_core",  "Primary loop"),
    ("Core outlet",            "c_core_post",   "Primary loop"),
    ("Post-core",              "c_post_orif",   "Primary loop"),
    ("Orifice plate out",      "c_orif_osep",   "Primary loop"),
    ("Chimney inlet",          "c_osep_vap",    "Primary loop"),
    ("Chimney outlet",         "c_chim_ssep",   "Primary loop"),
    ("Turbine inlet",          "c_ssep_vap",    "Steam cycle"),
    ("Turbine outlet",         "c_turb_cond",   "Steam cycle"),
    ("Condenser outlet",       "c_cond_pump",   "Steam cycle"),
    ("Pump outlet",            "c_pump_heat",   "Steam cycle"),
    ("Heater outlet",          "c_heat_mix",    "Steam cycle"),
    ("Liquid return",          "c_liq_ret",     "Liquid return"),
    ("SteamSep liquid",        "c_ssep_liq",    "Liquid return"),
]


def default_params() -> dict[str, Any]:
    """Return default operating/geometry parameter dictionary."""
    return {
        # Operating conditions
        "p_reactor":    7.0e6,   # Pa
        "p_cond":       1.0e5,   # Pa
        "Q_core":       2.0e8,   # W
        "m_core":       700.0,   # kg/s
        # Orifice / mixing plate
        "n_holes":      50,
        "D_hole":       0.04,    # m
        "Cd_plate":     0.61,
        # Post-core riser
        "post_core_ft": 3.0,     # feet
        # Core channel
        "core_L":       4.0,
        "core_D":       0.08,
        "core_A":       0.30,
        "core_eps":     1e-5,
        "core_dz":      4.0,
        "core_K":       5.0,
        "core_K_bundle": 25.0,
        "core_K_grid":  2.0,
        "core_n_grids": 6,
        "core_two_phase": "homogeneous",
        # Downcomer (split at venturi)
        "dc_L_upper":   10.0,
        "dc_L_lower":   2.0,
        "dc_D":         1.2,
        "dc_eps":       1e-5,
        "dc_K_upper":   1.5,
        "dc_K_lower":   0.5,
        # Chimney
        "chim_L":       10.0,
        "chim_D":       1.2,
        "chim_eps":     1e-5,
        "chim_K":       2.0,
        # Orifice-phase separator
        "orif_x_vap":   0.999,
        "orif_x_liq":   0.001,
        # Steam separator
        "ssep_dp":      2.0e5,
        "ssep_x_vap":   0.995,
        "ssep_x_liq":   0.005,
        # Steam-cycle machinery
        "turb_eta":     0.85,
        "pump_eta":     0.80,
        "heater_T_K":   540.0,   # K
        # Plant
        "eta_gen":      0.98,
    }


def build_and_solve(
    params: dict[str, Any],
    max_iter: int = 50,
    tol: float = 1e-7,
) -> tuple[Network, SolveResult, dict[str, Any], str]:
    """
    Construct the network, solve it, and return
    (network, solve_result, connection_refs_dict, solver_log_string).
    """
    p = params
    p_reactor = p["p_reactor"]
    p_cond    = p["p_cond"]
    Q_core    = p["Q_core"]
    m_core    = p["m_core"]

    A_holes = int(p["n_holes"]) * math.pi * p["D_hole"] ** 2 / 4.0
    m_vap   = m_core * 0.15   # initial steam-fraction guess
    m_liq   = m_core - m_vap

    nw = Network()

    steam_mixer = Mixer("SteamMixer")

    downcomer_upper = Pipe(
        "Downcomer_upper",
        L=p["dc_L_upper"], D=p["dc_D"], eps=p["dc_eps"],
        K=p["dc_K_upper"], dz=-p["dc_L_upper"],
        two_phase_friction="homogeneous", include_acceleration=True,
    )

    venturi = Mixer("Venturi")

    downcomer_lower = Pipe(
        "Downcomer_lower",
        L=p["dc_L_lower"], D=p["dc_D"], eps=p["dc_eps"],
        K=p["dc_K_lower"], dz=-p["dc_L_lower"],
        two_phase_friction="homogeneous", include_acceleration=True,
    )

    core = CoreChannel(
        "Core",
        L=p["core_L"], D=p["core_D"], A=p["core_A"],
        eps=p["core_eps"], dz=p["core_dz"],
        K=p["core_K"], K_bundle=p["core_K_bundle"],
        K_grid=p["core_K_grid"], n_grids=int(p["core_n_grids"]),
        two_phase_friction=p["core_two_phase"], include_acceleration=True,
    )

    L_post = p["post_core_ft"] * 0.3048
    post_core = Pipe(
        "PostCorePipe",
        L=L_post, D=p["dc_D"], eps=1e-5, K=0.0, dz=L_post,
        two_phase_friction="homogeneous", include_acceleration=True,
    )

    orifice = OrificePlate("OrificePlate", Cd=p["Cd_plate"], A=A_holes)

    orifice_sep = Separator(
        "OrificePhase", dp=0.0,
        x_vap_target=p["orif_x_vap"], x_liq_target=p["orif_x_liq"],
    )

    chimney = Pipe(
        "Chimney",
        L=p["chim_L"], D=p["chim_D"], eps=p["chim_eps"],
        K=p["chim_K"], dz=p["chim_L"],
        two_phase_friction="homogeneous", include_acceleration=True,
    )

    steam_sep = Separator(
        "SteamSep", dp=p["ssep_dp"],
        x_vap_target=p["ssep_x_vap"], x_liq_target=p["ssep_x_liq"],
    )

    turbine   = Turbine("Turbine",         eta_is=p["turb_eta"], p_out=p_cond)
    condenser = Condenser("Condenser",     x_out=0.0, p_out=p_cond)
    pump      = Pump("Pump",               eta=p["pump_eta"], p_out=p_reactor)
    heater    = Heater("FeedwaterHeater",  T_out=p["heater_T_K"])

    for comp in [steam_mixer, downcomer_upper, venturi, downcomer_lower,
                 core, post_core, orifice, orifice_sep,
                 chimney, steam_sep, turbine, condenser, pump, heater]:
        nw.add_component(comp)

    # ── connections ───────────────────────────────────────────────────────────
    c_mix_dcup   = nw.connect(steam_mixer,     "out", downcomer_upper, "in",
                              "c_mix_dcup",
                              m_guess=m_vap,  p_guess=p_reactor,     h_guess=1.20e6)
    nw.connect(downcomer_upper, "out", venturi, "dc", "c_dcup_vent",
               m_guess=m_vap, p_guess=p_reactor - 1e5, h_guess=1.20e6)
    c_vent_dclow = nw.connect(venturi,          "out", downcomer_lower, "in",
                              "c_vent_dclow",
                              m_guess=m_core, p_guess=p_reactor - 1e5, h_guess=1.10e6)
    c_dclow_core = nw.connect(downcomer_lower,  "out", core,           "in",
                              "c_dclow_core",
                              m_guess=m_core, p_guess=p_reactor - 2e5, h_guess=1.10e6)
    c_core_post  = nw.connect(core,             "out", post_core,      "in",
                              "c_core_post",
                              m_guess=m_core, p_guess=p_reactor - 3e5, h_guess=1.35e6)
    c_post_orif  = nw.connect(post_core,        "out", orifice,        "in",
                              "c_post_orif",
                              m_guess=m_core, p_guess=p_reactor - 4e5, h_guess=1.35e6)
    c_orif_osep  = nw.connect(orifice,          "out", orifice_sep,    "in",
                              "c_orif_osep",
                              m_guess=m_core, p_guess=p_reactor - 5e5, h_guess=1.35e6)
    c_osep_vap   = nw.connect(orifice_sep,      "vap", chimney,        "in",
                              "c_osep_vap",
                              m_guess=m_vap,  p_guess=p_reactor - 5e5, h_guess=2.75e6)
    c_liq_ret    = nw.connect(orifice_sep,      "liq", venturi,        "liq",
                              "c_liq_ret",
                              m_guess=m_liq,  p_guess=p_reactor - 5e5, h_guess=1.27e6)
    c_chim_ssep  = nw.connect(chimney,          "out", steam_sep,      "in",
                              "c_chim_ssep",
                              m_guess=m_vap,  p_guess=p_reactor - 7e5, h_guess=2.77e6)

    m_ssep_liq = m_vap * 0.05
    m_ssep_vap = m_vap * 0.95

    c_ssep_liq   = nw.connect(steam_sep,  "liq", steam_mixer, "a",
                              "c_ssep_liq",
                              m_guess=m_ssep_liq, p_guess=p_reactor - 8e5, h_guess=1.27e6)
    c_ssep_vap   = nw.connect(steam_sep,  "vap", turbine,     "in",
                              "c_ssep_vap",
                              m_guess=m_ssep_vap, p_guess=p_reactor - 8e5, h_guess=2.77e6)
    c_turb_cond  = nw.connect(turbine,    "out", condenser,   "in",
                              "c_turb_cond",
                              m_guess=m_ssep_vap, p_guess=p_cond, h_guess=2.20e6)
    c_cond_pump  = nw.connect(condenser,  "out", pump,        "in",
                              "c_cond_pump",
                              m_guess=m_ssep_vap, p_guess=p_cond, h_guess=4.00e5)
    c_pump_heat  = nw.connect(pump,       "out", heater,      "in",
                              "c_pump_heat",
                              m_guess=m_ssep_vap, p_guess=p_reactor, h_guess=4.50e5)
    c_heat_mix   = nw.connect(heater,     "out", steam_mixer, "b",
                              "c_heat_mix",
                              m_guess=m_ssep_vap, p_guess=p_reactor, h_guess=1.20e6)

    # ── boundary conditions ───────────────────────────────────────────────────
    c_cond_pump.p.fix(p_cond)
    c_pump_heat.p.fix(p_reactor)
    c_vent_dclow.m.fix(m_core)
    core.set_power(Q_core)

    # ── solve (capture stdout for the log) ────────────────────────────────────
    opts = SolveOptions(max_iter=max_iter, tol=tol, verbose=True)
    buf  = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = nw.solve(opts)
    log = buf.getvalue()

    refs: dict[str, Any] = dict(
        c_mix_dcup=c_mix_dcup,   c_vent_dclow=c_vent_dclow,
        c_dclow_core=c_dclow_core, c_core_post=c_core_post,
        c_post_orif=c_post_orif,  c_orif_osep=c_orif_osep,
        c_osep_vap=c_osep_vap,   c_liq_ret=c_liq_ret,
        c_chim_ssep=c_chim_ssep,  c_ssep_liq=c_ssep_liq,
        c_ssep_vap=c_ssep_vap,   c_turb_cond=c_turb_cond,
        c_cond_pump=c_cond_pump,  c_pump_heat=c_pump_heat,
        c_heat_mix=c_heat_mix,
        # components needed for post-processing
        core=core, condenser=condenser, pump=pump, heater=heater,
    )

    return nw, result, refs, log


def extract_states_df(refs: dict[str, Any], props: WaterIAPWS) -> pd.DataFrame:
    """Return a DataFrame with full thermodynamic state at every station."""
    rows = []
    for label, key, branch in STATION_SEQUENCE:
        if key not in refs:
            continue
        conn = refs[key]
        p_pa = conn.p.value
        h_jkg = conn.h.value
        m_kgs = conn.m.value
        T_K   = props.T_ph(p_pa, h_jkg)
        x     = props.quality_ph(p_pa, h_jkg)
        alpha = props.void_fraction_ph(p_pa, h_jkg)
        try:
            s_jkgK = props.s_ph(p_pa, h_jkg)
        except Exception:
            s_jkgK = float("nan")
        rho = props.rho_ph(p_pa, h_jkg)
        rows.append(dict(
            Station=label, Branch=branch,
            m_kgs=m_kgs, p_MPa=p_pa / 1e6, h_kJkg=h_jkg / 1e3,
            T_C=T_K - 273.15, x=x, alpha=alpha,
            s_kJkgK=s_jkgK / 1e3, rho_kgm3=rho,
        ))
    return pd.DataFrame(rows)


def compute_performance(refs: dict[str, Any], params: dict[str, Any]) -> dict[str, float]:
    """Compute key scalar performance indicators after solving."""
    sv   = refs["c_ssep_vap"]
    tc   = refs["c_turb_cond"]
    core = refs["core"]
    cond = refs["condenser"]
    pump = refs["pump"]
    heater = refs["heater"]

    W_turb  = sv.m.value * (sv.h.value - tc.h.value)
    W_pump  = pump.shaft_power()
    W_net   = W_turb - W_pump
    W_elec  = W_net * params["eta_gen"]
    eta_net = W_elec / params["Q_core"]

    m_vap  = refs["c_osep_vap"].m.value
    m_liq  = refs["c_liq_ret"].m.value
    m_dcup = refs["c_mix_dcup"].m.value
    recirc = m_liq / m_dcup if m_dcup > 1e-9 else float("nan")

    return dict(
        W_turb_gross   = W_turb,
        W_pump_shaft   = W_pump,
        W_net_mech     = W_net,
        W_electric     = W_elec,
        eta_net        = eta_net,
        Q_rejected     = cond.heat_rejected(),
        Q_heater       = heater.heat_added(),
        m_core         = refs["c_vent_dclow"].m.value,
        m_vap_orifice  = m_vap,
        m_liq_return   = m_liq,
        m_steam_turbine= sv.m.value,
        m_ssep_liq     = refs["c_ssep_liq"].m.value,
        m_dc_upper     = m_dcup,
        recirc_ratio   = recirc,
        dp_orifice_kPa = (refs["c_post_orif"].p.value - refs["c_orif_osep"].p.value) / 1e3,
    )
