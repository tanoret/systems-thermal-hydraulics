"""Fixed-power, fixed-core-flow operating point — venturi recirculation topology.

Physical layout
---------------
The orifice (mixing) plate sits 3 ft above the core exit and acts as both a
flow restriction and a phase splitter:

  - Vapor fraction rises into the chimney (natural convection draft).
  - Liquid fraction drains back down and is sucked into the downcomer by a
    venturi (jet-pump) located at the corresponding elevation.

The downcomer is therefore split into two sections with the venturi in between:

  SteamMixer ──► Downcomer_upper ──► Venturi ──► Downcomer_lower ──► Core
                                         ▲                               │
                              liquid return from                     PostCorePipe
                              OrificePhase separator                      │
                                                                     OrificePlate (dp)
                                                                          │
                                                                    OrificePhase (Separator)
                                                                     /           \\
                                                                 vapor          liquid ─► Venturi
                                                                   │
                                                                Chimney
                                                                   │
                                                               SteamSep (Separator)
                                                              /           \\
                                                          vapor           liquid
                                                            │               │
                                                         Turbine       SteamMixer (inlet "a")
                                                            │
                                               Condenser ─► Pump ─► Heater ─► SteamMixer (inlet "b")

Mass balance insight
--------------------
Denote:
  m_core  = core inlet mass flow (prescribed)
  m_vap   = vapor extracted at OrificePhase (≈ m_core × quality)
  m_liq   = liquid returned to Venturi     (= m_core - m_vap)

From the global balance, the SteamMixer outlet (upper-downcomer flow) equals
m_vap (the net steam leaving the primary loop). The Venturi adds m_liq on top,
so the core always sees m_core. The recirculation ratio is m_liq / m_vap.

Fixed quantities
----------------
  c_cond_pump.p = p_cond    (LP pressure anchor)
  c_pump_heat.p = p_reactor (HP pressure anchor)
  c_vent_dclow.m = m_core   (core inlet / venturi outlet mass flow)
  core.Q_var     fixed      (prescribed thermal power)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from systems_th import Network
from systems_th.components import (
    Pipe, CoreChannel, OrificePlate, Separator,
    Turbine, Condenser, Pump, Heater, Mixer,
)
from systems_th.solver import SolveOptions


def main():
    # -------------------------------------------------------------------------
    # Operating conditions
    # -------------------------------------------------------------------------
    p_reactor = 7.0e6    # Pa  – HP (reactor) side pressure reference
    p_cond    = 1.0e5    # Pa  – LP (condenser) side pressure reference
    Q_core    = 2.0e8    # W   – prescribed reactor thermal power
    m_core    = 700.0    # kg/s – prescribed core inlet mass flow rate

    # -------------------------------------------------------------------------
    # Orifice / mixing-plate geometry
    # -------------------------------------------------------------------------
    n_holes  = 50
    D_hole   = 0.04                                   # hole diameter [m]
    Cd_plate = 0.61                                   # discharge coefficient (sharp-edged)
    A_holes  = n_holes * math.pi * D_hole**2 / 4.0   # total open area [m²]

    # -------------------------------------------------------------------------
    # Mass-flow guesses
    # Assuming ~15 % steam quality at the orifice plate:
    #   m_vap ≈ m_core × 0.15  →  net steam-cycle flow through SteamMixer
    #   m_liq ≈ m_core × 0.85  →  liquid recirculated through Venturi
    # -------------------------------------------------------------------------
    m_vap = m_core * 0.15   # vapor to chimney (≈ SteamMixer outlet)
    m_liq = m_core - m_vap  # liquid returned to Venturi

    # -------------------------------------------------------------------------
    # Components
    # -------------------------------------------------------------------------
    nw = Network()

    # Feedwater mixer: receives SteamSep liquid (a) + Heater return (b)
    steam_mixer = Mixer("SteamMixer")

    # Upper downcomer: from SteamMixer down to the Venturi injection point
    downcomer_upper = Pipe(
        "Downcomer_upper",
        L=10.0, D=1.2, eps=1e-5, K=1.5, dz=-10.0,
        two_phase_friction="homogeneous",
        include_acceleration=True,
    )

    # Venturi (jet-pump): mixes the downcomer flow with the orifice liquid return
    venturi = Mixer("Venturi")

    # Lower downcomer: from Venturi down to the lower plenum / core inlet
    downcomer_lower = Pipe(
        "Downcomer_lower",
        L=2.0, D=1.2, eps=1e-5, K=0.5, dz=-2.0,
        two_phase_friction="homogeneous",
        include_acceleration=True,
    )

    core = CoreChannel(
        "Core",
        L=4.0, D=0.08, A=0.30, eps=1e-5, dz=4.0,
        K=5.0, K_bundle=25.0, K_grid=2.0, n_grids=6,
        two_phase_friction="homogeneous",
        include_acceleration=True,
    )

    # Short riser from core exit to orifice plate: 3 ft vertically
    post_core = Pipe(
        "PostCorePipe",
        L=3 * 0.3048, D=1.2, eps=1e-5, K=0.0, dz=3 * 0.3048,
        two_phase_friction="homogeneous",
        include_acceleration=True,
    )

    # Orifice plate: flow restriction computed from hole geometry (isenthalpic dp)
    orifice = OrificePlate("OrificePlate", Cd=Cd_plate, A=A_holes)

    # Phase splitter at orifice plate: vapor rises to chimney, liquid returns to Venturi
    orifice_sep = Separator(
        "OrificePhase",
        dp=0.0,
        x_vap_target=0.999,
        x_liq_target=0.001,
    )

    # Chimney: carries near-pure vapor upward by natural convection
    chimney = Pipe(
        "Chimney",
        L=10.0, D=1.2, eps=1e-5, K=2.0, dz=10.0,
        two_phase_friction="homogeneous",
        include_acceleration=True,
    )

    # Final steam separator at chimney exit: polishes quality before turbine
    steam_sep = Separator("SteamSep", dp=2.0e5, x_vap_target=0.995, x_liq_target=0.005)

    turbine   = Turbine("Turbine",        eta_is=0.85, p_out=p_cond)
    condenser = Condenser("Condenser",    x_out=0.0,   p_out=p_cond)
    pump      = Pump("Pump",              eta=0.80,    p_out=p_reactor)
    heater    = Heater("FeedwaterHeater", T_out=540.0)   # K (~267 °C)

    for c in [steam_mixer, downcomer_upper, venturi, downcomer_lower,
              core, post_core, orifice, orifice_sep,
              chimney, steam_sep, turbine, condenser, pump, heater]:
        nw.add_component(c)

    # -------------------------------------------------------------------------
    # Connections – primary loop
    # -------------------------------------------------------------------------
    c_mix_dcup  = nw.connect(steam_mixer,    "out", downcomer_upper, "in",
                             "c_mix_dcup",
                             m_guess=m_vap,        p_guess=p_reactor,     h_guess=1.20e6)

    nw.connect(downcomer_upper, "out", venturi,         "dc",
                             "c_dcup_vent",
                             m_guess=m_vap,        p_guess=p_reactor-1e5, h_guess=1.20e6)

    # Venturi outlet = lower-downcomer inlet: this connection carries m_core (fixed)
    c_vent_dclow = nw.connect(venturi,        "out", downcomer_lower, "in",
                              "c_vent_dclow",
                              m_guess=m_core,       p_guess=p_reactor-1e5, h_guess=1.10e6)

    c_dclow_core = nw.connect(downcomer_lower, "out", core,           "in",
                              "c_dclow_core",
                              m_guess=m_core,       p_guess=p_reactor-2e5, h_guess=1.10e6)

    c_core_post  = nw.connect(core,           "out", post_core,       "in",
                              "c_core_post",
                              m_guess=m_core,       p_guess=p_reactor-3e5, h_guess=1.35e6)

    c_post_orif  = nw.connect(post_core,      "out", orifice,         "in",
                              "c_post_orif",
                              m_guess=m_core,       p_guess=p_reactor-4e5, h_guess=1.35e6)

    c_orif_osep  = nw.connect(orifice,        "out", orifice_sep,     "in",
                              "c_orif_osep",
                              m_guess=m_core,       p_guess=p_reactor-5e5, h_guess=1.35e6)

    # Orifice separator outlets
    c_osep_vap   = nw.connect(orifice_sep,    "vap", chimney,         "in",
                              "c_osep_vap",
                              m_guess=m_vap,        p_guess=p_reactor-5e5, h_guess=2.75e6)

    c_liq_ret    = nw.connect(orifice_sep,    "liq", venturi,         "liq",
                              "c_liq_ret",
                              m_guess=m_liq,        p_guess=p_reactor-5e5, h_guess=1.27e6)

    c_chim_ssep  = nw.connect(chimney,        "out", steam_sep,       "in",
                              "c_chim_ssep",
                              m_guess=m_vap,        p_guess=p_reactor-7e5, h_guess=2.77e6)

    # SteamSep outlets: liquid back to SteamMixer, vapor to turbine
    m_ssep_liq = m_vap * 0.05
    m_ssep_vap = m_vap - m_ssep_liq

    c_ssep_liq   = nw.connect(steam_sep,      "liq", steam_mixer,     "a",
                              "c_ssep_liq",
                              m_guess=m_ssep_liq,   p_guess=p_reactor-8e5, h_guess=1.27e6)

    # -------------------------------------------------------------------------
    # Connections – steam (Rankine) cycle
    # -------------------------------------------------------------------------
    c_ssep_vap   = nw.connect(steam_sep,      "vap", turbine,         "in",
                              "c_ssep_vap",
                              m_guess=m_ssep_vap,   p_guess=p_reactor-8e5, h_guess=2.77e6)

    c_turb_cond  = nw.connect(turbine,        "out", condenser,       "in",
                              "c_turb_cond",
                              m_guess=m_ssep_vap,   p_guess=p_cond,        h_guess=2.20e6)

    c_cond_pump  = nw.connect(condenser,      "out", pump,            "in",
                              "c_cond_pump",
                              m_guess=m_ssep_vap,   p_guess=p_cond,        h_guess=4.00e5)

    c_pump_heat  = nw.connect(pump,           "out", heater,          "in",
                              "c_pump_heat",
                              m_guess=m_ssep_vap,   p_guess=p_reactor,     h_guess=4.50e5)

    c_heat_mix   = nw.connect(heater,         "out", steam_mixer,     "b",
                              "c_heat_mix",
                              m_guess=m_ssep_vap,   p_guess=p_reactor,     h_guess=1.20e6)

    # -------------------------------------------------------------------------
    # Boundary conditions
    # -------------------------------------------------------------------------
    c_cond_pump.p.fix(p_cond)      # LP pressure anchor
    c_pump_heat.p.fix(p_reactor)   # HP pressure anchor

    # Fix core inlet mass flow at the Venturi outlet (the physically prescribed quantity)
    c_vent_dclow.m.fix(m_core)

    # Fixed thermal power; no exit-void-fraction constraint is added
    core.set_power(Q_core)

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    print(nw.summary())
    print(f"\nPrescribed core power     : {Q_core:.3e} W")
    print(f"Prescribed core inlet flow: {m_core:.1f} kg/s")
    print(f"Mixing plate              : {n_holes} holes × ø{D_hole*1e3:.1f} mm"
          f"  (Cd={Cd_plate}, A_open={A_holes*1e4:.2f} cm²)")
    print("\nSolving...\n")

    result = nw.solve(SolveOptions(max_iter=50, tol=1e-7, verbose=True))

    print("\n--- Solver result ---")
    print(result)

    # -------------------------------------------------------------------------
    # Post-processing
    # -------------------------------------------------------------------------
    props = nw.props

    T_core_in      = props.T_ph(c_dclow_core.p.value, c_dclow_core.h.value)
    T_core_out     = props.T_ph(c_core_post.p.value,  c_core_post.h.value)
    x_core_out     = props.quality_ph(c_core_post.p.value, c_core_post.h.value)
    alpha_core_out = props.void_fraction_ph(c_core_post.p.value, c_core_post.h.value)

    print("\n--- Core ---")
    print(f"  Power                        : {core.Q_var.value:.3e} W")
    print(f"  Inlet temperature            : {T_core_in - 273.15:.1f} °C")
    print(f"  Outlet temperature           : {T_core_out - 273.15:.1f} °C")
    print(f"  Outlet steam quality x       : {x_core_out:.4f}")
    print(f"  Outlet void fraction α       : {alpha_core_out:.4f}")

    m_dc_upper   = c_mix_dcup.m.value          # mass conservation: same as c_dcup_vent
    m_liq_return = c_liq_ret.m.value
    recirc_ratio = m_liq_return / m_dc_upper if m_dc_upper > 0 else float("nan")

    print("\n--- Venturi / recirculation ---")
    print(f"  Core inlet flow              : {c_vent_dclow.m.value:.1f} kg/s")
    print(f"  Upper downcomer (net) flow   : {m_dc_upper:.1f} kg/s")
    print(f"  Liquid return from orifice   : {m_liq_return:.1f} kg/s")
    print(f"  Recirculation ratio          : {recirc_ratio:.2f}")

    dp_orifice = c_post_orif.p.value - c_orif_osep.p.value
    T_chim_out = props.T_ph(c_chim_ssep.p.value, c_chim_ssep.h.value)

    print("\n--- Steam cycle flow ---")
    print(f"  Orifice plate dp             : {dp_orifice/1e3:.1f} kPa")
    print(f"  Vapor from orifice plate     : {c_osep_vap.m.value:.1f} kg/s")
    print(f"  Chimney exit temperature     : {T_chim_out - 273.15:.1f} °C")
    print(f"  Steam to turbine             : {c_ssep_vap.m.value:.1f} kg/s")
    print(f"  SteamSep liquid return       : {c_ssep_liq.m.value:.1f} kg/s")

    T_fw = props.T_ph(c_heat_mix.p.value, c_heat_mix.h.value)

    print("\n--- Energy balance ---")
    print(f"  Feedwater temperature        : {T_fw - 273.15:.1f} °C")
    print(f"  Condenser heat rejected      : {condenser.heat_rejected():.3e} W")
    print(f"  Pump shaft power             : {pump.shaft_power():.3e} W")
    print(f"  Feedwater heater duty        : {heater.heat_added():.3e} W")

    # -------------------------------------------------------------------------
    # Cycle efficiency and electric power
    # -------------------------------------------------------------------------
    W_turb_gross = c_ssep_vap.m.value * (c_ssep_vap.h.value - c_turb_cond.h.value)
    W_pump_shaft = pump.shaft_power()
    W_net_mech   = W_turb_gross - W_pump_shaft
    eta_gen      = 0.98                        # generator mechanical → electrical
    W_electric   = W_net_mech * eta_gen
    eta_net      = W_electric / Q_core

    print("\n--- Cycle performance ---")
    print(f"  Turbine gross power          : {W_turb_gross:.3e} W")
    print(f"  Pump parasitic power         : {W_pump_shaft:.3e} W")
    print(f"  Net mechanical power         : {W_net_mech:.3e} W")
    print(f"  Generator efficiency         : {eta_gen*100:.1f} %")
    print(f"  Net electric power           : {W_electric/1e6:.2f} MWe")
    print(f"  Net cycle efficiency         : {eta_net*100:.2f} %")


if __name__ == "__main__":
    main()
