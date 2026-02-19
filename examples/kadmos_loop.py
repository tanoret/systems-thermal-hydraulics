from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from systems_th import Network
from systems_th.components import Pipe, CoreChannel, OrificePlate, Separator, Turbine, Condenser, Pump, Heater, Mixer
from systems_th.solver import SolveOptions


def main():
    # Representative pressures
    p_reactor = 7.0e6        # Pa
    p_cond = 1.0e5           # Pa

    nw = Network()

    mixer = Mixer("Mixer")

    downcomer = Pipe(
        "Downcomer",
        L=12.0, D=1.2, eps=1e-5, K=2.0, dz=-12.0,
        two_phase_friction="homogeneous",
        include_acceleration=True,
    )

    core = CoreChannel(
        "Core",
        L=4.0, D=0.08, A=0.30, eps=1e-5, dz=4.0,
        K=5.0, K_bundle=25.0, K_grid=2.0, n_grids=6,
        two_phase_friction="homogeneous",  # optional: "chisholm"
        include_acceleration=True,
    )

    orifice = OrificePlate("Orifice", K=50.0, A=0.30)

    chimney = Pipe(
        "Chimney",
        L=10.0, D=1.2, eps=1e-5, K=2.0, dz=10.0,
        two_phase_friction="homogeneous",
        include_acceleration=True,
    )

    separator = Separator("Separator", dp=2.0e5, x_vap_target=0.995, x_liq_target=0.005)

    turbine = Turbine("Turbine", eta_is=0.85, p_out=p_cond)
    condenser = Condenser("Condenser", x_out=0.0, p_out=p_cond)
    pump = Pump("Pump", eta=0.80, p_out=p_reactor)
    heater = Heater("FeedwaterHeater", T_out=540.0)  # K (~267 C)

    for c in [mixer, downcomer, core, orifice, chimney, separator, turbine, condenser, pump, heater]:
        nw.add_component(c)

    # Reactor loop
    c_mix = nw.connect(mixer, "out", downcomer, "in", "c_mix_dc", m_guess=3000.0, p_guess=p_reactor, h_guess=1.2e6)
    c_dc = nw.connect(downcomer, "out", core, "in", "c_dc_core", m_guess=3000.0, p_guess=p_reactor, h_guess=1.2e6)
    c_core = nw.connect(core, "out", orifice, "in", "c_core_orif", m_guess=3000.0, p_guess=p_reactor-2e5, h_guess=1.35e6)
    c_orif = nw.connect(orifice, "out", chimney, "in", "c_orif_chim", m_guess=3000.0, p_guess=p_reactor-4e5, h_guess=1.35e6)
    c_chim = nw.connect(chimney, "out", separator, "in", "c_chim_sep", m_guess=3000.0, p_guess=p_reactor-6e5, h_guess=1.40e6)
    c_sep_liq = nw.connect(separator, "liq", mixer, "a", "c_sep_liq_mix", m_guess=2700.0, p_guess=p_reactor-8e5, h_guess=1.1e6)

    # Steam cycle
    c_sep_vap = nw.connect(separator, "vap", turbine, "in", "c_sep_vap_turb", m_guess=300.0, p_guess=p_reactor-8e5, h_guess=2.8e6)
    c_turb = nw.connect(turbine, "out", condenser, "in", "c_turb_cond", m_guess=300.0, p_guess=p_cond, h_guess=2.2e6)
    c_cond = nw.connect(condenser, "out", pump, "in", "c_cond_pump", m_guess=300.0, p_guess=p_cond, h_guess=4.0e5)
    c_pump = nw.connect(pump, "out", heater, "in", "c_pump_heater", m_guess=300.0, p_guess=p_reactor, h_guess=4.5e5)
    c_heat = nw.connect(heater, "out", mixer, "b", "c_heater_mix", m_guess=300.0, p_guess=p_reactor, h_guess=1.1e6)

    # Fix reference pressures
    c_cond.p.fix(p_cond)
    c_pump.p.fix(p_reactor)

    # Target core exit void fraction ~0.4, solve for core power
    core.set_exit_void_fraction(alpha=0.40, Q_guess_w=1.0e8)

    print(nw.summary())
    print("\nSolving... (requires iapws installed)\n")

    result = nw.solve(SolveOptions(max_iter=50, tol=1e-7, verbose=True))
    print("\nResult:")
    print(result)

    props = nw.props
    alpha_core_out = props.void_fraction_ph(c_core.p.value, c_core.h.value)
    x_core_out = props.quality_ph(c_core.p.value, c_core.h.value)
    print(f"\nCore power [W]: {core.Q_var.value: .3e}")
    print(f"Core outlet quality x: {x_core_out: .4f}")
    print(f"Core outlet void fraction alpha: {alpha_core_out: .4f}")

    print(f"Steam to turbine m_dot [kg/s]: {c_sep_vap.m.value: .3f}")
    print(f"Liquid return m_dot [kg/s]: {c_sep_liq.m.value: .3f}")

    print(f"Condenser heat rejected [W]: {condenser.heat_rejected(): .3e}")
    print(f"Pump shaft power [W]: {pump.shaft_power(): .3e}")
    print(f"Heater heat added [W]: {heater.heat_added(): .3e}")


if __name__ == "__main__":
    main()
