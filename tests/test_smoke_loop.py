import pytest
pytest.importorskip("iapws")

from systems_th import Network
from systems_th.components import Pipe, Mixer, CoreChannel, Separator, OrificePlate, Turbine, Condenser, Pump, Heater
from systems_th.solver import SolveOptions


def test_systems_loop_smoke():
    p_reactor = 7e6
    p_cond = 1e5

    nw = Network()

    mixer = Mixer("Mixer")
    downcomer = Pipe("Downcomer", L=5.0, D=1.0, K=1.0, dz=-5.0)
    core = CoreChannel("Core", L=2.0, D=0.05, A=0.1, K=2.0, dz=2.0, K_bundle=10.0, two_phase_friction="homogeneous")
    orif = OrificePlate("Orifice", K=10.0, A=0.1)
    chimney = Pipe("Chimney", L=5.0, D=1.0, K=1.0, dz=5.0)
    sep = Separator("Separator", dp=1e5, x_vap_target=0.99, x_liq_target=0.01)
    turb = Turbine("Turbine", eta_is=0.85, p_out=p_cond)
    cond = Condenser("Condenser", p_out=p_cond, x_out=0.0)
    pump = Pump("Pump", p_out=p_reactor, eta=0.8)
    heater = Heater("Heater", T_out=520.0)

    for c in [mixer, downcomer, core, orif, chimney, sep, turb, cond, pump, heater]:
        nw.add_component(c)

    c_mix = nw.connect(mixer, "out", downcomer, "in", "c_mix_dc", m_guess=1000, p_guess=p_reactor, h_guess=1.2e6)
    c_dc = nw.connect(downcomer, "out", core, "in", "c_dc_core", m_guess=1000, p_guess=p_reactor, h_guess=1.2e6)
    c_core = nw.connect(core, "out", orif, "in", "c_core_orif", m_guess=1000, p_guess=p_reactor-1e5, h_guess=1.35e6)
    c_or = nw.connect(orif, "out", chimney, "in", "c_or_chim", m_guess=1000, p_guess=p_reactor-2e5, h_guess=1.35e6)
    c_ch = nw.connect(chimney, "out", sep, "in", "c_ch_sep", m_guess=1000, p_guess=p_reactor-3e5, h_guess=1.40e6)
    c_liq = nw.connect(sep, "liq", mixer, "a", "c_liq_mix", m_guess=900, p_guess=p_reactor-4e5, h_guess=1.1e6)

    c_vap = nw.connect(sep, "vap", turb, "in", "c_vap_turb", m_guess=100, p_guess=p_reactor-4e5, h_guess=2.8e6)
    c_t = nw.connect(turb, "out", cond, "in", "c_t_cond", m_guess=100, p_guess=p_cond, h_guess=2.2e6)
    c_c = nw.connect(cond, "out", pump, "in", "c_cond_pump", m_guess=100, p_guess=p_cond, h_guess=4e5)
    c_p = nw.connect(pump, "out", heater, "in", "c_pump_heat", m_guess=100, p_guess=p_reactor, h_guess=5e5)
    c_h = nw.connect(heater, "out", mixer, "b", "c_heat_mix", m_guess=100, p_guess=p_reactor, h_guess=1.1e6)

    c_c.p.fix(p_cond)
    c_p.p.fix(p_reactor)

    core.set_exit_void_fraction(alpha=0.3, Q_guess_w=5e7)

    res = nw.solve(SolveOptions(max_iter=40, tol=1e-6, verbose=False))
    assert res.converged
