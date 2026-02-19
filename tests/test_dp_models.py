import pytest
pytest.importorskip("iapws")

from systems_th.props import WaterIAPWS
from systems_th.correlations.pressure_drop import dp_pipe


def test_dp_models_monotonic_with_length():
    props = WaterIAPWS()
    m = 100.0
    p_in = 7e6
    # Pick a subcooled-ish enthalpy (approx) and a slightly higher for outlet
    h_in = props.h_pT(p_in, 540.0)
    h_out = h_in + 2e5
    p_out = p_in - 1e5

    dp1 = dp_pipe(m, p_in, h_in, p_out, h_out, props, L=1.0, D=0.1, eps=1e-5, A=0.01).total
    dp2 = dp_pipe(m, p_in, h_in, p_out, h_out, props, L=2.0, D=0.1, eps=1e-5, A=0.01).total
    assert dp2 > dp1


def test_chisholm_runs():
    props = WaterIAPWS()
    m = 50.0
    p_in = 7e6
    # Force a 2-phase-ish state by picking sat mid quality
    h_mid = props.h_px(p_in, 0.5)
    p_out = p_in - 2e5
    h_out = h_mid

    dp_h = dp_pipe(m, p_in, h_mid, p_out, h_out, props, L=5.0, D=0.05, eps=1e-5, A=0.002, two_phase_friction="homogeneous").total
    dp_c = dp_pipe(m, p_in, h_mid, p_out, h_out, props, L=5.0, D=0.05, eps=1e-5, A=0.002, two_phase_friction="chisholm").total
    assert dp_h > 0
    assert dp_c > 0
