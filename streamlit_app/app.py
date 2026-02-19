"""
app.py — kadmos-TH Streamlit GUI
---------------------------------
Run with:
    cd streamlit_app
    streamlit run app.py
or from the project root:
    streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))                        # for network_builder / plotting
sys.path.insert(0, str(_ROOT.parent / "src"))         # for kadmos_th

from network_builder import (
    default_params, build_and_solve,
    extract_states_df, compute_performance,
)
from plotting import (
    TOPOLOGY_DOT,
    plot_ph_diagram, plot_ts_diagram,
    plot_loop_profile, plot_void_quality,
    plot_mass_flows, plot_energy_flows,
    plot_energy_pie, plot_convergence,
)

# ── page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="kadmos-TH · Reactor TH Solver",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session-state initialisation ──────────────────────────────────────────────
if "params"  not in st.session_state:
    st.session_state.params  = default_params()
if "results" not in st.session_state:
    st.session_state.results = None
if "log"     not in st.session_state:
    st.session_state.log     = ""
if "iters"   not in st.session_state:
    st.session_state.iters   = []
if "resids"  not in st.session_state:
    st.session_state.resids  = []

p = st.session_state.params   # mutable shorthand

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR – operating point + solver controls
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚛️ kadmos-TH")
    st.caption("Steady-state BWR-like loop · IAPWS-97")
    st.divider()

    st.subheader("🎛️ Operating Point")
    p["p_reactor"] = st.number_input(
        "Reactor pressure (MPa)", 1.0, 22.0,
        value=round(p["p_reactor"] / 1e6, 3), step=0.1, format="%.2f",
    ) * 1e6
    p["p_cond"] = st.number_input(
        "Condenser pressure (kPa)", 10.0, 500.0,
        value=round(p["p_cond"] / 1e3, 2), step=5.0, format="%.1f",
    ) * 1e3
    p["Q_core"] = st.number_input(
        "Core thermal power (MW)", 10.0, 5000.0,
        value=round(p["Q_core"] / 1e6, 1), step=10.0, format="%.1f",
    ) * 1e6
    p["m_core"] = st.number_input(
        "Core inlet flow (kg/s)", 50.0, 10000.0,
        value=float(p["m_core"]), step=50.0, format="%.0f",
    )

    st.divider()
    st.subheader("⚙️ Solver")
    max_iter = st.slider("Max iterations", 10, 150, 50)
    tol = st.select_slider(
        "Convergence tolerance",
        options=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        value=1e-7,
        format_func=lambda x: f"{x:.0e}",
    )

    st.divider()
    run_clicked = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

    if st.session_state.results is not None:
        res = st.session_state.results["solve_result"]
        if res.converged:
            st.success(f"✅ Converged  ({res.iterations} iters, |F|={res.residual_norm:.2e})")
        else:
            st.warning(f"⚠️ {res.message}")

# ── run simulation ────────────────────────────────────────────────────────────
if run_clicked:
    with st.spinner("Solving network …"):
        try:
            nw, result, refs, log = build_and_solve(p, max_iter=max_iter, tol=tol)
            props     = nw.props
            states_df = extract_states_df(refs, props)
            perf      = compute_performance(refs, p)

            # parse convergence history from captured stdout
            iters, resids = [], []
            for line in log.splitlines():
                m = re.match(r"\[systems-th\] iter\s+(\d+):\s+\|F\|=([\d.eE+\-]+)", line)
                if m:
                    iters.append(int(m.group(1)))
                    resids.append(float(m.group(2)))

            st.session_state.results = dict(
                solve_result=result,
                refs=refs, nw=nw, props=props,
                states_df=states_df, perf=perf,
            )
            st.session_state.log    = log
            st.session_state.iters  = iters
            st.session_state.resids = resids
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            st.session_state.results = None
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_cfg, tab_dash, tab_plots, tab_log = st.tabs([
    "🏗️ Configuration",
    "📊 Dashboard",
    "📈 Plots",
    "📋 Solver Log",
])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 – Configuration
# ─────────────────────────────────────────────────────────────────────────────
with tab_cfg:
    st.header("Network Configuration")

    with st.expander("🗺️ Loop Topology", expanded=True):
        st.graphviz_chart(TOPOLOGY_DOT, use_container_width=True)

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        # ── Core channel ──────────────────────────────────────────────────────
        with st.expander("🌡️ Core Channel", expanded=True):
            c1, c2 = st.columns(2)
            p["core_L"]  = c1.number_input("Length L (m)",        value=float(p["core_L"]),  step=0.5,  key="cL")
            p["core_dz"] = c2.number_input("Height dz (m)",       value=float(p["core_dz"]), step=0.5,  key="cdz")
            p["core_D"]  = c1.number_input("Hydr. diameter (m)",  value=float(p["core_D"]),  step=0.005, format="%.4f", key="cD")
            p["core_A"]  = c2.number_input("Flow area (m²)",      value=float(p["core_A"]),  step=0.01,  key="cA")
            p["core_K"]  = c1.number_input("K form loss",         value=float(p["core_K"]),  step=0.5,  key="cK")
            p["core_K_bundle"] = c2.number_input("K bundle",      value=float(p["core_K_bundle"]), step=1.0, key="cKb")
            p["core_K_grid"]   = c1.number_input("K spacer grid", value=float(p["core_K_grid"]),   step=0.5, key="cKg")
            p["core_n_grids"]  = int(c2.number_input("# grids",   value=int(p["core_n_grids"]),    step=1,   key="cNg"))
            p["core_two_phase"] = st.selectbox(
                "Two-phase friction model",
                ["homogeneous", "chisholm"],
                index=0 if p["core_two_phase"] == "homogeneous" else 1,
            )

        # ── Orifice plate ─────────────────────────────────────────────────────
        with st.expander("🕳️ Orifice / Mixing Plate"):
            import math
            c1, c2 = st.columns(2)
            p["n_holes"]      = int(c1.number_input("# holes", value=int(p["n_holes"]), step=5, key="nh"))
            p["D_hole"]       = c2.number_input("Hole ⌀ (mm)", value=float(p["D_hole"]*1e3), step=1.0, key="Dh") * 1e-3
            p["Cd_plate"]     = st.slider("Discharge coeff. Cd", 0.50, 0.85, float(p["Cd_plate"]), step=0.01)
            p["post_core_ft"] = st.number_input("Core-to-plate distance (ft)", value=float(p["post_core_ft"]), step=0.5, key="pc_ft")
            A_open = p["n_holes"] * math.pi * p["D_hole"]**2 / 4.0
            st.info(f"Open area: **{A_open*1e4:.2f} cm²**  |  "
                    f"Constriction ratio A_holes/A_core: **{A_open/p['core_A']:.3f}**")

        # ── Separators ────────────────────────────────────────────────────────
        with st.expander("🌀 Separators"):
            st.markdown("**OrificePhase** (primary phase splitter)")
            c1, c2 = st.columns(2)
            p["orif_x_vap"] = c1.slider("Vapor outlet quality",  0.980, 1.000, float(p["orif_x_vap"]), step=0.001, key="oxv")
            p["orif_x_liq"] = c2.slider("Liquid outlet quality", 0.000, 0.020, float(p["orif_x_liq"]), step=0.001, key="oxl")

            st.markdown("**SteamSep** (chimney exit / turbine feed)")
            c1, c2 = st.columns(2)
            p["ssep_dp"]    = c1.number_input("Δp (kPa)", value=float(p["ssep_dp"]/1e3), step=10.0, key="sdp") * 1e3
            p["ssep_x_vap"] = c2.slider("Vapor outlet quality",  0.980, 1.000, float(p["ssep_x_vap"]), step=0.001, key="sxv")
            p["ssep_x_liq"] = c1.slider("Liquid outlet quality", 0.000, 0.020, float(p["ssep_x_liq"]), step=0.001, key="sxl")

    with col_right:
        # ── Downcomer / venturi ───────────────────────────────────────────────
        with st.expander("⬇️ Downcomer & Venturi", expanded=True):
            p["dc_D"] = st.number_input("Diameter (m)", value=float(p["dc_D"]), step=0.05, key="dcD")
            c1, c2 = st.columns(2)
            p["dc_L_upper"] = c1.number_input("Upper length (m)",  value=float(p["dc_L_upper"]), step=0.5, key="dcLu")
            p["dc_K_upper"] = c2.number_input("Upper K",           value=float(p["dc_K_upper"]), step=0.1, key="dcKu")
            p["dc_L_lower"] = c1.number_input("Lower length (m)",  value=float(p["dc_L_lower"]), step=0.5, key="dcLl")
            p["dc_K_lower"] = c2.number_input("Lower K",           value=float(p["dc_K_lower"]), step=0.1, key="dcKl")
            st.info(f"Total downcomer: **{p['dc_L_upper']+p['dc_L_lower']:.1f} m**")

        # ── Chimney ───────────────────────────────────────────────────────────
        with st.expander("🏭 Chimney"):
            c1, c2 = st.columns(2)
            p["chim_L"] = c1.number_input("Length (m)",   value=float(p["chim_L"]), step=0.5, key="chL")
            p["chim_D"] = c2.number_input("Diameter (m)", value=float(p["chim_D"]), step=0.05, key="chD")
            p["chim_K"] = st.number_input("K form loss",  value=float(p["chim_K"]), step=0.1,  key="chK")

        # ── Steam cycle ───────────────────────────────────────────────────────
        with st.expander("⚡ Steam Cycle"):
            p["turb_eta"]  = st.slider("Turbine isentropic η", 0.60, 1.00, float(p["turb_eta"]), step=0.01)
            p["pump_eta"]  = st.slider("Pump efficiency η",     0.50, 1.00, float(p["pump_eta"]), step=0.01)
            p["heater_T_K"] = st.number_input(
                "Feedwater heater outlet T (°C)",
                value=float(p["heater_T_K"] - 273.15), step=1.0, key="fwT",
            ) + 273.15
            p["eta_gen"] = st.slider("Generator efficiency η", 0.90, 1.00, float(p["eta_gen"]), step=0.005)

    st.caption(
        "ℹ️ Geometry changes take effect on the next **Run Simulation**. "
        "Operating-point sliders in the sidebar update immediately."
    )

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 – Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tab_dash:
    if st.session_state.results is None:
        st.info("▶️ Press **Run Simulation** in the sidebar to compute results.")
    else:
        r       = st.session_state.results
        perf    = r["perf"]
        states  = r["states_df"]
        res     = r["solve_result"]

        st.header("Performance Dashboard")
        conv_badge = "✅ Converged" if res.converged else "⚠️ Not converged"
        st.caption(f"{conv_badge}  ·  {res.iterations} iterations  ·  |F| = {res.residual_norm:.2e}")
        st.divider()

        # ── KPI cards ─────────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("⚡ Net Electric Power",  f"{perf['W_electric']/1e6:.2f} MWe")
        k2.metric("📊 Net Efficiency",      f"{perf['eta_net']*100:.2f} %")
        k3.metric("🔄 Recirc. Ratio",       f"{perf['recirc_ratio']:.2f}")
        k4.metric("💨 Steam → Turbine",     f"{perf['m_steam_turbine']:.1f} kg/s")
        k5.metric("🕳️ Orifice Δp",          f"{perf['dp_orifice_kPa']:.0f} kPa")

        st.divider()

        # ── Energy + flow tables ───────────────────────────────────────────────
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.subheader("⚡ Energy Balance")
            MW = 1e-6
            W_gen_loss = perf["W_net_mech"] - perf["W_electric"]
            edf = pd.DataFrame({
                "Quantity": [
                    "Core thermal power *", "Turbine gross output",
                    "Pump parasitic", "Net shaft power",
                    "Net electric output", "Heat rejected (condenser)",
                    "Generator losses",
                ],
                "Value [MW]": [
                    (perf["W_turb_gross"] + perf["Q_rejected"]) * MW,
                    perf["W_turb_gross"] * MW,
                    perf["W_pump_shaft"] * MW,
                    perf["W_net_mech"] * MW,
                    perf["W_electric"] * MW,
                    perf["Q_rejected"] * MW,
                    W_gen_loss * MW,
                ],
            })
            st.dataframe(edf.style.format({"Value [MW]": "{:.3f}"}), use_container_width=True, hide_index=True)
            st.caption("* Approximated as W_turbine + Q_condenser")

        with col_b:
            st.subheader("🌊 Mass Flow Distribution")
            mdf = pd.DataFrame({
                "Stream": [
                    "Core inlet (total)", "Vapor to chimney (orifice)",
                    "Liquid return → Venturi", "Steam to turbine",
                    "SteamSep liquid return",
                    "Upper downcomer (net cycle flow)",
                ],
                "Flow [kg/s]": [
                    perf["m_core"],        perf["m_vap_orifice"],
                    perf["m_liq_return"],  perf["m_steam_turbine"],
                    perf["m_ssep_liq"],    perf["m_dc_upper"],
                ],
            })
            st.dataframe(mdf.style.format({"Flow [kg/s]": "{:.2f}"}), use_container_width=True, hide_index=True)

        st.divider()

        # ── Energy pie ────────────────────────────────────────────────────────
        col_pie, col_tbl = st.columns([1, 2], gap="large")
        with col_pie:
            st.plotly_chart(plot_energy_pie(perf), use_container_width=True)

        with col_tbl:
            st.subheader("Connection State Table")
            show_cols = ["Station", "Branch", "m_kgs", "p_MPa", "T_C", "x", "alpha", "h_kJkg"]
            fmt = {
                "m_kgs":   "{:.2f}", "p_MPa":  "{:.4f}",
                "T_C":     "{:.2f}", "x":      "{:.4f}",
                "alpha":   "{:.4f}", "h_kJkg": "{:.1f}",
            }
            styled = (
                states[show_cols]
                .style
                .format(fmt)
                .background_gradient(subset=["T_C"], cmap="RdYlBu_r")
                .background_gradient(subset=["alpha"], cmap="Blues")
            )
            st.dataframe(styled, use_container_width=True, height=430)

        st.divider()
        csv = states.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download full state table (CSV)",
            data=csv, file_name="kadmos_th_states.csv", mime="text/csv",
        )

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 – Plots
# ─────────────────────────────────────────────────────────────────────────────
with tab_plots:
    if st.session_state.results is None:
        st.info("▶️ Press **Run Simulation** in the sidebar to generate plots.")
    else:
        r      = st.session_state.results
        states = r["states_df"]
        perf   = r["perf"]

        PLOT_OPTIONS = {
            "🌡️ Loop Profile (P & T)":             "loop",
            "💧 Quality & Void Fraction":           "qvoid",
            "📐 Pressure–Enthalpy (P-h) Diagram":  "ph",
            "📐 Temperature–Entropy (T-s) Diagram":"ts",
            "🌊 Mass Flow Distribution":            "mass",
            "⚡ Energy Flow Balance":               "energy",
        }

        # Allow multiple simultaneous plots with checkboxes
        st.subheader("Select plots to display")
        checks = {}
        cols_ck = st.columns(3)
        for i, (label, key) in enumerate(PLOT_OPTIONS.items()):
            checks[key] = cols_ck[i % 3].checkbox(label, value=(key in ("loop", "ph")))

        st.divider()

        if checks["loop"]:
            st.plotly_chart(plot_loop_profile(states), use_container_width=True)

        if checks["qvoid"]:
            st.plotly_chart(plot_void_quality(states), use_container_width=True)

        if checks["ph"]:
            st.plotly_chart(plot_ph_diagram(states), use_container_width=True)

        if checks["ts"]:
            st.plotly_chart(plot_ts_diagram(states), use_container_width=True)

        if checks["mass"]:
            st.plotly_chart(plot_mass_flows(perf), use_container_width=True)

        if checks["energy"]:
            st.plotly_chart(plot_energy_flows(perf), use_container_width=True)

        if not any(checks.values()):
            st.info("Check at least one plot above.")

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 – Solver Log
# ─────────────────────────────────────────────────────────────────────────────
with tab_log:
    if not st.session_state.log:
        st.info("No solver log yet. Run a simulation first.")
    else:
        iters  = st.session_state.iters
        resids = st.session_state.resids

        if iters:
            st.plotly_chart(plot_convergence(iters, resids), use_container_width=True)
        else:
            st.warning("Could not parse convergence history from log.")

        st.divider()
        with st.expander("📄 Raw solver output", expanded=False):
            st.text(st.session_state.log)
