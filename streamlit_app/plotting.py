"""
plotting.py
-----------
Plotly chart builders for the kadmos-TH Streamlit GUI.
All functions accept post-solve data and return go.Figure objects.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── colour palette ────────────────────────────────────────────────────────────
BRANCH_COLOR = {
    "Primary loop":  "#1f77b4",
    "Steam cycle":   "#d62728",
    "Liquid return": "#2ca02c",
}
_TEAL   = "#17becf"
_ORANGE = "#ff7f0e"
_GRAY   = "rgba(150,150,150,0.5)"


# ── saturation dome (cached for the process lifetime) ────────────────────────
@lru_cache(maxsize=1)
def _saturation_dome() -> dict | None:
    """
    Return liquid / vapour saturation arrays for P-h and T-s diagrams.
    Returns None if iapws is not installed.
    Values are in *plot units*: P [MPa], h [kJ/kg], T [°C], s [kJ/kg·K].
    """
    try:
        from iapws import IAPWS97  # type: ignore
    except ImportError:
        return None

    # Sample 80 pressures from near triple point to critical point
    p_mpa = np.concatenate([
        np.linspace(0.001, 1.0, 30),
        np.linspace(1.0, 22.0, 50),
    ])
    h_f, h_g, T_sat, s_f, s_g = [], [], [], [], []
    for p in p_mpa:
        try:
            wl = IAPWS97(P=p, x=0.0)
            wv = IAPWS97(P=p, x=1.0)
            h_f.append(wl.h);  h_g.append(wv.h)
            T_sat.append(wl.T - 273.15)
            s_f.append(wl.s);  s_g.append(wv.s)
        except Exception:
            h_f.append(np.nan); h_g.append(np.nan)
            T_sat.append(np.nan)
            s_f.append(np.nan); s_g.append(np.nan)

    return dict(
        p_MPa=p_mpa,
        h_f=np.array(h_f), h_g=np.array(h_g),
        T_C=np.array(T_sat),
        s_f=np.array(s_f),  s_g=np.array(s_g),
    )


def _dome_trace_ph(sat: dict) -> go.Scatter:
    h = np.concatenate([sat["h_f"], sat["h_g"][::-1], [sat["h_f"][0]]])
    p = np.concatenate([sat["p_MPa"], sat["p_MPa"][::-1], [sat["p_MPa"][0]]])
    return go.Scatter(
        x=h, y=p, mode="lines",
        line=dict(color="gray", width=1, dash="dot"),
        name="Sat. dome", showlegend=True,
        hoverinfo="skip",
    )


def _dome_trace_ts(sat: dict) -> go.Scatter:
    s = np.concatenate([sat["s_f"], sat["s_g"][::-1], [sat["s_f"][0]]])
    T = np.concatenate([sat["T_C"], sat["T_C"][::-1], [sat["T_C"][0]]])
    return go.Scatter(
        x=s, y=T, mode="lines",
        line=dict(color="gray", width=1, dash="dot"),
        name="Sat. dome", showlegend=True,
        hoverinfo="skip",
    )


# ── P-h diagram ───────────────────────────────────────────────────────────────
def plot_ph_diagram(states_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    sat = _saturation_dome()
    if sat is not None:
        fig.add_trace(_dome_trace_ph(sat))

    for branch, grp in states_df.groupby("Branch"):
        cd = np.stack([grp["T_C"], grp["x"], grp["m_kgs"], grp["alpha"]], axis=-1)
        fig.add_trace(go.Scatter(
            x=grp["h_kJkg"], y=grp["p_MPa"],
            mode="markers+text",
            marker=dict(size=11, color=BRANCH_COLOR.get(branch, "black"),
                        line=dict(width=1, color="white")),
            text=grp["Station"], textposition="top center",
            textfont=dict(size=8),
            name=branch,
            customdata=cd,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "h = %{x:.1f} kJ/kg<br>"
                "p = %{y:.4f} MPa<br>"
                "T = %{customdata[0]:.1f} °C<br>"
                "x = %{customdata[1]:.4f}<br>"
                "α = %{customdata[3]:.4f}<br>"
                "ṁ = %{customdata[2]:.2f} kg/s"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Pressure–Enthalpy (P-h) Diagram",
        xaxis_title="Specific enthalpy  h  [kJ/kg]",
        yaxis_title="Pressure  P  [MPa]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=520,
        template="plotly_white",
    )
    return fig


# ── T-s diagram ───────────────────────────────────────────────────────────────
def plot_ts_diagram(states_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    sat = _saturation_dome()
    if sat is not None:
        fig.add_trace(_dome_trace_ts(sat))

    for branch, grp in states_df.groupby("Branch"):
        grp = grp.dropna(subset=["s_kJkgK"])
        cd = np.stack([grp["p_MPa"], grp["x"], grp["m_kgs"]], axis=-1)
        fig.add_trace(go.Scatter(
            x=grp["s_kJkgK"], y=grp["T_C"],
            mode="markers+text",
            marker=dict(size=11, color=BRANCH_COLOR.get(branch, "black"),
                        line=dict(width=1, color="white")),
            text=grp["Station"], textposition="top center",
            textfont=dict(size=8),
            name=branch,
            customdata=cd,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "s = %{x:.3f} kJ/kg·K<br>"
                "T = %{y:.1f} °C<br>"
                "p = %{customdata[0]:.4f} MPa<br>"
                "x = %{customdata[1]:.4f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Temperature–Entropy (T-s) Diagram",
        xaxis_title="Specific entropy  s  [kJ/kg·K]",
        yaxis_title="Temperature  T  [°C]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=520,
        template="plotly_white",
    )
    return fig


# ── Loop P & T profile ────────────────────────────────────────────────────────
_LOOP_ORDER = [
    "Core inlet", "Core outlet", "Post-core", "Orifice plate out",
    "Chimney inlet", "Chimney outlet",
    "Turbine inlet", "Turbine outlet",
    "Condenser outlet", "Pump outlet", "Heater outlet", "Mixer → DC upper",
]

def plot_loop_profile(states_df: pd.DataFrame) -> go.Figure:
    order_map = {s: i for i, s in enumerate(_LOOP_ORDER)}
    df = (states_df[states_df["Station"].isin(_LOOP_ORDER)]
          .assign(_ord=lambda d: d["Station"].map(order_map))
          .sort_values("_ord"))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df["Station"], y=df["p_MPa"],
        mode="lines+markers", name="Pressure [MPa]",
        line=dict(color="#1f77b4", width=2.5),
        marker=dict(size=8, symbol="circle"),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["Station"], y=df["T_C"],
        mode="lines+markers", name="Temperature [°C]",
        line=dict(color="#d62728", width=2.5, dash="dash"),
        marker=dict(size=8, symbol="diamond"),
    ), secondary_y=True)

    fig.update_xaxes(tickangle=40)
    fig.update_yaxes(title_text="Pressure [MPa]", color="#1f77b4", secondary_y=False)
    fig.update_yaxes(title_text="Temperature [°C]", color="#d62728", secondary_y=True)
    fig.update_layout(
        title="Loop Pressure & Temperature Profile",
        height=460, template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )
    return fig


# ── Quality & void fraction profile ──────────────────────────────────────────
_PRIMARY_ORDER = [
    "Core inlet", "Core outlet", "Post-core", "Orifice plate out",
    "Chimney inlet", "Chimney outlet",
]

def plot_void_quality(states_df: pd.DataFrame) -> go.Figure:
    order_map = {s: i for i, s in enumerate(_PRIMARY_ORDER)}
    df = (states_df[states_df["Station"].isin(_PRIMARY_ORDER)]
          .assign(_ord=lambda d: d["Station"].map(order_map))
          .sort_values("_ord"))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Station"], y=df["x"],
        name="Steam quality  x", marker_color=_ORANGE, opacity=0.65,
    ))
    fig.add_trace(go.Scatter(
        x=df["Station"], y=df["alpha"],
        mode="lines+markers", name="Void fraction  α",
        line=dict(color="#2ca02c", width=2.5),
        marker=dict(size=9),
    ))
    fig.update_layout(
        title="Steam Quality & Void Fraction — Primary Loop",
        yaxis_title="[-]", xaxis_tickangle=30,
        height=420, template="plotly_white",
        legend=dict(x=0.01, y=0.99),
        barmode="overlay",
    )
    return fig


# ── Mass flow bar chart ───────────────────────────────────────────────────────
def plot_mass_flows(perf: dict) -> go.Figure:
    labels = [
        "Core inlet", "Vapor to chimney",
        "Liquid return\n(venturi)", "Steam to turbine",
        "SteamSep liquid\nreturn",
    ]
    values = [
        perf["m_core"],
        perf["m_vap_orifice"],
        perf["m_liq_return"],
        perf["m_steam_turbine"],
        perf["m_ssep_liq"],
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Mass Flow Distribution",
        yaxis_title="Mass flow [kg/s]",
        height=420, template="plotly_white",
        showlegend=False,
    )
    return fig


# ── Energy flow bar chart ─────────────────────────────────────────────────────
def plot_energy_flows(perf: dict) -> go.Figure:
    MW = 1e-6
    labels = [
        "Core thermal\npower", "Turbine gross",
        "Pump parasitic", "Net shaft",
        "Net electric\noutput", "Heat rejected\n(condenser)",
    ]
    values = [
        perf["W_turb_gross"] * MW + perf["Q_rejected"] * MW,  # ≈ Q_core
        perf["W_turb_gross"] * MW,
        perf["W_pump_shaft"] * MW,
        perf["W_net_mech"] * MW,
        perf["W_electric"] * MW,
        perf["Q_rejected"] * MW,
    ]
    colors = ["#e377c2", "#17becf", "#d62728", "#17becf", "#2ca02c", "#8c564b"]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Energy Balance [MWth / MWe]",
        yaxis_title="Power [MW]",
        height=420, template="plotly_white",
        showlegend=False,
    )
    return fig


# ── Energy pie chart (for dashboard) ─────────────────────────────────────────
def plot_energy_pie(perf: dict) -> go.Figure:
    MW = 1e-6
    W_gen_loss = perf["W_net_mech"] - perf["W_electric"]
    labels = ["Net electric", "Heat rejected", "Pump parasitic", "Generator loss"]
    values = [
        perf["W_electric"] * MW,
        perf["Q_rejected"] * MW,
        perf["W_pump_shaft"] * MW,
        W_gen_loss * MW,
    ]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.45,
        marker_colors=["#2ca02c", "#8c564b", "#d62728", "#c7c7c7"],
        textinfo="label+percent",
    ))
    fig.update_layout(
        title=f"Thermal Power Allocation  (η = {perf['eta_net']*100:.1f} %)",
        height=380, template="plotly_white",
    )
    return fig


# ── Convergence history ────────────────────────────────────────────────────────
def plot_convergence(iterations: list[int], residuals: list[float]) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=iterations, y=residuals,
        mode="lines+markers",
        line=dict(color="#1f77b4", width=2.5),
        marker=dict(size=7),
        name="|F| scaled",
    ))
    fig.update_layout(
        title="Newton Solver Convergence",
        xaxis_title="Iteration",
        yaxis_title="|F|  (scaled residual norm)",
        yaxis_type="log",
        height=360, template="plotly_white",
    )
    return fig


# ── Network topology (graphviz dot string) ────────────────────────────────────
TOPOLOGY_DOT = """\
digraph {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fontsize=10, fontname=Helvetica];
    edge [fontsize=9];

    SteamMixer   [label="Steam\\nMixer",      fillcolor="#aec7e8"]
    DC_up        [label="Downcomer\\nupper",  fillcolor="#aec7e8"]
    Venturi      [label="Venturi\\n(jet pump)",fillcolor="#98df8a", shape=ellipse]
    DC_low       [label="Downcomer\\nlower",  fillcolor="#aec7e8"]
    Core         [label="Core\\nChannel",     fillcolor="#ff9896", shape=box3d]
    PostCore     [label="Post-core\\n(3 ft)", fillcolor="#c7c7c7"]
    Orifice      [label="Orifice\\nPlate",    fillcolor="#c5b0d5", shape=diamond]
    OrifPhase    [label="Orifice\\nSeparator",fillcolor="#c5b0d5"]
    Chimney      [label="Chimney",            fillcolor="#aec7e8"]
    SteamSep     [label="Steam\\nSeparator",  fillcolor="#c5b0d5"]
    Turbine      [label="Turbine",            fillcolor="#ffbb78", shape=trapezium]
    Condenser    [label="Condenser",          fillcolor="#dbdb8d"]
    Pump         [label="Pump",               fillcolor="#9edae5", shape=invtrapezium]
    Heater       [label="Feedwater\\nHeater", fillcolor="#f7b6d2"]

    SteamMixer -> DC_up -> Venturi -> DC_low -> Core
    Core -> PostCore -> Orifice -> OrifPhase
    OrifPhase -> Chimney        [label="vapor ↑"                      color="#d62728"]
    OrifPhase -> Venturi        [label="liquid return" style=dashed   color="#2ca02c"]
    Chimney   -> SteamSep
    SteamSep  -> Turbine        [label="steam"                        color="#d62728"]
    SteamSep  -> SteamMixer     [label="sep. liquid"  style=dashed    color="#2ca02c"]
    Turbine   -> Condenser -> Pump -> Heater -> SteamMixer
}
"""
