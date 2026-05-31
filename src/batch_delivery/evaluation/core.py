"""KPI computation, scenario comparison tables, and result export."""


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from batch_delivery.config.constants import (
    COST_SCALE,
    N_DAYS,
    SC_BASELINE,
    SC_FIXED_BATCH,
    SC_FIXED_EXPRESS,
    SC_SA_BATCH,
    SC_SA_EXPRESS,
    WEEKDAYS,
)
from batch_delivery.utils import log

# ─────────────────────────────────────────────────────────────────────────────
# Per-scenario KPI aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_scenario_kpis(
    df_routes: pd.DataFrame,
    scenario_name: str,
    avg_waiting_days: float = 0.0,
) -> dict:
    """Compute aggregate KPIs for one scenario from parsed routes.

    Parameters
    ----------
    df_routes : DataFrame
        Parsed route DataFrame (output of ``routing.parse_routes``).
    scenario_name : str
    avg_waiting_days : float
        Pre-computed average customer waiting time (days).

    Returns
    -------
    dict of KPI name → value.
    """
    if len(df_routes) == 0:
        return {}

    n_routes = len(df_routes)
    parcels = int(df_routes["parcels"].sum())
    km = df_routes["distance_km"].sum()
    cost = df_routes["cost"].sum() / COST_SCALE

    return {
        "scenario": scenario_name,
        "routes": n_routes,
        "parcels": parcels,
        "distance_km": round(km, 0),
        "cost_eur": round(cost, 0),
        "eur_per_parcel": round(cost / max(1, parcels), 2),
        "eur_per_km": round(cost / max(1, km), 2),
        "avg_load_factor": round(df_routes["load_factor"].mean() * 100, 1),
        "avg_parcels_per_route": round(parcels / max(1, n_routes), 1),
        "avg_stops_per_route": round(df_routes["n_stops"].mean(), 1),
        "avg_travel_h": round(df_routes["travel_h"].mean(), 2),
        "avg_service_h": round(df_routes["service_h"].mean(), 2),
        "avg_waiting_h": round(df_routes["waiting_h"].mean(), 2),
        "customer_wait_days": round(avg_waiting_days, 2),
        "co2_proxy_kg": round(km * 0.15, 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-scenario comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_scenarios(
    kpi_list: list[dict],
    baseline_name: str = "Baseline",
) -> pd.DataFrame:
    """Build a comparison DataFrame with absolute and relative changes.

    Parameters
    ----------
    kpi_list : list of dicts
        Each from ``compute_scenario_kpis``.
    baseline_name : str
        Name of the baseline scenario (used for Δ% columns).

    Returns
    -------
    DataFrame with one row per scenario.
    """
    df = pd.DataFrame(kpi_list)
    if len(df) == 0:
        return df
    df = df.set_index("scenario")

    # Add Δ% columns relative to baseline
    base = df.loc[baseline_name] if baseline_name in df.index else None
    if base is not None:
        for col in ["cost_eur", "routes", "distance_km", "eur_per_parcel"]:
            if col in df.columns and base[col] != 0:
                df[f"delta_{col}_pct"] = round(
                    (df[col] / base[col] - 1) * 100, 1
                )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Daily fleet profile
# ─────────────────────────────────────────────────────────────────────────────

def daily_fleet_profile(df_routes: pd.DataFrame) -> pd.DataFrame:
    """Count routes (≈ vehicles) per day for a scenario.

    Returns DataFrame: day, day_name, n_routes, parcels, distance_km.
    """
    if len(df_routes) == 0:
        return pd.DataFrame(columns=["day_idx", "day", "n_routes", "parcels", "distance_km"])

    grouped = df_routes.groupby(["day_idx", "day"]).agg(
        n_routes=("vehicle_id", "count"),
        parcels=("parcels", "sum"),
        distance_km=("distance_km", "sum"),
    ).reset_index()
    return grouped.sort_values("day_idx")


# ─────────────────────────────────────────────────────────────────────────────
# Per-hub KPI summary
# ─────────────────────────────────────────────────────────────────────────────

def per_hub_kpis(df_routes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate KPIs per hub.

    Returns DataFrame with columns: hub, n_routes, parcels, cost_eur,
    distance_km, avg_load_factor, avg_stops.
    """
    if len(df_routes) == 0:
        return pd.DataFrame()

    g = df_routes.groupby("hub").agg(
        n_routes=("vehicle_id", "count"),
        parcels=("parcels", "sum"),
        cost_cents=("cost", "sum"),
        distance_km=("distance_km", "sum"),
        avg_load_factor=("load_factor", "mean"),
        avg_stops=("n_stops", "mean"),
    ).reset_index()
    g["cost_eur"] = (g["cost_cents"] / COST_SCALE).round(0)
    g["avg_load_factor"] = (g["avg_load_factor"] * 100).round(1)
    g["avg_stops"] = g["avg_stops"].round(1)
    return g.drop(columns=["cost_cents"])


# ─────────────────────────────────────────────────────────────────────────────
# HTML table rendering
# ─────────────────────────────────────────────────────────────────────────────

_SCENARIO_COLORS = {
    SC_BASELINE: "#2196F3",
    SC_FIXED_EXPRESS: "#FF9800",
    SC_SA_EXPRESS: "#4CAF50",
    SC_FIXED_BATCH: "#9E9E9E",
    SC_SA_BATCH: "#795548",
}


def kpi_table_html(df_kpi: pd.DataFrame) -> str:
    """Render scenario comparison DataFrame as styled HTML table."""
    color_map = _SCENARIO_COLORS

    header = "<tr><th>Scenario</th>"
    for col in df_kpi.columns:
        header += f"<th>{col}</th>"
    header += "</tr>"

    rows = []
    for sc in df_kpi.index:
        c = color_map.get(sc, "#333")
        row = f'<tr><td style="color:{c};font-weight:bold">{sc}</td>'
        for col in df_kpi.columns:
            val = df_kpi.loc[sc, col]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            if isinstance(val, float):
                if "pct" in col:
                    sign = "+" if val > 0 else ""
                    row += f"<td>{sign}{val:.1f}%</td>"
                else:
                    row += f"<td>{val:,.1f}</td>"
            else:
                row += f"<td>{val:,}</td>"
        row += "</tr>"
        rows.append(row)

    return (
        '<table style="border-collapse:collapse;font-family:sans-serif;font-size:13px">'
        f"<thead>{header}</thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def save_kpis(
    df_kpi: pd.DataFrame,
    output_path: str | None = None,
) -> None:
    """Save scenario KPI comparison to CSV."""
    from batch_delivery.config.constants import RESULTS_DIR

    if output_path is None:
        output_path = RESULTS_DIR / "scenario_comparison_kpis.csv"
    df_kpi.to_csv(output_path)
    log.debug(f"KPIs saved to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Schedule overview — interactive Plotly visualisation
# ─────────────────────────────────────────────────────────────────────────────

_SCENARIO_EMOJI = {
    SC_BASELINE: "📦",
    SC_FIXED_EXPRESS: "📌",
    SC_SA_EXPRESS: "🧠",
    SC_FIXED_BATCH: "📅",
    SC_SA_BATCH: "🚀",
}


def plot_schedule_overview(
    scenario_schedules_by_provider: dict,
    providers: list[str],
    scenario_names: list[str],
    weekdays: list[str] | None = None,
) -> go.Figure:
    """Build an animated Plotly figure showing delivery schedules per scenario.

    Parameters
    ----------
    scenario_schedules_by_provider
        ``{provider: {scenario: {plz: frozenset(day_indices)}}}``
    providers
        Ordered list of LSP names.
    scenario_names
        Ordered list of scenario names (used as animation frames).
    weekdays
        Day labels; defaults to ``WEEKDAYS[:N_DAYS]``.

    Returns
    -------
    go.Figure — animated heatmap + frequency chart with play controls.
    """
    if weekdays is None:
        weekdays = WEEKDAYS[:N_DAYS]
    n_days = len(weekdays)

    # ── pre-compute frame data ──────────────────────────────────────────
    frame_data: list[dict] = []
    for sc in scenario_names:
        # heatmap: % of PLZ active per provider × day
        heatmap = np.zeros((len(providers), n_days))
        # frequency: how many PLZ have freq k (summed across providers)
        freq_counter: dict[int, int] = {}
        total_deliveries = 0
        total_plz = 0

        for pi, prov in enumerate(providers):
            schedules = scenario_schedules_by_provider[prov].get(sc, {})
            n_plz = max(len(schedules), 1)
            for di in range(n_days):
                active = sum(1 for s in schedules.values() if di in s)
                heatmap[pi, di] = round(active / n_plz * 100, 1)

            for s in schedules.values():
                f = len(s)
                freq_counter[f] = freq_counter.get(f, 0) + 1
                total_deliveries += f
                total_plz += 1

        avg_freq = total_deliveries / max(total_plz, 1)

        # frequency bins 1..6
        freq_x = list(range(1, n_days + 1))
        freq_y = [freq_counter.get(k, 0) for k in freq_x]

        frame_data.append({
            "heatmap": heatmap,
            "freq_x": freq_x,
            "freq_y": freq_y,
            "avg_freq": avg_freq,
            "total_del": total_deliveries,
            "total_plz": total_plz,
        })

    # ── build figure with subplots ──────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.08,
        subplot_titles=["Delivery Activity per Day (%)", "Frequency Distribution"],
    )

    # initial traces (first scenario)
    d0 = frame_data[0]
    day_labels = [d[:3] for d in weekdays]

    hm = go.Heatmap(
        z=d0["heatmap"],
        x=day_labels,
        y=providers,
        colorscale="Viridis",
        zmin=0, zmax=100,
        colorbar=dict(
            title=dict(text="Active %", font=dict(size=11)),
            len=0.85, thickness=12,
            ticksuffix="%",
        ),
        hovertemplate=(
            "<b>%{y}</b> — %{x}<br>"
            "Active: %{z:.0f}%<extra></extra>"
        ),
    )
    fig.add_trace(hm, row=1, col=1)

    bar = go.Bar(
        x=[f"{k}×/wk" for k in d0["freq_x"]],
        y=d0["freq_y"],
        marker_color="#5E35B1",
        marker_line=dict(width=0),
        hovertemplate="<b>%{x}</b>: %{y} PLZ<extra></extra>",
    )
    fig.add_trace(bar, row=1, col=2)

    # ── animation frames ────────────────────────────────────────────────
    frames = []
    for i, sc in enumerate(scenario_names):
        d = frame_data[i]
        emoji = _SCENARIO_EMOJI.get(sc, "")
        frames.append(go.Frame(
            data=[
                go.Heatmap(
                    z=d["heatmap"],
                    x=day_labels,
                    y=providers,
                    colorscale="Viridis",
                    zmin=0, zmax=100,
                ),
                go.Bar(
                    x=[f"{k}×/wk" for k in d["freq_x"]],
                    y=d["freq_y"],
                    marker_color="#5E35B1",
                ),
            ],
            name=sc,
            layout=go.Layout(
                title_text=(
                    f"{emoji} <b>{sc}</b> — "
                    f"avg {d['avg_freq']:.1f}×/wk, "
                    f"{d['total_del']:,} deliveries, "
                    f"{d['total_plz']:,} PLZ"
                ),
            ),
        ))
    fig.frames = frames

    # ── slider + play/pause buttons ─────────────────────────────────────
    sliders = [dict(
        active=0,
        pad=dict(b=10, t=40),
        len=0.9, x=0.05,
        currentvalue=dict(
            prefix="Scenario: ",
            font=dict(size=13),
            visible=True,
            xanchor="center",
        ),
        steps=[
            dict(
                args=[[sc], dict(
                    frame=dict(duration=500, redraw=True),
                    mode="immediate",
                    transition=dict(duration=300),
                )],
                label=sc,
                method="animate",
            )
            for sc in scenario_names
        ],
    )]

    fig.update_layout(
        sliders=sliders,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.15, x=0.05, xanchor="left",
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=800, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=400),
                    )],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
        )],
        title=dict(
            text=(
                f"{_SCENARIO_EMOJI.get(scenario_names[0], '')} "
                f"<b>{scenario_names[0]}</b> — "
                f"avg {frame_data[0]['avg_freq']:.1f}×/wk, "
                f"{frame_data[0]['total_del']:,} deliveries, "
                f"{frame_data[0]['total_plz']:,} PLZ"
            ),
            font=dict(size=16),
            x=0.5,
        ),
        template="plotly_white",
        height=480,
        margin=dict(l=80, r=30, t=100, b=80),
        yaxis=dict(autorange="reversed"),
        yaxis2=dict(title="Number of PLZ"),
    )

    fig.update_xaxes(title_text="Weekday", row=1, col=1)
    fig.update_xaxes(title_text="Delivery Frequency", row=1, col=2)

    return fig
