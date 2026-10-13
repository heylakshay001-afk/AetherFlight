"""
graphs.py — AetherFlight Plotting
==================================
Builds every Plotly figure shown in the UI from already-computed data.

Module boundary: this file does NO aerodynamic or physics math of its own
— everything it plots is handed in by the caller (app.py), which gets it
from physics.py / simulation.py. That keeps a single source of truth for
every number on screen; this file only draws it.

All inputs are SI units (m/s, N, m, s). The `unit` argument controls ONLY
the display conversion on speed axes.
"""

import numpy as np
import plotly.graph_objects as go

MS_TO_KMH = 3.6
PLOT_LINE_WIDTH = 3
MARKER_SIZE = 12


def _dark_layout(fig, title, x_title, y_title):
    """Shared dark theme + consistent legend/margin styling, applied to
    every chart in the app so they read as one coherent set."""
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, l=60, r=30, b=50),
    )
    return fig


def _speed_converter(unit):
    """Returns (convert_fn, axis_label) for the chosen display unit."""
    if unit == "km/h":
        return (lambda v: np.asarray(v) * MS_TO_KMH), "km/h"
    return (lambda v: np.asarray(v)), "m/s"


def _build_aero_chart(airspeed_display, speed_label, lift, drag, thrust_curve,
                       stall_display, max_speed_display, required_lift_n):
    """Lift / Drag / Thrust vs Airspeed — the primary flight-envelope
    chart. Marks the stall region, the liftoff lift threshold, and the
    max-speed point where available thrust meets drag."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=airspeed_display, y=lift, mode="lines", name="Lift", line=dict(width=PLOT_LINE_WIDTH),
        hovertemplate=f"Airspeed: %{{x:.1f}} {speed_label}<br>Lift: %{{y:.2f}} N<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=airspeed_display, y=drag, mode="lines", name="Drag", line=dict(width=PLOT_LINE_WIDTH),
        hovertemplate=f"Airspeed: %{{x:.1f}} {speed_label}<br>Drag: %{{y:.2f}} N<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=airspeed_display, y=thrust_curve, mode="lines", name="Thrust", line=dict(width=PLOT_LINE_WIDTH),
        hovertemplate=f"Airspeed: %{{x:.1f}} {speed_label}<br>Thrust: %{{y:.2f}} N<extra></extra>"
    ))

    thrust_at_max = np.interp(max_speed_display, airspeed_display, thrust_curve)
    fig.add_trace(go.Scatter(
        x=[max_speed_display], y=[thrust_at_max], mode="markers", name="Max Speed",
        marker=dict(size=MARKER_SIZE, color="yellow", symbol="star"),
        hovertemplate=f"Max Speed: %{{x:.1f}} {speed_label}<extra></extra>"
    ))

    fig.add_hline(
        y=required_lift_n, line_dash="dot", line_color="lime",
        annotation_text=f"Liftoff threshold ({required_lift_n:.1f} N)",
        annotation_position="bottom right"
    )
    fig.add_vline(x=stall_display, line_dash="dash", line_color="red", annotation_text="Stall Speed")
    fig.add_vrect(
        x0=0, x1=stall_display, fillcolor="red", opacity=0.1, line_width=0,
        annotation_text="Stall Region", annotation_position="top left"
    )
    fig.add_vline(x=max_speed_display, line_dash="dash", line_color="yellow", annotation_text="Max Speed")

    return _dark_layout(fig, "Lift / Drag / Thrust vs Airspeed", f"Airspeed ({speed_label})", "Force (N)")


def _build_ld_chart(airspeed_display, speed_label, lift, drag):
    """Lift-to-Drag ratio vs Airspeed — aerodynamic efficiency."""
    fig = go.Figure()
    ld_curve = lift / np.maximum(drag, 1e-6)
    fig.add_trace(go.Scatter(
        x=airspeed_display, y=ld_curve, mode="lines", name="L/D Ratio", line=dict(width=PLOT_LINE_WIDTH),
        hovertemplate=f"Airspeed: %{{x:.1f}} {speed_label}<br>L/D: %{{y:.2f}}<extra></extra>"
    ))
    return _dark_layout(fig, "Lift-to-Drag Ratio vs Airspeed", f"Airspeed ({speed_label})", "L/D Ratio")


def _build_velocity_time_chart(time_data, vel_display, speed_label, takeoff_time, takeoff_speed_display):
    """Ground speed during the takeoff roll, with the liftoff point
    marked if the aircraft got off the ground."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_data, y=vel_display, mode="lines", name="Ground Speed", line=dict(width=PLOT_LINE_WIDTH),
        hovertemplate=f"t=%{{x:.1f}}s<br>%{{y:.1f}} {speed_label}<extra></extra>"
    ))
    if takeoff_time is not None and takeoff_speed_display is not None:
        fig.add_trace(go.Scatter(
            x=[takeoff_time], y=[takeoff_speed_display], mode="markers+text",
            marker=dict(size=MARKER_SIZE, color="lime", symbol="star"),
            text=["Liftoff"], textposition="top center", name="Liftoff"
        ))
    return _dark_layout(fig, "Ground Speed vs Time", "Time (s)", f"Speed ({speed_label})")


def _build_acceleration_chart(time_data, accel_time):
    """Acceleration during the takeoff roll."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_data, y=accel_time, mode="lines", name="Acceleration", line=dict(width=PLOT_LINE_WIDTH),
        hovertemplate="t=%{x:.1f}s<br>%{y:.2f} m/s²<extra></extra>"
    ))
    return _dark_layout(fig, "Acceleration vs Time", "Time (s)", "Acceleration (m/s²)")


def _build_distance_chart(time_data, dist_time):
    """Ground roll distance covered during the takeoff roll."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_data, y=dist_time, mode="lines", name="Distance", line=dict(width=PLOT_LINE_WIDTH),
        hovertemplate="t=%{x:.1f}s<br>%{y:.2f} m<extra></extra>"
    ))
    return _dark_layout(fig, "Ground Roll Distance vs Time", "Time (s)", "Distance (m)")


def create_flight_graphs(
    unit,
    airspeed,
    lift,
    drag,
    thrust_curve,
    stall_speed_ms,
    max_airspeed_ms,
    required_lift_n,
    time_data,
    vel_time,
    accel_time,
    dist_time,
    takeoff_time=None,
    takeoff_speed_ms=None,
):
    """Build every chart AetherFlight displays, from already-computed
    physics/simulation results.

    `unit` is "m/s" or "km/h" and controls display conversion for every
    speed axis — every value passed in is SI.

    Returns a dict of five Plotly figures:
        lift_drag_thrust, ld_ratio, velocity_time, acceleration_time, distance_time

    (A separate "Thrust vs Drag" chart was dropped in v1.0 — it was a
    strict subset of lift_drag_thrust, with no information the primary
    chart didn't already show.)
    """
    convert, speed_label = _speed_converter(unit)

    airspeed_display = convert(airspeed)
    stall_display = float(convert(stall_speed_ms))
    max_speed_display = float(convert(max_airspeed_ms))
    vel_time_display = convert(vel_time)
    takeoff_speed_display = float(convert(takeoff_speed_ms)) if takeoff_speed_ms is not None else None

    return {
        "lift_drag_thrust": _build_aero_chart(
            airspeed_display, speed_label, lift, drag, thrust_curve,
            stall_display, max_speed_display, required_lift_n
        ),
        "ld_ratio": _build_ld_chart(airspeed_display, speed_label, lift, drag),
        "velocity_time": _build_velocity_time_chart(
            time_data, vel_time_display, speed_label, takeoff_time, takeoff_speed_display
        ),
        "acceleration_time": _build_acceleration_chart(time_data, accel_time),
        "distance_time": _build_distance_chart(time_data, dist_time),
    }
