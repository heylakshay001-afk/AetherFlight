# ========================= graphs_test.py =========================

import plotly.graph_objects as go
import numpy as np


def dark_layout(fig, title, x, y):
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        font=dict(color="white")
    )
    return fig


def create_flight_graphs(
    unit,
    velocity,
    lift,
    drag,
    thrust_curve,
    display_max,
    stall_speed,
    label,
    time_data,
    vel_time,
    accel_time,
    dist_time
):
    # ---------------- VELOCITY DISPLAY ----------------
    if unit == "km/h":
        velocity_display = velocity * 3.6
        stall_display = stall_speed * 3.6
        vel_display = np.array(vel_time) * 3.6
        vel_label = "km/h"
    else:
        velocity_display = velocity
        stall_display = stall_speed
        vel_display = np.array(vel_time)
        vel_label = "m/s"

    # ================= LIFT / DRAG =================
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=velocity_display, y=lift,
        mode='lines', name='Lift', line=dict(width=3)
    ))
    fig.add_trace(go.Scatter(
        x=velocity_display, y=drag,
        mode='lines', name='Drag', line=dict(width=3)
    ))
    fig.add_trace(go.Scatter(
        x=velocity_display, y=thrust_curve,
        mode='lines', name='Thrust', line=dict(width=3)
    ))

    # Max speed marker — interpolate thrust at display_max on correct x-axis.
    fig.add_trace(go.Scatter(
        x=[display_max],
        y=[np.interp(display_max, velocity_display, thrust_curve)],
        mode='markers',
        marker=dict(size=12, color='yellow', symbol='star'),
        name='Max Speed'
    ))

    fig.add_vline(
        x=stall_display,
        line_dash="dash",
        line_color="red",
        annotation_text="Stall Speed"
    )
    fig.add_vrect(
        x0=0, x1=stall_display,
        fillcolor="red", opacity=0.1, line_width=0,
        annotation_text="Stall Region",
        annotation_position="top left"
    )

    dark_layout(fig, "Lift / Drag Analysis", f"Velocity ({label})", "Force (N)")

    # ================= VELOCITY / TIME =================
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=time_data, y=vel_display,
        mode="lines", name="Velocity", line=dict(width=3)
    ))
    dark_layout(fig2, "Velocity vs Time", "Time (s)", f"Speed ({vel_label})")

    # ================= L/D RATIO =================
    fig_ld = go.Figure()
    ld_curve = lift / np.maximum(drag, 1e-6)
    fig_ld.add_trace(go.Scatter(
        x=velocity_display, y=ld_curve,
        mode="lines", name="L/D Ratio", line=dict(width=3)
    ))
    dark_layout(fig_ld, "Lift-to-Drag Ratio", f"Velocity ({label})", "L/D Ratio")

    # ================= THRUST / DRAG =================
    fig_td = go.Figure()
    fig_td.add_trace(go.Scatter(
        x=velocity_display, y=drag,
        mode="lines", name="Drag", line=dict(width=3)
    ))
    fig_td.add_trace(go.Scatter(
        x=velocity_display, y=thrust_curve,
        mode="lines", name="Available Thrust", line=dict(width=3)
    ))
    dark_layout(fig_td, "Thrust vs Drag", f"Velocity ({label})", "Force (N)")

    # ================= ACCELERATION / TIME =================
    fig_accel = go.Figure()
    fig_accel.add_trace(go.Scatter(
        x=time_data, y=accel_time,
        mode="lines", name="Acceleration", line=dict(width=3)
    ))
    dark_layout(fig_accel, "Acceleration vs Time", "Time (s)", "Acceleration (m/s²)")

    # ================= TAKEOFF DISTANCE / TIME =================
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Scatter(
        x=time_data, y=dist_time,
        mode="lines", name="Distance", line=dict(width=3)
    ))
    dark_layout(fig_dist, "Takeoff Distance vs Time", "Time (s)", "Distance (m)")

    return {
        "lift_drag": fig,
        "velocity": fig2,
        "ld_ratio": fig_ld,
        "thrust_drag": fig_td,
        "acceleration": fig_accel,
        "distance": fig_dist
    }
