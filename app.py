"""
app.py — AetherFlight
======================
Streamlit front-end: collects aircraft parameters from the sidebar, runs
the aerodynamics (physics.py) and takeoff simulation (simulation.py), and
renders the results as metrics, status messages, and Plotly charts
(graphs.py).

Module boundary: this file owns UI/orchestration ONLY — no aerodynamic or
time-stepping math lives here; it just collects inputs, calls the two
engine modules, and displays what comes back.
"""

import numpy as np
import streamlit as st

from physics import calculate_aerodynamics, G
from simulation import simulate_takeoff, LIFTOFF_LIFT_MARGIN
from graphs import create_flight_graphs

MS_TO_KMH = 3.6
APP_VERSION = "1.0.0"

# ============================================================
# Slider ranges per "Aircraft Type" — purely a UI convenience for picking
# sensible defaults. It has NO effect on the physics itself (see
# physics.py / simulation.py, which never receive plane_type at all).
# ============================================================
PLANE_TYPE_RANGES = {
    "Light RC": {
        "weight":    dict(min_value=0.01, max_value=1.0, value=0.8, step=0.001),
        "wing_area": dict(min_value=0.05, max_value=0.3, value=0.12, step=0.001),
        "thrust":    dict(min_value=5.0, max_value=40.0, value=15.0, step=0.1),
    },
    "Sport RC": {
        "weight":    dict(min_value=1.0, max_value=5.0, value=2.5, step=0.001),
        "wing_area": dict(min_value=0.1, max_value=0.5, value=0.25, step=0.001),
        "thrust":    dict(min_value=50.0, max_value=150.0, value=50.0, step=1.0),
    },
    "Heavy RC": {
        "weight":    dict(min_value=5.0, max_value=15.0, value=10.0, step=0.001),
        "wing_area": dict(min_value=0.3, max_value=2.0, value=0.6, step=0.001),
        "thrust":    dict(min_value=100.0, max_value=500.0, value=150.0, step=1.0),
    },
}


# ============================================================
# Small formatting helpers — the ONLY place speed unit conversion happens
# in the UI, so every metric/chart stays consistent with the sidebar's
# unit toggle.
# ============================================================
def convert_speed(value_ms, unit):
    """Convert an SI speed (m/s) to the display unit. Returns a raw float."""
    return value_ms * MS_TO_KMH if unit == "km/h" else value_ms


def format_speed(value_ms, unit, decimals=1):
    """Convert + format an SI speed (m/s) as a display string, e.g. '42.8 km/h'."""
    return f"{convert_speed(value_ms, unit):.{decimals}f} {unit}"


def inject_styles():
    """App-wide dark theme and metric-card styling."""
    st.markdown("""
    <style>
    body { background-color: #0b0f1a; color: #e6f1ff; }
    [data-testid="stMetric"] {
        background-color: #111827;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0px 0px 10px rgba(0,255,200,0.1);
    }
    h1, h2, h3 { color: #00ffd5; }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# Sidebar — input collection
# ============================================================
def render_sidebar():
    """Collect every aircraft/flight-condition input from the sidebar.
    Returns a dict of raw values, ready to hand to physics/simulation."""
    st.sidebar.header("Aircraft Setup")

    plane_type = st.sidebar.selectbox(
        "Aircraft Type", list(PLANE_TYPE_RANGES) + ["Custom"],
        help="Picks sensible slider ranges below. Has no effect on the physics."
    )
    aircraft_class = st.sidebar.selectbox(
        "Aircraft Class",
        ["Trainer", "Flying Wing", "EDF Jet", "Glider", "Aerobatic", "Warbird"],
        help="Sets drag/lift/efficiency characteristics typical of this airframe style."
    )

    # Each section is a collapsible expander so the sidebar can be folded
    # down to just the controls you're actively tweaking. Widgets created
    # inside a `with` block render inside that expander.
    with st.sidebar.expander("Mass & Thrust", expanded=True):
        if plane_type in PLANE_TYPE_RANGES:
            ranges = PLANE_TYPE_RANGES[plane_type]
            weight = st.slider("Weight (kg)", **ranges["weight"])
            wing_area = st.slider("Wing Area (m²)", **ranges["wing_area"])
            thrust = st.slider("Thrust (N)", **ranges["thrust"])
        else:
            weight = st.number_input("Weight (kg)", min_value=0.01, value=1.5)
            wing_area = st.number_input("Wing Area (m²)", min_value=0.01, value=0.2)
            thrust = st.number_input("Thrust (N)", min_value=0.1, value=30.0)

    with st.sidebar.expander("Wing Geometry", expanded=True):
        cd_base = st.slider(
            "Parasitic Drag Coefficient (Cd₀)", 0.02, 0.2, 0.05, 0.001,
            help="Baseline drag from everything that ISN'T lift-induced (skin friction, fuselage, etc.)."
        )
        aoa = st.slider(
            "Angle of Attack (°)", -5.0, 20.0, 5.0, 0.1,
            help="The angle the wing meets the air at. More lift, up to a point — then it stalls."
        )
        stall_aoa = st.slider(
            "Critical AoA (°)", 10.0, 20.0, 15.0,
            help="The AoA at which the wing stalls (lift collapses)."
        )
        chord = st.slider(
            "Mean Chord (m)", 0.05, 0.5, 0.15,
            help="Wing depth, front to back. Drives Reynolds number and, with wing area, aspect ratio."
        )

    with st.sidebar.expander("Flight Conditions", expanded=True):
        wind_kmh = st.slider("Headwind (+) / Tailwind (-) (km/h)", -30, 30, 0)
        wind = wind_kmh / MS_TO_KMH
        pitch_speed = st.slider(
            "Prop Pitch Speed (m/s)", 10.0, 80.0, 45.0,
            help="Airspeed at which a fixed-pitch propeller's thrust theoretically reaches zero."
        )

    st.sidebar.markdown("#### Display")
    unit = st.sidebar.radio("Speed Unit", ["km/h", "m/s"], horizontal=True)

    return dict(
        plane_type=plane_type, aircraft_class=aircraft_class,
        weight=weight, wing_area=wing_area, thrust=thrust,
        cd_base=cd_base, aoa=aoa, stall_aoa=stall_aoa, chord=chord,
        wind=wind, pitch_speed=pitch_speed, unit=unit,
    )


# ============================================================
# Tab 1 — Flight Performance
# ============================================================
def render_efficiency_badge(ld_ratio):
    """A single-line qualitative read on the aircraft's L/D ratio."""
    if ld_ratio > 15:
        st.success("🟢 Excellent aerodynamic efficiency (L/D > 15)")
    elif ld_ratio > 10:
        st.info("🔵 Good flight characteristics (L/D > 10)")
    elif ld_ratio > 6:
        st.warning("🟡 Marginal efficiency (L/D > 6)")
    else:
        st.error("🔴 Poor aerodynamic performance (L/D ≤ 6)")


def render_performance_tab(aero, graphs, unit):
    """Key aerodynamic metrics plus the speed-domain charts (lift/drag/
    thrust and L/D ratio)."""
    st.subheader("Key Metrics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Stall Speed", format_speed(aero["stall_speed"], unit),
            help="Minimum airspeed the wing can generate enough lift at, at its critical AoA."
        )
    with col2:
        margin = convert_speed(aero["max_airspeed"] - aero["stall_speed"], unit)
        st.metric(
            "Max Speed (ground)", format_speed(aero["max_groundspeed"], unit),
            delta=f"{margin:+.1f} {unit} vs. stall",
            help="Groundspeed where available thrust equals drag. Reported as groundspeed "
                 "(what you'd see over the field), which differs from airspeed when wind != 0."
        )
    with col3:
        st.metric("Wing Loading", f"{aero['wing_loading']:.2f} kg/m²")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("L/D Ratio (max)", f"{aero['ld_ratio']:.2f}")
    with col5:
        st.metric("Aspect Ratio", f"{aero['aspect_ratio']:.2f}")
    with col6:
        st.metric("CLmax", f"{aero['cl_max']:.3f}", help="Max lift coefficient, at the critical AoA.")

    render_efficiency_badge(aero["ld_ratio"])

    st.divider()
    st.subheader("Aerodynamics vs Airspeed")
    col_main, col_side = st.columns([3, 2])
    with col_main:
        st.plotly_chart(graphs["lift_drag_thrust"], use_container_width=True)
    with col_side:
        st.plotly_chart(graphs["ld_ratio"], use_container_width=True)


# ============================================================
# Tab 2 — Takeoff Roll
# ============================================================
def render_takeoff_status(sim, unit):
    """The single authoritative takeoff status message, driven by
    `takeoff_reason` so the explanation always matches what actually
    happened in the simulation."""
    reason = sim["takeoff_reason"]

    if reason == "success":
        st.success("✅ This configuration takes off successfully.")
    elif reason == "insufficient_static_thrust":
        st.error(
            "❌ Thrust can't even overcome rolling friction — the aircraft will never start "
            "moving. Increase thrust or reduce weight."
        )
    elif reason == "insufficient_top_speed":
        req = format_speed(sim["required_liftoff_speed_ms"], unit, decimals=0)
        mx = format_speed(sim["max_reachable_speed_ms"], unit, decimals=0)
        st.warning(
            f"⚠️ Needs roughly **{req}** to lift off, but this configuration tops out around "
            f"**{mx}**. Increase thrust, reduce weight, or increase wing area."
        )
        required = sim["required_liftoff_speed_ms"]
        reachable_ratio = min(1.0, sim["max_reachable_speed_ms"] / required) if required > 0 else 0.0
        st.progress(reachable_ratio, text=f"Reaches {reachable_ratio * 100:.0f}% of the speed needed to lift off")
    else:  # "timeout"
        st.warning("⚠️ Didn't lift off within the simulated time window — check your configuration.")


def render_takeoff_tab(sim, graphs, unit):
    """Takeoff result status, headline numbers, and the three time-domain
    charts (velocity / acceleration / distance vs. time)."""
    st.subheader("Takeoff Result")
    render_takeoff_status(sim, unit)

    col1, col2, col3 = st.columns(3)
    with col1:
        value = f"{sim['takeoff_distance']:.1f} m" if sim["takeoff_distance"] is not None else "—"
        st.metric("Takeoff Distance", value)
    with col2:
        value = format_speed(sim["takeoff_speed_ms"], unit) if sim["takeoff_speed_ms"] is not None else "—"
        st.metric("Takeoff Speed", value)
    with col3:
        value = f"{sim['takeoff_time']:.1f} s" if sim["takeoff_time"] is not None else "—"
        st.metric("Takeoff Time", value)

    st.divider()
    st.subheader("Takeoff Roll Over Time")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(graphs["velocity_time"], use_container_width=True)
    with col2:
        st.plotly_chart(graphs["acceleration_time"], use_container_width=True)
    with col3:
        st.plotly_chart(graphs["distance_time"], use_container_width=True)


# ============================================================
# Tab 3 — Physics Details
# ============================================================
def render_physics_tab(inputs, aero):
    """Aircraft configuration, derived coefficients, and the governing
    equations — the "show your work" tab."""
    st.subheader("Aircraft Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Type:** {inputs['plane_type']}  |  **Class:** {inputs['aircraft_class']}")
        st.write(f"**Weight:** {inputs['weight']:.3f} kg  ({inputs['weight'] * G:.2f} N)")
        st.write(f"**Wing Area:** {inputs['wing_area']:.3f} m²  |  **Chord:** {inputs['chord']:.3f} m")
        st.write(f"**Aspect Ratio:** {aero['aspect_ratio']:.2f}")
    with col2:
        st.write(f"**Oswald Efficiency (e):** {aero['oswald_efficiency']:.2f}")
        st.write(f"**Induced Drag Factor (k):** {aero['induced_drag_factor']:.4f}")
        st.write(f"**Parasitic Drag Multiplier:** {aero['parasitic_multiplier']:.2f}")
        st.write(f"**Reynolds Number (at max speed):** {np.max(aero['re_curve']):,.0f}")

    st.divider()
    st.subheader("Equations Used")
    st.latex(r"L = \tfrac{1}{2}\rho v^2 S C_L \qquad D = \tfrac{1}{2}\rho v^2 S C_D")
    st.latex(r"C_D = C_{D_0}\cdot(\text{parasitic mult.}) + \dfrac{C_L^2}{\pi \cdot AR \cdot e}")
    st.latex(r"V_{stall} = \sqrt{\dfrac{2W}{\rho\, S\, C_{L_{max}}}} \quad \text{(evaluated at the critical AoA)}")
    st.latex(r"Re = \dfrac{\rho\, v\, c}{\mu}")


# ============================================================
# Main
# ============================================================
def main():
    st.set_page_config(page_title="AetherFlight", layout="wide", page_icon="✈️")
    inject_styles()

    st.title("✈️ AetherFlight")
    st.caption("Aerodynamic analysis & takeoff simulation for RC aircraft")
    st.divider()

    inputs = render_sidebar()

    aero = calculate_aerodynamics(
        inputs["weight"], inputs["wing_area"], inputs["thrust"], inputs["cd_base"],
        inputs["aoa"], inputs["stall_aoa"], inputs["chord"], inputs["wind"],
        inputs["pitch_speed"], inputs["aircraft_class"]
    )
    sim = simulate_takeoff(
        inputs["weight"], inputs["wing_area"], inputs["thrust"], inputs["cd_base"],
        inputs["aoa"], inputs["stall_aoa"], inputs["chord"], inputs["wind"],
        inputs["pitch_speed"], inputs["aircraft_class"]
    )

    required_lift_n = LIFTOFF_LIFT_MARGIN * inputs["weight"] * G

    graphs = create_flight_graphs(
        inputs["unit"], aero["airspeed"], aero["lift"], aero["drag"], aero["thrust_curve"],
        aero["stall_speed"], aero["max_airspeed"], required_lift_n,
        sim["time_data"], sim["vel_time"], sim["accel_time"], sim["dist_time"],
        takeoff_time=sim["takeoff_time"], takeoff_speed_ms=sim["takeoff_speed_ms"],
    )

    tab1, tab2, tab3 = st.tabs(["📊 Flight Performance", "🛫 Takeoff Roll", "⚙️ Physics Details"])
    with tab1:
        render_performance_tab(aero, graphs, inputs["unit"])
    with tab2:
        render_takeoff_tab(sim, graphs, inputs["unit"])
    with tab3:
        render_physics_tab(inputs, aero)

    st.caption(f"AetherFlight v{APP_VERSION}")


if __name__ == "__main__":
    main()
