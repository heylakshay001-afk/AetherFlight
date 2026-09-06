# ========================= app.py =========================

import streamlit as st
import numpy as np
from physics import calculate_aerodynamics
from simulation import simulate_takeoff
from graphs import create_flight_graphs

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AetherFlight", layout="wide")

# ---------------- STYLING ----------------
st.markdown("""
<style>
body {
    background-color: #0b0f1a;
    color: #e6f1ff;
}
[data-testid="stMetric"] {
    background-color: #111827;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    box-shadow: 0px 0px 10px rgba(0,255,200,0.1);
}
h1, h2, h3 {
    color: #00ffd5;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("✈️ AetherFlight")
st.subheader("Flight Simulation & Analysis")
st.markdown("---")

# ========================= SIDEBAR =========================
st.sidebar.header("Aircraft Setup")

plane_type = st.sidebar.selectbox(
    "Aircraft Type",
    ["Light RC", "Sport RC", "Heavy RC", "Custom"]
)

aircraft_class = st.sidebar.selectbox(
    "Aircraft Class",
    ["Trainer", "Flying Wing", "EDF Jet", "Glider", "Aerobatic", "Warbird"]
)

# ---------------- PARAMETERS ----------------
with st.sidebar.expander("Parameters", expanded=True):
    if plane_type == "Light RC":
        weight = st.slider("Weight (kg)", 0.01, 1.0, 0.8, 0.001)
        wing_area = st.slider("Wing Area (m²)", 0.05, 0.3, 0.12, 0.001)
        thrust = st.slider("Thrust (N)", 5.0, 40.0, 15.0, 1.0)
    elif plane_type == "Sport RC":
        weight = st.slider("Weight (kg)", 1.0, 5.0, 2.5, 0.001)
        wing_area = st.slider("Wing Area (m²)", 0.1, 0.5, 0.25, 0.001)
        thrust = st.slider("Thrust (N)", 50.0, 150.0, 50.0, 1.0)
    elif plane_type == "Heavy RC":
        weight = st.slider("Weight (kg)", 5.0, 15.0, 10.0, 0.001)
        wing_area = st.slider("Wing Area (m²)", 0.3, 2.0, 0.6, 0.001)
        thrust = st.slider("Thrust (N)", 100.0, 500.0, 150.0, 1.0)
    else:
        weight = st.number_input("Weight (kg)", min_value=0.01, value=1.5)
        wing_area = st.number_input("Wing Area (m²)", min_value=0.01, value=0.2)
        thrust = st.number_input("Thrust (N)", min_value=0.1, value=30.0)

# ---------------- WING PARAMETERS ----------------
with st.sidebar.expander("Wing Parameters"):
    cd_base = st.slider("Drag Coefficient (Cd)", 0.02, 0.2, 0.05, 0.001)
    aoa = st.slider("Angle of Attack (°)", -5.0, 20.0, 5.0, 0.1)
    stall_aoa = st.slider("Critical AoA (°)", 10.0, 20.0, 15.0)
    chord = st.slider("Mean Chord (m)", 0.05, 0.5, 0.15)

# ---------------- FLIGHT CONDITIONS ----------------
with st.sidebar.expander("Flight Conditions"):
    wind = st.slider("Headwind (+) / Tailwind (-) km/h", -30, 30, 0) / 3.6
    max_prop_speed = st.slider("Prop Efficiency Limit (m/s)", 10.0, 80.0, 45.0)

# ---------------- SPEED UNIT ----------------
unit = st.sidebar.selectbox("Speed Unit", ["m/s", "km/h"])

# ========================= CALCULATIONS =========================
aero_results = calculate_aerodynamics(
    weight, wing_area, thrust, cd_base,
    aoa, stall_aoa, chord, wind,
    plane_type, max_prop_speed, aircraft_class
)

sim_results = simulate_takeoff(
    weight, wing_area, thrust, cd_base,
    aoa, stall_aoa, chord, wind,
    plane_type, max_prop_speed, aircraft_class
)

# ========================= EXTRACT DATA =========================
lift = aero_results["lift"]
drag = aero_results["drag"]
velocity = aero_results["velocity"]
# BUG FIX 5: use relative_velocity for x-axis so graphs correctly reflect.
# airspeed (wind-corrected). Without this, lift/drag curves are plotted.
# against the wrong velocity when wind != 0.
relative_velocity = aero_results["relative_velocity"]
stall_speed = aero_results["stall_speed"]
max_velocity = aero_results["max_velocity"]
ld_ratio = aero_results["ld_ratio"]
thrust_curve = aero_results["thrust_curve"]
wing_loading = aero_results["wing_loading"]
Re = np.max(aero_results["re_curve"])

takeoff_distance = sim_results["takeoff_distance"]
takeoff_speed = sim_results["takeoff_speed"]
takeoff_time = sim_results["takeoff_time"]
vel_time = sim_results["vel_time"]
dist_time = sim_results["dist_time"]
time_data = sim_results["time_data"]
accel_time = sim_results["accel_time"]

# ========================= DISPLAY UNITS =========================
if unit == "km/h":
    display_stall = stall_speed * 3.6
    display_max = max_velocity * 3.6
    label = "km/h"
else:
    display_stall = stall_speed
    display_max = max_velocity
    label = "m/s"

# ========================= GRAPHS =========================
graph_result = create_flight_graphs(
    unit,
    relative_velocity,   # BUG FIX 5: was velocity, now correctly relative_velocity
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
)

lift_dragG = graph_result["lift_drag"]
velocityG = graph_result["velocity"]
ld_ratioG = graph_result["ld_ratio"]
thrust_dragG = graph_result["thrust_drag"]
accelerationG = graph_result["acceleration"]
distanceG = graph_result["distance"]

# ========================= TABS =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "🧩 Analysis",
    "🛫 Simulation",
    "📊 Graphs",
    "⚙️ Physics"
])

# ========================= TAB 1: ANALYSIS =========================
with tab1:
    st.info(f"""
    Aircraft Type: {plane_type}  |  Class: {aircraft_class}
    Weight: {weight:.2f} kg  |  Wing Area: {wing_area:.2f} m²  |  Wing Loading: {wing_loading:.2f} kg/m²
    """)

    st.header("Aerodynamic Analysis")
    st.plotly_chart(lift_dragG, use_container_width=True)

    st.header("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    with col1:
        st.metric(f"Stall Speed ({label})", f"{display_stall:.1f}")

    with col2:
        st.metric(f"Max Speed ({label})", f"{display_max:.1f}")

    with col3:
        if takeoff_distance is not None:
            st.metric("Takeoff Distance (m)", f"{takeoff_distance:.1f}")
        else:
            st.metric("Takeoff Distance (m)", "N/A")

    with col4:
        if takeoff_speed is not None:
            takeoff_display = takeoff_speed if unit == "km/h" else takeoff_speed / 3.6
            st.metric(f"Takeoff Speed ({label})", f"{takeoff_display:.1f}")
        else:
            st.metric(f"Takeoff Speed ({label})", "N/A")

    with col5:
        if takeoff_time is not None:
            st.metric("Takeoff Time (s)", f"{takeoff_time:.1f}")
        else:
            st.metric("Takeoff Time (s)", "N/A")

    with col6:
        st.metric("Reynolds Number", f"{Re:,.0f}")

    # Aircraft validation.
    if stall_speed > max_velocity:
        st.error("❌ Aircraft cannot sustain stable flight — thrust insufficient to exceed stall speed.")
    elif takeoff_distance is None:
        st.warning("⚠️ Aircraft failed to take off within 60 seconds.")
    else:
        st.success("✅ Aircraft configuration is flyable.")

# ========================= TAB 2: SIMULATION =========================
with tab2:
    st.header("Takeoff Simulation")
    st.plotly_chart(velocityG, use_container_width=True)

    if ld_ratio > 15:
        st.success("🟢 Excellent Efficiency (L/D > 15)")
    elif ld_ratio > 10:
        st.info("🔵 Good Flight Characteristics (L/D > 10)")
    elif ld_ratio > 6:
        st.warning("🟡 Marginal Efficiency (L/D > 6)")
    else:
        st.error("🔴 Poor Aerodynamic Performance (L/D ≤ 6)")

# ========================= TAB 3: GRAPHS =========================
with tab3:
    colA, colB = st.columns(2)
    colC, colD = st.columns(2)

    with colA:
        st.plotly_chart(ld_ratioG, use_container_width=True)
    with colB:
        st.plotly_chart(thrust_dragG, use_container_width=True)
    with colC:
        st.plotly_chart(accelerationG, use_container_width=True)
    with colD:
        st.plotly_chart(distanceG, use_container_width=True)

# ========================= TAB 4: PHYSICS =========================
with tab4:
    st.header("Physics Breakdown")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Aircraft Config")
        st.write(f"*Type:* {plane_type}  |  *Class:* {aircraft_class}")
        st.write(f"*Weight:* {weight:.3f} kg  ({weight * 9.81:.2f} N)")
        st.write(f"*Wing Area:* {wing_area:.3f} m²")
        st.write(f"*Mean Chord:* {chord:.3f} m")
        st.write(f"*Wing Loading:* {wing_loading:.2f} kg/m²")
        st.write(f"*Thrust:* {thrust:.2f} N")
        st.write(f"*Drag Coefficient (Cd₀):* {cd_base:.3f}")
        st.write(f"*Angle of Attack:* {aoa:.1f}°  (stall at {stall_aoa:.1f}°)")

    with c2:
        st.subheader("Computed Results")
        st.write(f"*Stall Speed:* {display_stall:.2f} {label}")
        st.write(f"*Max Speed:* {display_max:.2f} {label}")
        st.write(f"*L/D Ratio (max):* {ld_ratio:.2f}")
        st.write(f"*Reynolds Number (max):* {Re:,.0f}")
        if takeoff_distance:
            st.write(f"*Takeoff Distance:* {takeoff_distance:.2f} m")
            st.write(f"*Takeoff Speed:* {takeoff_speed:.1f} km/h")
            st.write(f"*Takeoff Time:* {takeoff_time:.1f} s")
        else:
            st.write("*Takeoff:* Failed")

    st.subheader("Equations Used")
    st.latex(r"L = \frac{1}{2} \rho v^2 S C_L")
    st.latex(r"D = \frac{1}{2} \rho v^2 S C_D")
    st.latex(r"C_D = C_{D_0} + k C_L^2")
    st.latex(r"V_{stall} = \sqrt{\frac{2W}{\rho S C_{L_{max}}}}")
    st.latex(r"Re = \frac{\rho v c}{\mu}")