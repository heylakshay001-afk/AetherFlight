# ========================= simulation_test.py =========================

import numpy as np
from physics import (
    get_drag_factor,
    calculate_cl,
    RHO,
    MU,
    G,
    AIRCRAFT_PRESETS
)


def simulate_takeoff(
    weight,
    wing_area,
    thrust,
    cd_base,
    aoa,
    stall_aoa,
    chord,
    wind,
    plane_type,
    max_prop_speed,
    aircraft_class
):
    mass = weight
    W = mass * G
    k = get_drag_factor(plane_type)

    preset = AIRCRAFT_PRESETS[aircraft_class]
    drag_multiplier = preset["drag_multiplier"]
    stall_softness = preset["stall_softness"]
    lift_multiplier = preset["lift_multiplier"]
    thrust_type = preset["thrust_type"]

    dt = 0.1
    vel_time = []
    dist_time = []
    time_data = []
    accel_time = []

    v = 0.0
    distance = 0.0
    time_elapsed = 0.0
    takeoff = False

    while time_elapsed < 60:
        v_air = max(0.1, v + wind)

        Re_local = (RHO * v_air * chord) / MU
        re_factor = np.clip((Re_local / 100000) ** 0.15, 0.6, 1.15)

        # BUG FIX 3: was calling calculate_cl with only 3 args — missing stall_softness → TypeError.
        cl_eff = calculate_cl(aoa, stall_aoa, re_factor, stall_softness)
        cl_eff *= lift_multiplier
        cl_eff = np.clip(cl_eff, 0, 2.0)

        cd_local = (cd_base + k * cl_eff ** 2) * drag_multiplier
        cd_local *= (1 / re_factor)

        lift = 0.5 * RHO * v_air ** 2 * wing_area * cl_eff
        drag = 0.5 * RHO * v_air ** 2 * wing_area * cd_local
        rolling_friction = 0.03 * W

        # BUG FIX 4: EDF thrust type was never handled in simulation (always used prop formula).
        if thrust_type == "edf":
            thrust_dynamic = thrust * (
                0.75 + 0.25 * np.exp(-v_air / max_prop_speed)
            )
        else:
            thrust_dynamic = thrust * np.exp(-v_air / max_prop_speed)

        thrust_dynamic = max(0.0, thrust_dynamic)

        net_force = thrust_dynamic - drag - rolling_friction
        acceleration = net_force / mass
        acceleration = max(acceleration, -5.0)

        v = max(0.0, v + acceleration * dt)
        distance += v * dt

        vel_time.append(v)
        dist_time.append(distance)
        time_data.append(time_elapsed)
        accel_time.append(acceleration)

        if lift >= 1.05 * W:
            takeoff = True
            break

        time_elapsed += dt

    if takeoff:
        takeoff_distance = distance
        takeoff_speed = v * 3.6  # stored in km/h.
        takeoff_time = time_elapsed
    else:
        takeoff_distance = None
        takeoff_speed = None
        takeoff_time = None

    return {
        "takeoff_distance": takeoff_distance,
        "takeoff_speed": takeoff_speed,
        "takeoff_time": takeoff_time,
        "vel_time": vel_time,
        "dist_time": dist_time,
        "time_data": time_data,
        "accel_time": accel_time,
        "takeoff": takeoff
    }