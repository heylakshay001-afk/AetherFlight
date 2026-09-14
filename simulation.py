# ========================= simulation.py =========================
# v0.2 — Phase 2: Simulation rewrite
#
# Changes from v0.1:
#   1. Analytic pre-flight check BEFORE running the timestep loop: if the
#      aircraft can't even overcome rolling friction, or its maximum
#      reachable airspeed is below what's needed to lift off, we say so
#      immediately with a specific reason instead of silently burning
#      600 iterations of "nothing happens."
#   2. Time integration upgraded from forward Euler to Heun's method (RK2)
#      for velocity — meaningfully more accurate near liftoff, where lift
#      ramps quickly as v^2.
#   3. Distance integration upgraded from "v_after * dt" to trapezoidal
#      (average of v_before/v_after) — removes a small systematic bias.
#   4. Every result now carries a `takeoff_reason`: "success",
#      "insufficient_static_thrust", "insufficient_top_speed", or
#      "timeout" — so the UI (Phase 3) can explain *why* something failed
#      instead of just saying "N/A".

import numpy as np
from physics import (
    calculate_cl,
    calculate_induced_drag_factor,
    calculate_thrust_curve,
    RHO,
    MU,
    G,
    AIRCRAFT_PRESETS
)

MAX_SIM_TIME = 60.0
DT = 0.1


def _instantaneous_accel_and_lift(v, wind, mass, W, wing_area, chord, cd_base,
                                   aoa, stall_aoa, parasitic_multiplier, k,
                                   lift_multiplier, stall_softness, thrust,
                                   pitch_speed, thrust_type):

    """One evaluation of the equations of motion at ground speed v.
    Shared by both RK2 stages so the physics is defined in exactly one place."""
    v_air = max(0.1, v + wind)

    Re_local = (RHO * v_air * chord) / MU
    re_factor = np.clip((Re_local / 100000) ** 0.15, 0.6, 1.15)

    cl_eff = calculate_cl(aoa, stall_aoa, re_factor, stall_softness)
    cl_eff *= lift_multiplier
    cl_eff = np.clip(cl_eff, 0, 2.0)

    cd_local = (cd_base * parasitic_multiplier + k * cl_eff ** 2) * (1 / re_factor)

    lift = 0.5 * RHO * v_air ** 2 * wing_area * cl_eff
    drag = 0.5 * RHO * v_air ** 2 * wing_area * cd_local
    rolling_friction = 0.03 * W

    thrust_dynamic = max(0.0, float(calculate_thrust_curve(thrust, v_air, pitch_speed, thrust_type)))

    net_force = thrust_dynamic - drag - rolling_friction
    accel = net_force / mass
    accel = max(accel, -30.0)  # numerical safety clamp only — not a physical model feature

    return accel, lift


def _feasibility_precheck(weight, wing_area, thrust, cd_base, aoa, stall_aoa,
                           chord, wind, pitch_speed, aircraft_class):
    """Cheap analytic check, done before spending 600 timesteps finding out
    the same thing the hard way. Returns (required_liftoff_airspeed,
    max_reachable_airspeed, static_thrust, static_resistance)."""
    preset = AIRCRAFT_PRESETS[aircraft_class]
    W = weight * G
    k, _ = calculate_induced_drag_factor(wing_area, chord, preset["oswald_efficiency"])

    # CL at the takeoff-roll AoA, assuming a representative Re correction
    cl_cruise = calculate_cl(aoa, stall_aoa, 1.0, preset["stall_softness"]) * preset["lift_multiplier"]
    cl_cruise = np.clip(cl_cruise, 0.05, 2.0)

    required_liftoff_airspeed = np.sqrt((2 * 1.05 * W) / (RHO * wing_area * cl_cruise))

    v_sweep = np.linspace(0.1, 60, 200)
    v_air_sweep = np.clip(v_sweep + wind, 0.1, None)
    thrust_sweep = calculate_thrust_curve(thrust, v_air_sweep, pitch_speed, preset["thrust_type"])
    cd_approx = cd_base * preset["parasitic_multiplier"] + k * cl_cruise ** 2
    drag_sweep = 0.5 * RHO * v_air_sweep ** 2 * wing_area * cd_approx

    valid = np.where(thrust_sweep >= drag_sweep)[0]
    max_reachable_airspeed = v_air_sweep[valid[-1]] if len(valid) > 0 else 0.0

    static_thrust = float(calculate_thrust_curve(thrust, 0.1, pitch_speed, preset["thrust_type"]))
    static_resistance = 0.03 * W  # rolling friction at (near) zero speed, drag ~ 0

    return required_liftoff_airspeed, max_reachable_airspeed, static_thrust, static_resistance


def simulate_takeoff(
    weight,
    wing_area,
    thrust,
    cd_base,
    aoa,
    stall_aoa,
    chord,
    wind,
    pitch_speed,
    aircraft_class
):
    mass = weight
    W = mass * G
    preset = AIRCRAFT_PRESETS[aircraft_class]
    parasitic_multiplier = preset["parasitic_multiplier"]
    oswald_efficiency = preset["oswald_efficiency"]
    stall_softness = preset["stall_softness"]
    lift_multiplier = preset["lift_multiplier"]
    thrust_type = preset["thrust_type"]
    k, _ = calculate_induced_drag_factor(wing_area, chord, oswald_efficiency)

    req_liftoff_v, max_reachable_v, static_thrust, static_resistance = _feasibility_precheck(
        weight, wing_area, thrust, cd_base, aoa, stall_aoa, chord, wind, pitch_speed, aircraft_class
    )

    empty_series = {"vel_time": [], "dist_time": [], "time_data": [], "accel_time": []}

    # ---- Early exit 1: can't even overcome rolling friction ----
    if static_thrust <= static_resistance:
        return {
            "takeoff_distance": None, "takeoff_speed": None, "takeoff_time": None,
            "takeoff": False, "takeoff_reason": "insufficient_static_thrust",
            "required_liftoff_speed": req_liftoff_v * 3.6,
            "max_reachable_speed": max_reachable_v * 3.6,
            **empty_series,
        }

    # ---- Early exit 2: will never reach the speed needed to lift off ----
    if max_reachable_v < req_liftoff_v:
        return {
            "takeoff_distance": None, "takeoff_speed": None, "takeoff_time": None,
            "takeoff": False, "takeoff_reason": "insufficient_top_speed",
            "required_liftoff_speed": req_liftoff_v * 3.6,
            "max_reachable_speed": max_reachable_v * 3.6,
            **empty_series,
        }

    # ---- Full timestep simulation (RK2 / Heun's method) ----
    vel_time, dist_time, time_data, accel_time = [], [], [], []
    v = 0.0
    distance = 0.0
    time_elapsed = 0.0
    takeoff = False

    args = (wind, mass, W, wing_area, chord, cd_base, aoa, stall_aoa,
            parasitic_multiplier, k, lift_multiplier, stall_softness,
            thrust, pitch_speed, thrust_type)

    while time_elapsed < MAX_SIM_TIME:
        k1, _ = _instantaneous_accel_and_lift(v, *args)
        v_pred = max(0.0, v + DT * k1)
        k2, _ = _instantaneous_accel_and_lift(v_pred, *args)

        accel_avg = 0.5 * (k1 + k2)
        v_next = max(0.0, v + DT * accel_avg)

        distance += 0.5 * (v + v_next) * DT  # trapezoidal integration
        time_elapsed += DT
        v = v_next

        _, lift_now = _instantaneous_accel_and_lift(v, *args)

        vel_time.append(v)
        dist_time.append(distance)
        time_data.append(time_elapsed)
        accel_time.append(accel_avg)

        if lift_now >= 1.05 * W:
            takeoff = True
            break

    if takeoff:
        return {
            "takeoff_distance": distance,
            "takeoff_speed": v * 3.6,
            "takeoff_time": time_elapsed,
            "takeoff": True, "takeoff_reason": "success",
            "required_liftoff_speed": req_liftoff_v * 3.6,
            "max_reachable_speed": max_reachable_v * 3.6,
            "vel_time": vel_time, "dist_time": dist_time,
            "time_data": time_data, "accel_time": accel_time,
        }
    else:
        # The pre-check said it should succeed, but it didn't within.
        # MAX_SIM_TIME — rare, kept as a safety-net diagnosis.
        return {
            "takeoff_distance": None, "takeoff_speed": None, "takeoff_time": None,
            "takeoff": False, "takeoff_reason": "timeout",
            "required_liftoff_speed": req_liftoff_v * 3.6,
            "max_reachable_speed": max_reachable_v * 3.6,
            "vel_time": vel_time, "dist_time": dist_time,
            "time_data": time_data, "accel_time": accel_time,
        }