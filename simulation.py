"""
simulation.py — AetherFlight Takeoff Simulation
================================================
Time-steps the takeoff ground roll using RK2 (Heun's method) integration.

Module boundary: this file owns ONLY the time-stepping and takeoff/failure
logic. Every force and coefficient it needs comes from physics.py's
compute_lift_drag() / calculate_thrust_curve() — simulation.py never
re-derives an aerodynamic formula itself, so the takeoff roll and the
static analysis charts can never silently disagree with each other.
"""

import numpy as np
from physics import (
    compute_lift_drag,
    calculate_cl,
    calculate_induced_drag_factor,
    calculate_thrust_curve,
    AIRCRAFT_PRESETS,
    RHO,
    G,
    CL_MIN,
    CL_MAX,
    SWEEP_MIN_SPEED,
)

# ============================================================
# Simulation constants
# ============================================================
MAX_SIM_TIME = 60.0                    # s — give up looking for liftoff after this long
TIME_STEP = 0.1                        # s — RK2 integration step
ROLLING_FRICTION_COEFFICIENT = 0.03    # fraction of weight resisting the ground roll
LIFTOFF_LIFT_MARGIN = 1.05             # lift must exceed this multiple of weight to count as "off the ground"
ACCEL_SAFETY_FLOOR = -30.0             # m/s^2 — numerical safety clamp only, not a physical model feature
DIAGNOSTIC_SWEEP_POINTS = 200
DIAGNOSTIC_MAX_SPEED = 60.0            # m/s — upper bound of the speed sweep used for the pre-flight diagnosis


def _net_acceleration_and_lift(v, wind, mass, weight_n, wing_area, chord, cd_base,
                                aoa, stall_aoa, parasitic_multiplier, induced_drag_k,
                                lift_multiplier, stall_softness, thrust, pitch_speed,
                                thrust_type):
    """One evaluation of the equations of motion at ground speed v. Called
    twice per RK2 timestep (predictor + corrector) so the physics is
    evaluated identically both times."""
    v_air = max(SWEEP_MIN_SPEED, v + wind)

    # ---- aerodynamic calculation (delegated to physics.py) ----
    lift, drag, _, _, _ = compute_lift_drag(
        v_air, wing_area, chord, cd_base, aoa, stall_aoa,
        parasitic_multiplier, induced_drag_k, lift_multiplier, stall_softness
    )

    # ---- forces -> net acceleration ----
    rolling_friction = ROLLING_FRICTION_COEFFICIENT * weight_n
    thrust_dynamic = max(0.0, float(calculate_thrust_curve(thrust, v_air, pitch_speed, thrust_type)))

    net_force = thrust_dynamic - drag - rolling_friction
    accel = net_force / mass
    accel = max(accel, ACCEL_SAFETY_FLOOR)  # numerical safety clamp only

    return accel, lift


def _diagnose(weight, wing_area, thrust, cd_base, aoa, stall_aoa, chord, wind,
              pitch_speed, aircraft_class):
    """Cheap analytic estimate of (a) the airspeed needed to lift off at the
    current AoA, and (b) the max airspeed this config can actually reach.
    Always computed — useful even on success, as a margin indicator — and
    used to classify *why* a takeoff failed.

    Returns (required_liftoff_speed_ms, max_reachable_speed_ms,
    static_thrust_n, static_resistance_n).
    """
    preset = AIRCRAFT_PRESETS[aircraft_class]
    weight_n = weight * G
    induced_drag_k, _ = calculate_induced_drag_factor(wing_area, chord, preset["oswald_efficiency"])

    # CL at the takeoff-roll AoA, assuming a representative Re correction.
    cl_cruise = calculate_cl(aoa, stall_aoa, 1.0, preset["stall_softness"]) * preset["lift_multiplier"]
    cl_cruise = np.clip(cl_cruise, CL_MIN, CL_MAX)

    required_liftoff_speed = np.sqrt((2 * LIFTOFF_LIFT_MARGIN * weight_n) / (RHO * wing_area * cl_cruise))

    v_sweep = np.linspace(SWEEP_MIN_SPEED, DIAGNOSTIC_MAX_SPEED, DIAGNOSTIC_SWEEP_POINTS)
    v_air_sweep = np.clip(v_sweep + wind, SWEEP_MIN_SPEED, None)
    thrust_sweep = calculate_thrust_curve(thrust, v_air_sweep, pitch_speed, preset["thrust_type"])
    cd_approx = cd_base * preset["parasitic_multiplier"] + induced_drag_k * cl_cruise ** 2
    drag_sweep = 0.5 * RHO * v_air_sweep ** 2 * wing_area * cd_approx

    reachable = np.where(thrust_sweep >= drag_sweep)[0]
    max_reachable_speed = v_air_sweep[reachable[-1]] if len(reachable) > 0 else 0.0

    static_thrust = float(calculate_thrust_curve(thrust, SWEEP_MIN_SPEED, pitch_speed, preset["thrust_type"]))
    static_resistance = ROLLING_FRICTION_COEFFICIENT * weight_n

    return required_liftoff_speed, max_reachable_speed, static_thrust, static_resistance


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
    """Simulate the takeoff ground roll via RK2 (Heun's method) time
    integration until the aircraft generates LIFTOFF_LIFT_MARGIN x weight
    of lift, or MAX_SIM_TIME is reached.

    Returns a dict with the full time-history (vel_time / dist_time /
    time_data / accel_time, all SI units) plus headline results
    (takeoff_distance / takeoff_speed_ms / takeoff_time) and a
    `takeoff_reason` explaining success, or the specific failure mode:
    "success", "insufficient_static_thrust", "insufficient_top_speed",
    or "timeout".
    """
    # ===================== Step 1: Validation + setup =====================
    if weight <= 0:
        raise ValueError(f"weight must be positive, got {weight}")
    if wing_area <= 0:
        raise ValueError(f"wing_area must be positive, got {wing_area}")
    if chord <= 0:
        raise ValueError(f"chord must be positive, got {chord}")
    if thrust < 0:
        raise ValueError(f"thrust cannot be negative, got {thrust}")
    if pitch_speed <= 0:
        raise ValueError(f"pitch_speed must be positive, got {pitch_speed}")
    if aircraft_class not in AIRCRAFT_PRESETS:
        raise ValueError(f"Unknown aircraft_class '{aircraft_class}'. Must be one of {list(AIRCRAFT_PRESETS)}")

    mass = weight
    weight_n = mass * G
    preset = AIRCRAFT_PRESETS[aircraft_class]
    parasitic_multiplier = preset["parasitic_multiplier"]
    oswald_efficiency = preset["oswald_efficiency"]
    stall_softness = preset["stall_softness"]
    lift_multiplier = preset["lift_multiplier"]
    thrust_type = preset["thrust_type"]
    induced_drag_k, _ = calculate_induced_drag_factor(wing_area, chord, oswald_efficiency)

    # ===================== Step 2: Simulation state/time =====================
    v = 0.0
    distance = 0.0
    time_elapsed = 0.0
    takeoff = False
    vel_time, dist_time, time_data, accel_time = [], [], [], []

    state_args = (wind, mass, weight_n, wing_area, chord, cd_base, aoa, stall_aoa,
                  parasitic_multiplier, induced_drag_k, lift_multiplier, stall_softness,
                  thrust, pitch_speed, thrust_type)

    while time_elapsed < MAX_SIM_TIME:
        # ============ Step 3 + 4: aero + forces, RK2 predictor ============
        k1, _ = _net_acceleration_and_lift(v, *state_args)
        v_pred = max(0.0, v + TIME_STEP * k1)

        # ============ Step 3 + 4: aero + forces, RK2 corrector ============
        k2, _ = _net_acceleration_and_lift(v_pred, *state_args)

        accel_avg = 0.5 * (k1 + k2)                      # Heun's method (RK2)
        v_next = max(0.0, v + TIME_STEP * accel_avg)
        distance += 0.5 * (v + v_next) * TIME_STEP        # trapezoidal integration
        time_elapsed += TIME_STEP
        v = v_next

        # ============ Step 5: Takeoff detection ============
        _, lift_now = _net_acceleration_and_lift(v, *state_args)

        vel_time.append(v)
        dist_time.append(distance)
        time_data.append(time_elapsed)
        accel_time.append(accel_avg)

        if lift_now >= LIFTOFF_LIFT_MARGIN * weight_n:
            takeoff = True
            break

    # ===================== Step 6: Failure diagnostics =====================
    required_liftoff_speed, max_reachable_speed, static_thrust, static_resistance = _diagnose(
        weight, wing_area, thrust, cd_base, aoa, stall_aoa, chord, wind, pitch_speed, aircraft_class
    )

    if takeoff:
        reason = "success"
    elif static_thrust <= static_resistance:
        reason = "insufficient_static_thrust"
    elif max_reachable_speed < required_liftoff_speed:
        reason = "insufficient_top_speed"
    else:
        reason = "timeout"  # the pre-check said it should succeed, but didn't within MAX_SIM_TIME

    # ===================== Step 7: Return data =====================
    return {
        "takeoff_distance": distance if takeoff else None,          # m
        "takeoff_speed_ms": v if takeoff else None,                 # m/s
        "takeoff_time": time_elapsed if takeoff else None,          # s
        "takeoff": takeoff,
        "takeoff_reason": reason,
        "required_liftoff_speed_ms": required_liftoff_speed,        # m/s
        "max_reachable_speed_ms": max_reachable_speed,               # m/s
        "vel_time": vel_time,        # m/s per sample
        "dist_time": dist_time,      # m per sample
        "time_data": time_data,      # s per sample
        "accel_time": accel_time,    # m/s^2 per sample
    }


# ===================== Step 8: Self-test =====================
if __name__ == "__main__":
    cases = [
        ("Normal / should succeed",          dict(weight=2.5, wing_area=0.25, thrust=80.0, cd_base=0.05,
                                                    aoa=5.0, stall_aoa=15.0, chord=0.15, wind=0.0,
                                                    pitch_speed=45.0, aircraft_class="Trainer")),
        ("Tiny thrust / can't move",         dict(weight=2.5, wing_area=0.25, thrust=0.3, cd_base=0.05,
                                                    aoa=5.0, stall_aoa=15.0, chord=0.15, wind=0.0,
                                                    pitch_speed=45.0, aircraft_class="Trainer")),
        ("Underpowered+draggy / no liftoff", dict(weight=2.5, wing_area=0.05, thrust=6.0, cd_base=0.15,
                                                    aoa=5.0, stall_aoa=15.0, chord=0.15, wind=0.0,
                                                    pitch_speed=20.0, aircraft_class="Aerobatic")),
        ("Headwind (should shorten roll)",   dict(weight=2.5, wing_area=0.25, thrust=80.0, cd_base=0.05,
                                                    aoa=5.0, stall_aoa=15.0, chord=0.15, wind=15 / 3.6,
                                                    pitch_speed=45.0, aircraft_class="Trainer")),
    ]

    print(f"{'Case':38s} {'reason':26s} {'dist(m)':>9s} {'v(m/s)':>8s} {'t(s)':>6s}")
    print("-" * 92)
    for name, params in cases:
        r = simulate_takeoff(**params)
        dist = f"{r['takeoff_distance']:.2f}" if r["takeoff_distance"] is not None else "-"
        spd = f"{r['takeoff_speed_ms']:.1f}" if r["takeoff_speed_ms"] is not None else "-"
        t = f"{r['takeoff_time']:.1f}" if r["takeoff_time"] is not None else "-"
        print(f"{name:38s} {r['takeoff_reason']:26s} {dist:>9s} {spd:>8s} {t:>6s}")

    print("\nValidation check (should raise):")
    try:
        simulate_takeoff(weight=-1, wing_area=0.25, thrust=80.0, cd_base=0.05,
                          aoa=5.0, stall_aoa=15.0, chord=0.15, wind=0.0,
                          pitch_speed=45.0, aircraft_class="Trainer")
    except ValueError as e:
        print(f"  Correctly rejected: {e}")
