# ========================= physics.py =========================
# v0.2 — Phase 1: Physics cleanup
#
# Changes from v0.1:
#   1. True CLmax / stall speed: now computed at the CRITICAL AoA (stall_aoa),
#      independent of whatever cruise AoA the user has dialed in. Previously
#      "stall speed" accidentally tracked the current AoA slider.
#   2. Induced drag factor (k) is now derived from aspect ratio (AR = wing_area
#      / chord^2, rectangular-wing approximation) and a per-class Oswald
#      efficiency, instead of being keyed to plane_type (a weight category
#      with no aerodynamic meaning).
#   3. Thrust model replaced: fixed-pitch props now decay ~linearly to zero
#      at a "pitch speed" (physically motivated), EDF/jets decay gently and
#      floor at 50% of static thrust. Old model was an arbitrary exponential
#      that never reached zero.
#   4. Removed dead "stability" preset field (never used anywhere).
#   5. Removed plane_type from the physics/simulation signatures entirely —
#      it's now purely a UI convenience for picking slider ranges.

import numpy as np

RHO = 1.225
MU = 1.81e-5
G = 9.81

# parasitic_multiplier: scales form/interference drag (fuselage, struts, etc.)
# oswald_efficiency: how efficiently the wing distributes lift span-wise drives induced drag together with aspect ratio. Lower = worse (more induced drag at a given CL).
AIRCRAFT_PRESETS = {
    "Trainer":     {"parasitic_multiplier": 1.00, "oswald_efficiency": 0.80, "stall_softness": 1.3, "lift_multiplier": 1.10, "thrust_type": "prop"},
    "Flying Wing": {"parasitic_multiplier": 0.90, "oswald_efficiency": 0.65, "stall_softness": 0.8, "lift_multiplier": 0.95, "thrust_type": "prop"},
    "EDF Jet":     {"parasitic_multiplier": 0.85, "oswald_efficiency": 0.75, "stall_softness": 0.7, "lift_multiplier": 0.80, "thrust_type": "edf"},
    "Glider":      {"parasitic_multiplier": 0.70, "oswald_efficiency": 0.90, "stall_softness": 1.5, "lift_multiplier": 1.20, "thrust_type": "prop"},
    "Aerobatic":   {"parasitic_multiplier": 1.10, "oswald_efficiency": 0.70, "stall_softness": 1.0, "lift_multiplier": 1.00, "thrust_type": "prop"},
    "Warbird":     {"parasitic_multiplier": 1.15, "oswald_efficiency": 0.75, "stall_softness": 0.9, "lift_multiplier": 0.95, "thrust_type": "prop"},
}
# Note: Flying Wing now has LOWER parasitic drag (no fuselage/tail) but WORSE oswald efficiency (no tail to trim out wingtip twist / poor span loading) that's the actual aerodynamic story with tailless aircraft, rather that the old flat "20% draggier, for no stated reason."


def calculate_aspect_ratio(wing_area, chord):
    """Rectangular-wing approximation: span = wing_area / chord, so
    AR = span / chord = wing_area / chord^2. Real AR for tapered/swept
    wings differs, but this is a reasonable stand-in given the sliders
    we actually expose."""
    return wing_area / chord ** 2


def calculate_induced_drag_factor(wing_area, chord, oswald_efficiency):
    """k in CD = CD0 + k*CL^2, from classic finite-wing theory:
    k = 1 / (pi * AR * e)"""
    AR = calculate_aspect_ratio(wing_area, chord)
    k = 1.0 / (np.pi * AR * oswald_efficiency)
    return k, AR


def calculate_cl(aoa, stall_aoa, re_factor, stall_softness):
    """CL vs AoA: thin-airfoil slope (2*pi per radian) below stall,
    exponential falloff past the critical angle."""
    if aoa <= stall_aoa:
        cl = 2 * np.pi * np.radians(aoa)
    else:
        cl = (2 * np.pi * np.radians(stall_aoa)) * np.exp(
            -(aoa - stall_aoa) / (5 * stall_softness)
        )
    cl *= re_factor
    return np.clip(cl, 0, 2.0)


def calculate_cl_max_and_stall_speed(weight, wing_area, chord, stall_aoa,
                                      stall_softness, lift_multiplier):
    """The wing's actual CLmax — evaluated at the CRITICAL AoA, not the
    current cruise AoA. Stall speed is a property of the wing, not of
    however the pilot currently has it trimmed.

    Re affects CLmax slightly, but Re itself depends on the stall speed
    we're solving for — so we do one predictor/refine pass (the
    dependence is weak, one pass converges essentially exactly)."""
    W = weight * G

    # Pass 1: ignore Re correction to get a first estimate
    cl_max_guess = calculate_cl(stall_aoa, stall_aoa, 1.0, stall_softness) * lift_multiplier
    cl_max_guess = np.clip(cl_max_guess, 0.05, 2.0)
    v_stall_guess = np.sqrt((2 * W) / (RHO * wing_area * cl_max_guess))

    # Pass 2: refine using Re at that estimated stall speed
    Re_stall = (RHO * v_stall_guess * chord) / MU
    re_factor_stall = np.clip((Re_stall / 100000) ** 0.15, 0.6, 1.15)
    cl_max = calculate_cl(stall_aoa, stall_aoa, re_factor_stall, stall_softness) * lift_multiplier
    cl_max = np.clip(cl_max, 0.05, 2.0)
    v_stall = np.sqrt((2 * W) / (RHO * wing_area * cl_max))

    return v_stall, cl_max, Re_stall


def calculate_thrust_curve(thrust, airspeed, pitch_speed, thrust_type):
    """Static thrust vs airspeed.
    - Fixed-pitch prop: falls off ~linearly to zero at the theoretical
      pitch speed (standard simplified static-thrust model for RC props).
    - EDF/jet: falls off gently, floored at 50% of static thrust — jets
      hold their thrust far better across the speed range than props do."""
    if thrust_type == "edf":
        return thrust * np.clip(1.0 - 0.2 * (airspeed / pitch_speed), 0.5, 1.0)
    else:
        return thrust * np.clip(1.0 - (airspeed / pitch_speed), 0.0, 1.0)


def calculate_aerodynamics(
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
    preset = AIRCRAFT_PRESETS[aircraft_class]
    parasitic_multiplier = preset["parasitic_multiplier"]
    oswald_efficiency = preset["oswald_efficiency"]
    stall_softness = preset["stall_softness"]
    lift_multiplier = preset["lift_multiplier"]
    thrust_type = preset["thrust_type"]

    mass = weight
    W = mass * G

    k, AR = calculate_induced_drag_factor(wing_area, chord, oswald_efficiency)

    velocity = np.linspace(0.1, 50, 150)                    # ground speed
    relative_velocity = np.clip(velocity + wind, 0.1, None)  # airspeed

    thrust_curve = calculate_thrust_curve(thrust, relative_velocity, pitch_speed, thrust_type)

    cl_curve = []
    drag_curve = []
    re_curve = []

    for v in relative_velocity:
        Re_local = (RHO * v * chord) / MU
        re_curve.append(Re_local)
        re_factor = np.clip((Re_local / 100000) ** 0.15, 0.6, 1.15)

        cl_eff = calculate_cl(aoa, stall_aoa, re_factor, stall_softness)
        cl_eff *= lift_multiplier
        cl_eff = np.clip(cl_eff, 0, 2.0)

        cd_local = (cd_base * parasitic_multiplier + k * cl_eff ** 2) * (1 / re_factor)
        drag_force = 0.5 * RHO * v ** 2 * wing_area * cd_local

        cl_curve.append(cl_eff)
        drag_curve.append(drag_force)

    cl_curve = np.array(cl_curve)
    drag = np.array(drag_curve)
    re_curve = np.array(re_curve)

    lift = 0.5 * RHO * relative_velocity ** 2 * wing_area * cl_curve

    stall_speed, cl_max, Re_stall = calculate_cl_max_and_stall_speed(
        weight, wing_area, chord, stall_aoa, stall_softness, lift_multiplier
    )

    # Max speed: where thrust meets drag. Report BOTH airspeed (what the physics uses) and groundspeed (what you'd actually see the plane do over the field) these differ whenever wind != 0.
    valid = np.where(thrust_curve >= drag)[0]
    if len(valid) > 0:
        idx = valid[-1]
        max_airspeed = relative_velocity[idx]
        max_groundspeed = velocity[idx]
    else:
        max_airspeed = 0.0
        max_groundspeed = 0.0

    wing_loading = mass / wing_area
    ld_ratio_curve = lift / np.maximum(drag, 1e-6)
    ld_ratio = np.max(ld_ratio_curve)

    return {
        "lift": lift,
        "drag": drag,
        "velocity": velocity,
        "relative_velocity": relative_velocity,
        "stall_speed": stall_speed,
        "cl_max": cl_max,
        "max_airspeed": max_airspeed,
        "max_groundspeed": max_groundspeed,
        "ld_ratio": ld_ratio,
        "ld_ratio_curve": ld_ratio_curve,
        "thrust_curve": thrust_curve,
        "wing_loading": wing_loading,
        "re_curve": re_curve,
        "aspect_ratio": AR,
        "induced_drag_factor": k,
        "oswald_efficiency": oswald_efficiency,
        "parasitic_multiplier": parasitic_multiplier,
    }