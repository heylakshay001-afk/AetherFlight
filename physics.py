# ========================= physics_test.py =========================

import numpy as np

RHO = 1.225
MU = 1.81e-5
G = 9.81

AIRCRAFT_PRESETS = {
    "Trainer": {
        "drag_multiplier": 1.0,
        "stall_softness": 1.3,
        "lift_multiplier": 1.1,
        "thrust_type": "prop",
        "stability": 1.3
    },
    "Flying Wing": {
        "drag_multiplier": 1.2,
        "stall_softness": 0.8,
        "lift_multiplier": 0.95,
        "thrust_type": "prop",
        "stability": 0.7
    },
    "EDF Jet": {
        "drag_multiplier": 0.85,
        "stall_softness": 0.7,
        "lift_multiplier": 0.8,
        "thrust_type": "edf",
        "stability": 0.8
    },
    "Glider": {
        "drag_multiplier": 0.7,
        "stall_softness": 1.5,
        "lift_multiplier": 1.2,
        "thrust_type": "prop",
        "stability": 1.5
    },
    "Aerobatic": {
        "drag_multiplier": 1.1,
        "stall_softness": 1.0,
        "lift_multiplier": 1.0,
        "thrust_type": "prop",
        "stability": 0.9
    },
    "Warbird": {
        "drag_multiplier": 1.15,
        "stall_softness": 0.9,
        "lift_multiplier": 0.95,
        "thrust_type": "prop",
        "stability": 0.85
    }
}


def get_drag_factor(plane_type):
    if plane_type == "Light RC":
        return 0.03
    elif plane_type == "Sport RC":
        return 0.06
    elif plane_type == "Heavy RC":
        return 0.10
    else:
        return 0.08


def calculate_cl(aoa, stall_aoa, re_factor, stall_softness):
    if aoa <= stall_aoa:
        cl = 2 * np.pi * np.radians(aoa)
    else:
        cl = (
            2 * np.pi * np.radians(stall_aoa)
        ) * np.exp(
            -(aoa - stall_aoa) / (5 * stall_softness)
        )
    cl *= re_factor
    return np.clip(cl, 0, 2.0)


def calculate_aerodynamics(
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
    preset = AIRCRAFT_PRESETS[aircraft_class]
    drag_multiplier = preset["drag_multiplier"]
    stall_softness = preset["stall_softness"]
    lift_multiplier = preset["lift_multiplier"]
    thrust_type = preset["thrust_type"]

    mass = weight
    W = mass * G
    k = get_drag_factor(plane_type)

    velocity = np.linspace(0.1, 50, 150)
    relative_velocity = np.clip(velocity + wind, 0.1, None)

    # BUG FIX 1 & 2:
    # - Was using undefined variable v_air — should be relative_velocity.
    # - Was named thrust_dynamic but referenced as thrust_curve later — unified to thrust_curve.
    if thrust_type == "edf":
        thrust_curve = thrust * (
            0.75 + 0.25 * np.exp(-relative_velocity / max_prop_speed)
        )
    else:
        thrust_curve = thrust * np.exp(-relative_velocity / max_prop_speed)

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

        cd_local = (cd_base + k * cl_eff ** 2) * drag_multiplier
        cd_local *= (1 / re_factor)

        drag_force = 0.5 * RHO * v ** 2 * wing_area * cd_local

        cl_curve.append(cl_eff)
        drag_curve.append(drag_force)

    cl_curve = np.array(cl_curve)
    drag = np.array(drag_curve)
    re_curve = np.array(re_curve)

    lift = 0.5 * RHO * relative_velocity ** 2 * wing_area * cl_curve

    cl_max = np.max(cl_curve)
    stall_speed = np.sqrt((2 * W) / (RHO * wing_area * cl_max))

    # Find max velocity: where thrust meets drag.
    # BUG FIX 2 (cont): thrust_curve is now correctly defined above.
    valid = np.where(thrust_curve >= drag)[0]
    if len(valid) > 0:
        max_velocity = relative_velocity[valid[-1]]
    else:
        max_velocity = 0.0

    wing_loading = mass / wing_area
    ld_ratio_curve = lift / np.maximum(drag, 1e-6)
    ld_ratio = np.max(ld_ratio_curve)

    return {
        "lift": lift,
        "drag": drag,
        "velocity": velocity,
        "relative_velocity": relative_velocity,
        "stall_speed": stall_speed,
        "max_velocity": max_velocity,
        "ld_ratio": ld_ratio,
        "ld_ratio_curve": ld_ratio_curve,
        "thrust_curve": thrust_curve,
        "wing_loading": wing_loading,
        "re_curve": re_curve
    }