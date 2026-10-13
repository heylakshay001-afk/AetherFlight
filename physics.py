"""
physics.py — AetherFlight Aerodynamics Engine
==============================================
Pure aerodynamic math: lift, drag, stall, thrust-curve, and induced-drag
calculations for a single aircraft configuration at a single instant.

Module boundary: this file owns EVERY force/coefficient formula in
AetherFlight. It has no knowledge of time-stepping (that's simulation.py),
Streamlit, or Plotly (that's app.py / graphs.py) — it just turns aircraft
parameters into forces.
"""

import numpy as np

# ============================================================
# Physical constants (ISA standard day, sea level)
# ============================================================
RHO = 1.225     # air density, kg/m^3
MU = 1.81e-5    # dynamic viscosity of air, Pa*s
G = 9.81        # gravitational acceleration, m/s^2

# ============================================================
# Model tuning constants
# (named so the reasoning for each number lives next to it,
# instead of a bare literal scattered through the formulas)
# ============================================================
RE_REFERENCE = 100_000.0     # reference Reynolds number for the Re-correction curve
RE_FACTOR_EXPONENT = 0.15    # how strongly CL/CD respond to Re deviating from RE_REFERENCE
RE_FACTOR_MIN = 0.6          # floor on the Re-correction factor (small/slow wings)
RE_FACTOR_MAX = 1.15         # ceiling on the Re-correction factor (diminishing returns at high Re)

CL_MIN = 0.05                # sanity floor for CL (keeps 1/CL-ish terms well-behaved)
CL_MAX = 2.0                 # sanity ceiling for CL (well above anything a hobby airfoil reaches)

POST_STALL_DECAY_RATE = 5.0  # base divisor for how sharply CL falls off past the critical AoA;
                              # scaled further per-class by that class's stall_softness

EDF_THRUST_DECAY_RATE = 0.2  # fraction of static thrust an EDF loses per unit of (airspeed/pitch_speed)
EDF_THRUST_FLOOR = 0.5       # EDFs never drop below this fraction of static thrust (unlike props,
                              # which fall all the way to zero at their pitch speed)

MIN_DRAG_FOR_RATIO = 1e-6    # guards L/D divisions against near-zero drag at very low speed

# Speed sweep used to build the lift/drag/thrust curves shown in the UI
SWEEP_MIN_SPEED = 0.1        # m/s — not 0, to avoid a divide-by-zero in Reynolds number
SWEEP_MAX_SPEED = 50.0       # m/s — ~180 km/h, comfortably above any RC aircraft modeled here
SWEEP_POINTS = 150

# ============================================================
# Aircraft class presets
# ============================================================
# parasitic_multiplier : scales form/interference drag (fuselage, struts, etc.)
# oswald_efficiency    : span-wise lift-distribution efficiency; drives induced
#                        drag together with aspect ratio (lower = worse)
# stall_softness        : how gently CL falls off past the critical AoA
# lift_multiplier       : crude stand-in for airfoil choice (higher-lift vs thin/fast sections)
# thrust_type           : "prop" (linear thrust falloff) or "edf" (gentle falloff, floored)
AIRCRAFT_PRESETS = {
    "Trainer":     {"parasitic_multiplier": 1.00, "oswald_efficiency": 0.80, "stall_softness": 1.3, "lift_multiplier": 1.10, "thrust_type": "prop"},
    "Flying Wing": {"parasitic_multiplier": 0.90, "oswald_efficiency": 0.65, "stall_softness": 0.8, "lift_multiplier": 0.95, "thrust_type": "prop"},
    "EDF Jet":     {"parasitic_multiplier": 0.85, "oswald_efficiency": 0.75, "stall_softness": 0.7, "lift_multiplier": 0.80, "thrust_type": "edf"},
    "Glider":      {"parasitic_multiplier": 0.70, "oswald_efficiency": 0.90, "stall_softness": 1.5, "lift_multiplier": 1.20, "thrust_type": "prop"},
    "Aerobatic":   {"parasitic_multiplier": 1.10, "oswald_efficiency": 0.70, "stall_softness": 1.0, "lift_multiplier": 1.00, "thrust_type": "prop"},
    "Warbird":     {"parasitic_multiplier": 1.15, "oswald_efficiency": 0.75, "stall_softness": 0.9, "lift_multiplier": 0.95, "thrust_type": "prop"},
}
# Flying Wing has LOWER parasitic drag (no fuselage/tail) but WORSE oswald
# efficiency (no tail to trim out wingtip twist / poor span loading) — that's
# the actual aerodynamic story with tailless aircraft, not a flat "draggier"
# penalty with no stated reason.


def calculate_aspect_ratio(wing_area, chord):
    """Rectangular-wing approximation: span = wing_area / chord, so
    AR = span / chord = wing_area / chord^2. Real AR for tapered/swept
    wings differs, but this is a reasonable stand-in given the sliders
    the UI actually exposes (area + chord, not span)."""
    return wing_area / chord ** 2


def calculate_induced_drag_factor(wing_area, chord, oswald_efficiency):
    """k in CD = CD0 + k*CL^2, from classic finite-wing theory:
    k = 1 / (pi * AR * e). Returns (k, aspect_ratio)."""
    aspect_ratio = calculate_aspect_ratio(wing_area, chord)
    k = 1.0 / (np.pi * aspect_ratio * oswald_efficiency)
    return k, aspect_ratio


def reynolds_number(airspeed, chord):
    """Re = rho * v * c / mu — characteristic Reynolds number at this
    airspeed and chord length."""
    return (RHO * airspeed * chord) / MU


def reynolds_correction_factor(re):
    """Small, clamped correction applied to CL (and inversely to CD) to
    reflect that both improve somewhat as Reynolds number rises."""
    return np.clip((re / RE_REFERENCE) ** RE_FACTOR_EXPONENT, RE_FACTOR_MIN, RE_FACTOR_MAX)


def calculate_cl(aoa, stall_aoa, re_factor, stall_softness):
    """CL vs AoA: thin-airfoil slope (2*pi per radian) below the critical
    angle, exponential falloff past it."""
    if aoa <= stall_aoa:
        cl = 2 * np.pi * np.radians(aoa)
    else:
        cl = (2 * np.pi * np.radians(stall_aoa)) * np.exp(
            -(aoa - stall_aoa) / (POST_STALL_DECAY_RATE * stall_softness)
        )
    cl *= re_factor
    return np.clip(cl, 0, CL_MAX)


def calculate_cl_max_and_stall_speed(weight, wing_area, chord, stall_aoa,
                                      stall_softness, lift_multiplier):
    """The wing's actual CLmax, evaluated at the CRITICAL AoA — not
    whatever AoA the pilot currently has dialed in. Stall speed is a
    property of the wing, not of how it's currently trimmed.

    Re affects CLmax slightly, but Re itself depends on the stall speed
    we're solving for, so this does one predictor/refine pass. The
    dependence is weak (Re enters through a 0.15 power), so one pass
    converges essentially exactly.

    Returns (stall_speed_ms, cl_max, reynolds_at_stall).
    """
    weight_n = weight * G

    # Pass 1: ignore Re correction to get a first estimate.
    cl_max_guess = calculate_cl(stall_aoa, stall_aoa, 1.0, stall_softness) * lift_multiplier
    cl_max_guess = np.clip(cl_max_guess, CL_MIN, CL_MAX)
    v_stall_guess = np.sqrt((2 * weight_n) / (RHO * wing_area * cl_max_guess))

    # Pass 2: refine using Re at that estimated stall speed.
    re_stall = reynolds_number(v_stall_guess, chord)
    re_factor_stall = reynolds_correction_factor(re_stall)
    cl_max = calculate_cl(stall_aoa, stall_aoa, re_factor_stall, stall_softness) * lift_multiplier
    cl_max = np.clip(cl_max, CL_MIN, CL_MAX)
    stall_speed = np.sqrt((2 * weight_n) / (RHO * wing_area * cl_max))

    return stall_speed, cl_max, re_stall


def calculate_thrust_curve(thrust, airspeed, pitch_speed, thrust_type):
    """Static thrust vs airspeed.
    - Fixed-pitch prop: falls off ~linearly to zero at the theoretical
      pitch speed (a standard simplified static-thrust model for RC props).
    - EDF/jet: falls off gently, floored at EDF_THRUST_FLOOR of static
      thrust — jets hold their thrust far better across the speed range
      than props do.
    `airspeed` may be a scalar or a numpy array."""
    if thrust_type == "edf":
        return thrust * np.clip(1.0 - EDF_THRUST_DECAY_RATE * (airspeed / pitch_speed), EDF_THRUST_FLOOR, 1.0)
    return thrust * np.clip(1.0 - (airspeed / pitch_speed), 0.0, 1.0)


def compute_lift_drag(v_air, wing_area, chord, cd_base, aoa, stall_aoa,
                       parasitic_multiplier, induced_drag_k, lift_multiplier,
                       stall_softness):
    """The single source of truth for "what are lift and drag at this
    airspeed, for this aircraft". Used both by the speed-sweep in
    calculate_aerodynamics() below AND by simulation.py's per-timestep
    takeoff integration, so the static charts and the takeoff simulation
    can never silently disagree with each other.

    Returns (lift_n, drag_n, cl_eff, cd_local, re_factor).
    """
    re = reynolds_number(v_air, chord)
    re_factor = reynolds_correction_factor(re)

    cl_eff = calculate_cl(aoa, stall_aoa, re_factor, stall_softness)
    cl_eff *= lift_multiplier
    cl_eff = np.clip(cl_eff, 0, CL_MAX)

    cd_local = (cd_base * parasitic_multiplier + induced_drag_k * cl_eff ** 2) * (1 / re_factor)

    lift = 0.5 * RHO * v_air ** 2 * wing_area * cl_eff
    drag = 0.5 * RHO * v_air ** 2 * wing_area * cd_local

    return lift, drag, cl_eff, cd_local, re_factor


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
    """Build the full speed-sweep (lift/drag/thrust vs airspeed) for the
    current configuration, plus the headline numbers the UI displays
    (stall speed, max speed, L/D ratio, wing loading, aspect ratio, ...).

    All inputs are SI units (kg, m^2, N, m, m/s). Returns a dict — see the
    keys in the `return` statement below for the full contract.
    """
    preset = AIRCRAFT_PRESETS[aircraft_class]
    parasitic_multiplier = preset["parasitic_multiplier"]
    oswald_efficiency = preset["oswald_efficiency"]
    stall_softness = preset["stall_softness"]
    lift_multiplier = preset["lift_multiplier"]
    thrust_type = preset["thrust_type"]

    mass = weight
    induced_drag_k, aspect_ratio = calculate_induced_drag_factor(wing_area, chord, oswald_efficiency)

    ground_speed = np.linspace(SWEEP_MIN_SPEED, SWEEP_MAX_SPEED, SWEEP_POINTS)
    airspeed = np.clip(ground_speed + wind, SWEEP_MIN_SPEED, None)

    thrust_curve = calculate_thrust_curve(thrust, airspeed, pitch_speed, thrust_type)

    lift_curve = np.zeros_like(airspeed)
    drag_curve = np.zeros_like(airspeed)
    re_curve = np.zeros_like(airspeed)

    for i, v in enumerate(airspeed):
        lift, drag, _, _, _ = compute_lift_drag(
            v, wing_area, chord, cd_base, aoa, stall_aoa,
            parasitic_multiplier, induced_drag_k, lift_multiplier, stall_softness
        )
        lift_curve[i] = lift
        drag_curve[i] = drag
        re_curve[i] = reynolds_number(v, chord)

    stall_speed, cl_max, _ = calculate_cl_max_and_stall_speed(
        weight, wing_area, chord, stall_aoa, stall_softness, lift_multiplier
    )

    # Max speed: where thrust meets drag. Report BOTH airspeed (what the
    # physics uses) and groundspeed (what you'd actually see the plane do
    # over the field) — these differ whenever wind != 0.
    reachable = np.where(thrust_curve >= drag_curve)[0]
    if len(reachable) > 0:
        idx = reachable[-1]
        max_airspeed = airspeed[idx]
        max_groundspeed = ground_speed[idx]
    else:
        max_airspeed = 0.0
        max_groundspeed = 0.0

    wing_loading = mass / wing_area
    ld_ratio_curve = lift_curve / np.maximum(drag_curve, MIN_DRAG_FOR_RATIO)
    ld_ratio = np.max(ld_ratio_curve)

    return {
        "lift": lift_curve,                            # N, over the airspeed sweep
        "drag": drag_curve,                             # N, over the airspeed sweep
        "ground_speed": ground_speed,                   # m/s sweep (no wind)
        "airspeed": airspeed,                           # m/s sweep (wind-corrected)
        "stall_speed": stall_speed,                     # m/s
        "cl_max": cl_max,
        "max_airspeed": max_airspeed,                   # m/s
        "max_groundspeed": max_groundspeed,              # m/s
        "ld_ratio": ld_ratio,
        "ld_ratio_curve": ld_ratio_curve,
        "thrust_curve": thrust_curve,                    # N, over the airspeed sweep
        "wing_loading": wing_loading,                    # kg/m^2
        "re_curve": re_curve,
        "aspect_ratio": aspect_ratio,
        "induced_drag_factor": induced_drag_k,
        "oswald_efficiency": oswald_efficiency,
        "parasitic_multiplier": parasitic_multiplier,
    }
