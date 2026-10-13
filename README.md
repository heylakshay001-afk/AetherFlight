# ✈️ AetherFlight

AetherFlight is a flight-simulation and analysis tool for small RC aircraft. Enter an airframe's weight, wing geometry and thrust, and it computes stall speed, max speed, lift-to-drag ratio and takeoff performance from standard aerodynamic equations, with Reynolds number corrections for small-scale flight. The interface is built in Streamlit, with interactive Plotly charts.

It doesn't just tell you *whether* an aircraft takes off. When a configuration fails, it tells you *why*: not enough thrust to overcome rolling friction, or not enough top speed to generate lift.

## Features

- **Flight performance:** stall speed (at the critical angle of attack), max speed, wing loading, aspect ratio, CLmax and max L/D
- **Takeoff simulation:** ground-roll distance, liftoff speed and time, with velocity, acceleration and distance plotted over time
- **Failure diagnostics:** a specific explanation when takeoff isn't possible, plus a progress bar showing how close the aircraft gets to the speed it needs when top speed is the limit
- **Wind:** headwind and tailwind, with max speed reported as groundspeed and all aerodynamics computed on airspeed
- **Six aircraft classes:** Trainer, Flying Wing, EDF Jet, Glider, Aerobatic and Warbird, each with its own drag, lift and efficiency characteristics
- **Prop and EDF thrust models:** props lose thrust linearly up to their pitch speed, while EDFs hold thrust much better
- **Unit toggle:** switch between km/h and m/s; every speed metric and chart follows it
- **Collapsible sidebar sections** for quickly folding away the parameters you aren't changing

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.11 or newer (needed by the pinned NumPy 2.x release).

## Project structure

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI: sidebar inputs, metrics, status messages, tabs. Contains no aerodynamic math. |
| `physics.py` | All aerodynamic formulas: lift, drag, CL, Reynolds correction, stall speed, thrust curves, aircraft presets. |
| `simulation.py` | Takeoff time-stepping (RK2 integration), liftoff detection and failure diagnostics. Calls `physics.py` for every force. |
| `graphs.py` | Builds the five Plotly figures from already-computed results. Contains no physics. |

Each module has one job, and the aerodynamic equations live in exactly one place (`physics.compute_lift_drag`). The static charts and the takeoff simulation therefore can't disagree with each other.

All internal values are SI units (m, kg, N, m/s). Conversion to km/h happens only at the display layer.

## The model

**Lift and drag**

```
L  = ½ · ρ · v² · S · CL
D  = ½ · ρ · v² · S · CD
CD = CD0 · (parasitic multiplier) + CL² / (π · AR · e)
```

- **Aspect ratio** is estimated as `AR = wing_area / chord²` (a rectangular-wing approximation).
- **Oswald efficiency (e)** and the parasitic drag multiplier come from the selected aircraft class. For example, a flying wing has less parasitic drag (no fuselage or tail) but worse span efficiency than a glider.
- **CL vs angle of attack** follows thin-airfoil theory (2π per radian) up to the critical angle, then falls off exponentially.
- **Reynolds correction:** CL and CD are scaled by a clamped `(Re / 100,000)^0.15` factor.

**Stall speed** is computed from CLmax at the *critical* angle of attack, not whatever angle you've dialed in. It uses one Reynolds-refinement pass, since Re depends on the stall speed being solved for.

**Takeoff** is integrated with Heun's method (RK2) at a 0.1 s time step, using trapezoidal distance integration. The aircraft lifts off when lift reaches 1.05 × weight; rolling friction is 3% of weight. The run stops after 60 s, which is reported as a timeout.

## Assumptions and limitations

AetherFlight is a first-order performance estimator and an educational tool. It has not been validated against flight-test data, so treat its numbers as estimates rather than design guarantees.

- Sea-level ISA conditions (constant air density, viscosity)
- Rectangular-wing approximation for aspect ratio; tapered and swept wings aren't modelled
- Angle of attack is held constant during the takeoff roll (no rotation)
- No ground effect, climb, turns or stability/trim analysis
- Class presets (Oswald efficiency, lift multipliers, etc.) are reasonable estimates, not measured values

## Self-test

The simulation module has a built-in sanity check covering a normal takeoff, both failure modes, a headwind comparison and input validation:

```bash
python simulation.py
```

## Changelog

**1.0.0**
- True CLmax and stall speed evaluated at the critical angle of attack
- Induced drag derived from aspect ratio and per-class Oswald efficiency (replacing a weight-category lookup)
- Linear-to-zero prop thrust model and a separate EDF model
- RK2 takeoff integration with specific failure diagnostics
- Wind handled correctly: airspeed for physics, groundspeed for reported max speed
- Single shared lift/drag implementation; all magic numbers replaced with named constants
- Redesigned UI: three tabs, collapsible sidebar sections, tooltips, consistent units
- Charts consolidated from six to five, with consistent styling, hover info and annotations
