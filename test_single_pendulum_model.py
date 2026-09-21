# ============================================================
# Section 1: Imports
# ============================================================

import numpy as np

from single_pendulum_model import (
    simulate_single_pendulum,
    simulate_single_pendulum_degrees
)


# ============================================================
# Section 2: Undamped baseline test
# ============================================================

undamped_results = simulate_single_pendulum(
    theta_initial=1.5,
    omega_initial=0.0,
    length=1.0,
    mass=1.0,
    damping=0.0,
    total_time=13.0,
    step_size=0.01
)

print("Undamped single-pendulum test")
print("-----------------------------------")

print(
    "Number of time points:",
    len(undamped_results["time"])
)

print(
    "Initial state:",
    undamped_results["states"][0]
)

print(
    "Final state:",
    undamped_results["states"][-1]
)

print(
    "Initial position:",
    undamped_results["x"][0],
    undamped_results["y"][0]
)

print(
    "Initial total energy:",
    undamped_results["initial_total_energy"],
    "J"
)

print(
    "Maximum relative energy change:",
    undamped_results[
        "maximum_relative_energy_change"
    ]
)


# Check the number of numerical points.
assert len(undamped_results["time"]) == 1301

# Check the initial state.
assert np.allclose(
    undamped_results["states"][0],
    [1.5, 0.0]
)

# Check the initial physical coordinates.
assert np.isclose(
    undamped_results["x"][0],
    np.sin(1.5)
)

assert np.isclose(
    undamped_results["y"][0],
    -np.cos(1.5)
)

# The undamped RK4 simulation should conserve energy.
assert (
    undamped_results[
        "maximum_relative_energy_change"
    ]
    < 1e-6
)

print("\nUndamped test passed.")


# ============================================================
# Section 3: Damped website-interface test
# ============================================================

initial_angle_degrees = np.rad2deg(
    1.5
)

damped_results = simulate_single_pendulum_degrees(
    theta_degrees=initial_angle_degrees,
    omega_initial=0.0,
    length=1.0,
    mass=1.0,
    damping=0.08,
    total_time=13.0
)

print("\nDamped single-pendulum test")
print("-----------------------------------")

print(
    "Initial angle in degrees:",
    damped_results["theta_initial_degrees"]
)

print(
    "Initial angle in radians:",
    damped_results["theta"][0]
)

print(
    "Initial total energy:",
    damped_results["initial_total_energy"],
    "J"
)

print(
    "Final total energy:",
    damped_results["final_total_energy"],
    "J"
)

print(
    "Final relative energy loss:",
    damped_results["final_relative_energy_loss"]
)


# Check the degree-to-radian conversion.
assert np.isclose(
    damped_results["theta"][0],
    1.5
)

# Damping should reduce the mechanical energy.
assert (
    damped_results["final_total_energy"]
    < damped_results["initial_total_energy"]
)

# All states should remain finite.
assert np.all(
    np.isfinite(damped_results["states"])
)

print("\nDamped test passed.")
print("\nAll single-pendulum model tests passed.")