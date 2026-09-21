# ============================================================
# Section 1: Imports
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt

from double_pendulum_model import (
    simulate_double_pendulum_degrees
)

from double_pendulum_visualization import (
    create_double_pendulum_animation
)


# ============================================================
# Section 2: Animation parameters
# ============================================================

theta1_degrees = 120.0
theta2_degrees = -70.0

omega1_initial = 0.0
omega2_initial = 0.0

m1 = 1.0
m2 = 1.0

L1 = 1.0
L2 = 1.0

# Use 8 seconds for the first animation test.
total_time = 8.0


# ============================================================
# Section 3: Run the physical simulation
# ============================================================

results = simulate_double_pendulum_degrees(
    theta1_degrees=theta1_degrees,
    theta2_degrees=theta2_degrees,
    omega1_initial=omega1_initial,
    omega2_initial=omega2_initial,
    m1=m1,
    m2=m2,
    L1=L1,
    L2=L2,
    total_time=total_time
)

print("Simulation completed.")
print(
    "Number of numerical points:",
    len(results["time"])
)

print(
    "Maximum relative energy error:",
    results["maximum_relative_energy_error"]
)


# ============================================================
# Section 4: Create the animation
# ============================================================

animation, figure = create_double_pendulum_animation(
    results=results,

    # One animation frame for every 10 numerical points.
    frame_skip=10,

    # 40 milliseconds between displayed frames.
    interval=40
)


# ============================================================
# Section 5: Define the output location
# ============================================================

script_directory = Path(__file__).resolve().parent

animation_directory = (
    script_directory
    / "animations"
)

animation_directory.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    animation_directory
    / "double_pendulum_animation_test.gif"
)


# ============================================================
# Section 6: Save the animation
# ============================================================

print("\nSaving animation...")
print("This may take a few minutes.")

animation.save(
    output_path,
    writer="pillow",
    fps=25,
    dpi=100
)

print("\nAnimation saved successfully:")
print(output_path)


# ============================================================
# Section 7: Display the animation window
# ============================================================

plt.show()