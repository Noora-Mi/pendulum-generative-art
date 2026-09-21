# ============================================================
# Section 1: Imports
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt

from double_pendulum_model import (
    simulate_double_pendulum_degrees
)

from double_pendulum_visualization import (
    create_physical_trajectory_figure
)


# ============================================================
# Section 2: Define the test parameters
# ============================================================

theta1_degrees = 120.0
theta2_degrees = -70.0

omega1_initial = 0.0
omega2_initial = 0.0

m1 = 1.0
m2 = 1.0

L1 = 1.0
L2 = 1.0

total_time = 20.0


# ============================================================
# Section 3: Run the simulation
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


# ============================================================
# Section 4: Define the output location
# ============================================================

script_directory = Path(__file__).resolve().parent

figure_directory = (
    script_directory
    / "figures"
)

figure_directory.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    figure_directory
    / "new_double_pendulum_trajectory.png"
)


# ============================================================
# Section 5: Create and save the figure
# ============================================================

figure, axis = create_physical_trajectory_figure(
    results=results,
    save_path=output_path
)

print("Visualization created successfully.")
print("Saved figure:")
print(output_path)

print("\nMaximum relative energy error:")
print(
    results["maximum_relative_energy_error"]
)


# ============================================================
# Section 6: Display the figure
# ============================================================

plt.show()