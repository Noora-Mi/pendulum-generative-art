# ============================================================
# Section 1: Imports
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt

from single_pendulum_model import (
    simulate_single_pendulum_degrees
)

from single_pendulum_visualization import (
    create_single_pendulum_animation
)


# ============================================================
# Section 2: Run the single-pendulum simulation
# ============================================================

results = simulate_single_pendulum_degrees(
    theta_degrees=85.94,
    omega_initial=0.0,
    length=1.0,
    mass=1.0,
    damping=0.08,
    total_time=8.0
)

print("Single-pendulum simulation completed.")

print(
    "Number of numerical points:",
    len(results["time"])
)

print(
    "Initial total energy:",
    results["initial_total_energy"],
    "J"
)

print(
    "Final total energy:",
    results["final_total_energy"],
    "J"
)

print(
    "Final relative energy loss:",
    results["final_relative_energy_loss"]
)


# ============================================================
# Section 3: Create the synchronized animation
# ============================================================

animation, figure = create_single_pendulum_animation(
    results=results,
    frame_skip=3,
    color_map_name="turbo"
)


# ============================================================
# Section 4: Define the output location
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
    / "single_pendulum_animation_test.gif"
)


# ============================================================
# Section 5: Save the GIF
# ============================================================

print("\nSaving the single-pendulum animation...")
print("This may take a few minutes.")

animation.save(
    output_path,
    writer="pillow",
    fps=30,
    dpi=100
)

print("\nAnimation saved successfully:")
print(output_path)


# ============================================================
# Section 6: Close the figure
# ============================================================

plt.show()