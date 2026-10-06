from pathlib import Path

import matplotlib.pyplot as plt

from forced_damped_pendulum_model import (
    simulate_forced_damped_pendulum,
)
from forced_damped_pendulum_visualization import (
    create_forced_damped_pendulum_animation,
)


results = simulate_forced_damped_pendulum(
    theta_initial=0.2,
    omega_initial=0.0,
    damping=0.5,
    drive_amplitude=1.2,
    drive_frequency=2.0 / 3.0,
    total_time=12.0,
    step_size=0.01,
)

animation, figure = create_forced_damped_pendulum_animation(
    results=results,
    frame_skip=6,
    color_map_name="turbo",
    minimum_line_width=0.6,
    maximum_line_width=4.0,
    minimum_alpha=0.20,
    maximum_alpha=1.00,
)

output_directory = Path(__file__).resolve().parent / "animations"
output_directory.mkdir(parents=True, exist_ok=True)

output_path = (
    output_directory
    / "forced_damped_pendulum_animation_test.gif"
)

print("Rendering Part II animation...")

animation.save(
    output_path,
    writer="pillow",
    fps=30,
    dpi=90,
)

plt.close(figure)

print("Animation saved to:")
print(output_path)
print("Part II animation test passed.")
