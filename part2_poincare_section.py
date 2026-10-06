import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from forced_damped_pendulum_model import (
    simulate_forced_damped_pendulum
)


# ============================================================
# Section 1: Output directory
# ============================================================

project_directory = Path(__file__).resolve().parent

figure_directory = (
    project_directory / "figures"
)

figure_directory.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Section 2: Model parameters
# ============================================================

theta_initial = 0.2
omega_initial = 0.0

damping = 0.5
drive_amplitude = 1.2
drive_frequency = 2.0 / 3.0

step_size = 0.01

drive_period = (
    2.0 * np.pi / drive_frequency
)

# Ignore the first 100 driving periods because they contain
# transient behaviour.
discarded_cycles = 100

# Record the following 300 driving periods.
recorded_cycles = 300

total_cycles = (
    discarded_cycles
    + recorded_cycles
)

total_time = (
    total_cycles * drive_period
)


# ============================================================
# Section 3: Long-time simulation
# ============================================================

print(
    "Poincare-section experiment"
)

print(
    "-----------------------------------"
)

print(
    "Driving period:",
    drive_period
)

print(
    "Discarded driving cycles:",
    discarded_cycles
)

print(
    "Recorded driving cycles:",
    recorded_cycles
)

print(
    "Total simulation time:",
    total_time
)

print()
print(
    "Running long-time simulation..."
)

results = simulate_forced_damped_pendulum(
    theta_initial=theta_initial,
    omega_initial=omega_initial,
    damping=damping,
    drive_amplitude=drive_amplitude,
    drive_frequency=drive_frequency,
    total_time=total_time,
    step_size=step_size
)

print(
    "Long-time simulation completed."
)

print(
    "Number of numerical time points:",
    len(results["time"])
)


# ============================================================
# Section 4: Exact Poincare sampling times
# ============================================================

# Start sampling after the transient cycles have been removed.
sample_cycle_numbers = np.arange(
    discarded_cycles + 1,
    total_cycles + 1
)

poincare_times = (
    sample_cycle_numbers * drive_period
)

# The numerical time grid does not generally land exactly on
# every multiple of the driving period. Linear interpolation
# estimates theta and omega at the exact sampling times.

poincare_theta_unwrapped = np.interp(
    poincare_times,
    results["time"],
    results["theta"]
)

poincare_omega = np.interp(
    poincare_times,
    results["time"],
    results["omega"]
)

# Wrap the sampled angle into [-pi, pi).
poincare_theta = (
    poincare_theta_unwrapped + np.pi
) % (
    2.0 * np.pi
) - np.pi


# ============================================================
# Section 5: Verify the sampling phase
# ============================================================

sampled_drive_phase = (
    drive_frequency * poincare_times
) % (
    2.0 * np.pi
)

phase_distance_from_zero = np.minimum(
    sampled_drive_phase,
    2.0 * np.pi - sampled_drive_phase
)

maximum_phase_error = np.max(
    phase_distance_from_zero
)

print()
print(
    "Poincare sampling validation"
)

print(
    "-----------------------------------"
)

print(
    "Number of Poincare points:",
    len(poincare_theta)
)

print(
    "Maximum driving-phase error:",
    maximum_phase_error
)

print(
    "Poincare theta range:",
    np.min(poincare_theta),
    "to",
    np.max(poincare_theta)
)

print(
    "Poincare omega range:",
    np.min(poincare_omega),
    "to",
    np.max(poincare_omega)
)


# ============================================================
# Section 6: Select late-time continuous trajectory
# ============================================================

# Display only the final 30 driving periods in the continuous
# phase portrait so the figure is not overcrowded.

display_cycles = 30

display_start_time = (
    total_time
    - display_cycles * drive_period
)

display_mask = (
    results["time"] >= display_start_time
)

display_theta = (
    results["wrapped_theta"][display_mask]
)

display_omega = (
    results["omega"][display_mask]
)

# Break the continuous curve at the angle-wrapping boundary.
display_theta_plot = np.array(
    display_theta,
    copy=True
)

display_omega_plot = np.array(
    display_omega,
    copy=True
)

angle_jumps = (
    np.abs(
        np.diff(display_theta_plot)
    )
    > np.pi
)

display_theta_plot[1:][angle_jumps] = np.nan
display_omega_plot[1:][angle_jumps] = np.nan


# ============================================================
# Section 7: Continuous phase portrait and Poincare section
# ============================================================

figure, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(15, 6)
)

figure.suptitle(
    "Continuous Dynamics and Poincaré Section",
    fontsize=20,
    y=0.98
)


# ------------------------------------------------------------
# Left panel: continuous late-time phase portrait
# ------------------------------------------------------------

axes[0].plot(
    display_theta_plot,
    display_omega_plot,
    color="#8b5cf6",
    linewidth=0.8,
    alpha=0.75
)

axes[0].set_title(
    "Late-Time Continuous Phase Portrait"
)

axes[0].set_xlabel(
    r"Wrapped angle $\theta$"
)

axes[0].set_ylabel(
    r"Angular velocity $\omega$"
)

axes[0].set_xlim(
    -np.pi,
    np.pi
)

axes[0].grid(
    alpha=0.25
)


# ------------------------------------------------------------
# Right panel: Poincare section
# ------------------------------------------------------------

point_colors = np.arange(
    len(poincare_theta)
)

poincare_scatter = axes[1].scatter(
    poincare_theta,
    poincare_omega,
    c=point_colors,
    cmap="plasma",
    s=16,
    alpha=0.8,
    edgecolors="none"
)

axes[1].set_title(
    "Poincaré Section"
)

axes[1].set_xlabel(
    r"Wrapped angle $\theta_n$"
)

axes[1].set_ylabel(
    r"Angular velocity $\omega_n$"
)

axes[1].set_xlim(
    -np.pi,
    np.pi
)

axes[1].grid(
    alpha=0.25
)

color_bar = figure.colorbar(
    poincare_scatter,
    ax=axes[1],
    pad=0.02
)

color_bar.set_label(
    "Sample order"
)


# ------------------------------------------------------------
# Shared angle ticks
# ------------------------------------------------------------

for axis in axes:

    axis.axhline(
        0.0,
        color="#64748b",
        linewidth=0.8,
        linestyle="--",
        alpha=0.7
    )

    axis.axvline(
        0.0,
        color="#64748b",
        linewidth=0.8,
        linestyle="--",
        alpha=0.7
    )

    axis.set_xticks(
        [
            -np.pi,
            -np.pi / 2.0,
            0.0,
            np.pi / 2.0,
            np.pi
        ]
    )

    axis.set_xticklabels(
        [
            r"$-\pi$",
            r"$-\pi/2$",
            r"$0$",
            r"$\pi/2$",
            r"$\pi$"
        ]
    )


figure.text(
    0.5,
    0.015,
    (
        r"$q=0.5$, "
        r"$F=1.2$, "
        r"$\Omega=2/3$; "
        "one sample per driving period"
    ),
    ha="center",
    fontsize=11
)

figure.tight_layout(
    rect=[0.0, 0.045, 1.0, 0.94]
)

poincare_figure_path = (
    figure_directory
    / "part2_poincare_section.png"
)

figure.savefig(
    poincare_figure_path,
    dpi=300,
    bbox_inches="tight"
)

print()
print(
    "Poincare-section figure saved to:"
)

print(
    poincare_figure_path
)

plt.show()