import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


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
# Section 2: Representative driving amplitudes
# ============================================================

drive_amplitudes = np.array(
    [
        1.0500,
        1.0700,
        1.0815,
        1.2000
    ],
    dtype=float
)

expected_labels = [
    "Period 1",
    "Period 2",
    "Period 4",
    "Chaotic regime"
]

damping = 0.5
drive_frequency = 2.0 / 3.0
theta_initial = 0.2
omega_initial = 0.0

drive_period = (
    2.0 * np.pi / drive_frequency
)

steps_per_period = 500

step_size = (
    drive_period / steps_per_period
)

discarded_cycles = 600
recorded_cycles = 128

total_cycles = (
    discarded_cycles
    + recorded_cycles
)

total_steps = (
    total_cycles
    * steps_per_period
)


# ============================================================
# Section 3: Initial states
# ============================================================

number_of_cases = len(
    drive_amplitudes
)

theta_values = np.full(
    number_of_cases,
    theta_initial,
    dtype=float
)

omega_values = np.full(
    number_of_cases,
    omega_initial,
    dtype=float
)

recorded_theta = np.zeros(
    (
        recorded_cycles,
        number_of_cases
    )
)

recorded_omega = np.zeros(
    (
        recorded_cycles,
        number_of_cases
    )
)


# ============================================================
# Section 4: Vectorized derivative function
# ============================================================

def calculate_derivatives(
    time,
    theta,
    omega
):

    theta_derivative = omega

    omega_derivative = (
        -np.sin(theta)
        - damping * omega
        + drive_amplitudes
        * np.cos(drive_frequency * time)
    )

    return (
        theta_derivative,
        omega_derivative
    )


# ============================================================
# Section 5: Long-time RK4 simulations
# ============================================================

print(
    "Representative periodic-orbit experiment"
)

print(
    "----------------------------------------"
)

print(
    "Driving amplitudes:",
    drive_amplitudes
)

print(
    "Discarded cycles:",
    discarded_cycles
)

print(
    "Recorded cycles:",
    recorded_cycles
)

print(
    "Numerical step size:",
    step_size
)

print()
print(
    "Running simulations..."
)

current_time = 0.0
record_index = 0

for step_index in range(total_steps):

    (
        k1_theta,
        k1_omega
    ) = calculate_derivatives(
        current_time,
        theta_values,
        omega_values
    )

    (
        k2_theta,
        k2_omega
    ) = calculate_derivatives(
        current_time + 0.5 * step_size,
        theta_values
        + 0.5 * step_size * k1_theta,
        omega_values
        + 0.5 * step_size * k1_omega
    )

    (
        k3_theta,
        k3_omega
    ) = calculate_derivatives(
        current_time + 0.5 * step_size,
        theta_values
        + 0.5 * step_size * k2_theta,
        omega_values
        + 0.5 * step_size * k2_omega
    )

    (
        k4_theta,
        k4_omega
    ) = calculate_derivatives(
        current_time + step_size,
        theta_values
        + step_size * k3_theta,
        omega_values
        + step_size * k3_omega
    )

    theta_values = theta_values + (
        step_size / 6.0
    ) * (
        k1_theta
        + 2.0 * k2_theta
        + 2.0 * k3_theta
        + k4_theta
    )

    omega_values = omega_values + (
        step_size / 6.0
    ) * (
        k1_omega
        + 2.0 * k2_omega
        + 2.0 * k3_omega
        + k4_omega
    )

    theta_values = (
        theta_values + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi

    current_time += step_size

    completed_step = step_index + 1

    if completed_step % steps_per_period == 0:

        completed_cycle = (
            completed_step
            // steps_per_period
        )

        if completed_cycle > discarded_cycles:

            recorded_theta[
                record_index
            ] = theta_values

            recorded_omega[
                record_index
            ] = omega_values

            record_index += 1


print(
    "Simulations completed."
)


# ============================================================
# Section 6: Estimate Poincare orbit period
# ============================================================

def estimate_orbit_period(
    theta_samples,
    omega_samples,
    tolerance=1e-5
):
    """
    Find the smallest tested period p for which the Poincare
    state approximately repeats after p driving periods.
    """

    candidate_periods = range(
    1,
    65
)

    for period in candidate_periods:

        theta_difference = (
            theta_samples[period:]
            - theta_samples[:-period]
            + np.pi
        ) % (
            2.0 * np.pi
        ) - np.pi

        omega_difference = (
            omega_samples[period:]
            - omega_samples[:-period]
        )

        state_distance = np.sqrt(
            theta_difference**2
            + omega_difference**2
        )

        if np.max(state_distance) < tolerance:

            return period

    return None


estimated_periods = []

print()
print(
    "Estimated Poincare periods"
)

print(
    "----------------------------------------"
)

for case_index, drive_amplitude in enumerate(
    drive_amplitudes
):

    estimated_period = estimate_orbit_period(
        recorded_theta[:, case_index],
        recorded_omega[:, case_index]
    )

    estimated_periods.append(
        estimated_period
    )

    print(
        "F =",
        drive_amplitude,
        "| estimated period =",
        estimated_period
    )


# ============================================================
# Section 7: Poincare-point comparison figure
# ============================================================

figure, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(13, 10),
    sharex=True,
    sharey=True
)

figure.suptitle(
    "Period Doubling in the Forced and Damped Pendulum",
    fontsize=20,
    y=0.98
)

axes = axes.flatten()

color_maps = [
    "viridis",
    "plasma",
    "magma",
    "turbo"
]

for case_index, axis in enumerate(axes):

    point_order = np.arange(
        recorded_cycles
    )

    scatter = axis.scatter(
        recorded_theta[:, case_index],
        recorded_omega[:, case_index],
        c=point_order,
        cmap=color_maps[case_index],
        s=35,
        alpha=0.75,
        edgecolors="none"
    )

    estimated_period = (
        estimated_periods[case_index]
    )

    if estimated_period is None:

        period_text = (
            "No finite period detected"
        )

    else:

        period_text = (
            f"Detected period: {estimated_period}"
        )

    axis.set_title(
        (
            f"$F={drive_amplitudes[case_index]:.4f}$"
            "\n"
            f"{expected_labels[case_index]}"
        ),
        fontsize=13
    )

    axis.text(
        0.04,
        0.94,
        period_text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "#cbd5e1"
        }
    )

    axis.axhline(
        0.0,
        color="#64748b",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7
    )

    axis.axvline(
        0.0,
        color="#64748b",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7
    )

    axis.set_xlim(
        -np.pi,
        np.pi
    )

    axis.set_ylim(
        -0.6,
        2.2
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

    axis.grid(
        alpha=0.22
    )


axes[2].set_xlabel(
    r"Sampled angle $\theta_n$"
)

axes[3].set_xlabel(
    r"Sampled angle $\theta_n$"
)

axes[0].set_ylabel(
    r"Sampled angular velocity $\omega_n$"
)

axes[2].set_ylabel(
    r"Sampled angular velocity $\omega_n$"
)

figure.text(
    0.5,
    0.015,
    (
        r"$q=0.5$, "
        r"$\Omega=2/3$; "
        "one Poincaré sample per driving period"
    ),
    ha="center",
    fontsize=11
)

figure.tight_layout(
    rect=[0.0, 0.04, 1.0, 0.95]
)

periodic_orbits_path = (
    figure_directory
    / "part2_periodic_orbits.png"
)

figure.savefig(
    periodic_orbits_path,
    dpi=300,
    bbox_inches="tight"
)

print()
print(
    "Periodic-orbit comparison saved to:"
)

print(
    periodic_orbits_path
)

plt.show()