# ============================================================
# Part II: Step-size convergence test
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from forced_damped_pendulum_model import (
    simulate_forced_damped_pendulum
)


# ============================================================
# Fixed model parameters
# ============================================================

theta_initial = 0.2
omega_initial = 0.0

damping = 0.5
drive_frequency = 2.0 / 3.0

drive_period = 2.0 * np.pi / drive_frequency

drive_amplitudes = [
    1.05,
    1.07,
    1.0815,
    1.20
]

steps_per_period_values = [
    250,
    500,
    1000
]

discarded_cycles = 600
recorded_cycles = 128

total_cycles = discarded_cycles + recorded_cycles
total_time = total_cycles * drive_period


# ============================================================
# Helper functions
# ============================================================

def wrap_angle(angle):
    """Return an angle in the interval [-pi, pi)."""

    return (
        (angle + np.pi) % (2.0 * np.pi)
    ) - np.pi


def sample_poincare_states(results):
    """
    Sample theta and omega once per driving period after
    discarding the transient cycles.
    """

    sample_cycles = np.arange(
        discarded_cycles + 1,
        total_cycles + 1
    )

    sample_times = sample_cycles * drive_period

    theta_samples = np.interp(
        sample_times,
        results["time"],
        results["theta"]
    )

    omega_samples = np.interp(
        sample_times,
        results["time"],
        results["omega"]
    )

    theta_samples = wrap_angle(theta_samples)

    return np.column_stack(
        (
            theta_samples,
            omega_samples
        )
    )


def phase_space_difference(first_states, second_states):
    """
    Calculate phase-space differences while respecting the
    periodic nature of the angle.
    """

    theta_difference = wrap_angle(
        first_states[:, 0]
        - second_states[:, 0]
    )

    omega_difference = (
        first_states[:, 1]
        - second_states[:, 1]
    )

    return np.sqrt(
        theta_difference**2
        + omega_difference**2
    )


def estimate_period(
    poincare_states,
    maximum_period=16,
    tolerance=1.0e-4
):
    """
    Estimate the smallest short period p satisfying
    P_(n+p) approximately equal to P_n.
    """

    for period in range(
        1,
        maximum_period + 1
    ):

        earlier_states = poincare_states[:-period]
        later_states = poincare_states[period:]

        distances = phase_space_difference(
            later_states,
            earlier_states
        )

        if np.max(distances) < tolerance:
            return period

    return None


# ============================================================
# Run the simulations
# ============================================================

all_results = {}

print("Part II step-size convergence test")
print("----------------------------------------")
print("Driving period:", drive_period)
print("Discarded cycles:", discarded_cycles)
print("Recorded cycles:", recorded_cycles)
print()

for drive_amplitude in drive_amplitudes:

    all_results[drive_amplitude] = {}

    print(
        "Drive amplitude F =",
        drive_amplitude
    )

    for steps_per_period in steps_per_period_values:

        step_size = (
            drive_period
            / steps_per_period
        )

        print(
            "  Running",
            steps_per_period,
            "steps per period..."
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

        poincare_states = sample_poincare_states(
            results
        )

        estimated_period = estimate_period(
            poincare_states
        )

        all_results[drive_amplitude][
            steps_per_period
        ] = {
            "results": results,
            "poincare_states": poincare_states,
            "estimated_period": estimated_period
        }

        print(
            "   step size =",
            results["step_size"],
            "| estimated period =",
            estimated_period
        )

    print()


# ============================================================
# Numerical comparison with the finest solution
# ============================================================

print("Comparison with 1000 steps per period")
print("----------------------------------------")

for drive_amplitude in drive_amplitudes:

    reference_states = all_results[
        drive_amplitude
    ][1000]["poincare_states"]

    print(
        "F =",
        drive_amplitude
    )

    for steps_per_period in [
        250,
        500
    ]:

        comparison_states = all_results[
            drive_amplitude
        ][steps_per_period]["poincare_states"]

        distances = phase_space_difference(
            comparison_states,
            reference_states
        )

        print(
            " ",
            steps_per_period,
            "versus 1000",
            "| maximum pointwise difference =",
            np.max(distances),
            "| RMS difference =",
            np.sqrt(np.mean(distances**2))
        )

    for steps_per_period in steps_per_period_values:

        states = all_results[
            drive_amplitude
        ][steps_per_period]["poincare_states"]

        print(
            " ",
            steps_per_period,
            "steps/period",
            "| theta range =",
            (
                np.min(states[:, 0]),
                np.max(states[:, 0])
            ),
            "| omega range =",
            (
                np.min(states[:, 1]),
                np.max(states[:, 1])
            )
        )

    print()


# ============================================================
# Check classification consistency
# ============================================================

print("Period-classification consistency")
print("----------------------------------------")

all_classifications_consistent = True

for drive_amplitude in drive_amplitudes:

    periods = [
        all_results[
            drive_amplitude
        ][steps_per_period]["estimated_period"]
        for steps_per_period
        in steps_per_period_values
    ]

    classifications_consistent = (
        periods[0] == periods[1] == periods[2]
    )

    if not classifications_consistent:
        all_classifications_consistent = False

    print(
        "F =",
        drive_amplitude,
        "| periods =",
        periods,
        "| consistent =",
        classifications_consistent
    )

print()

if all_classifications_consistent:
    print(
        "All period classifications are stable "
        "under step-size refinement."
    )
else:
    print(
        "At least one classification changed under "
        "step-size refinement and requires investigation."
    )


# ============================================================
# Create the comparison figure
# ============================================================

figure, axes = plt.subplots(
    nrows=len(drive_amplitudes),
    ncols=len(steps_per_period_values),
    figsize=(15, 16),
    sharex=True,
    sharey=False
)

colors = [
    "#22c1c3",
    "#f59e0b",
    "#8b5cf6"
]

for row_index, drive_amplitude in enumerate(
    drive_amplitudes
):

    for column_index, steps_per_period in enumerate(
        steps_per_period_values
    ):

        axis = axes[
            row_index,
            column_index
        ]

        stored_result = all_results[
            drive_amplitude
        ][steps_per_period]

        states = stored_result[
            "poincare_states"
        ]

        estimated_period = stored_result[
            "estimated_period"
        ]

        axis.scatter(
            states[:, 0],
            states[:, 1],
            s=18,
            color=colors[column_index],
            alpha=0.75,
            edgecolors="none"
        )

        axis.axhline(
            0.0,
            color="#94a3b8",
            linewidth=0.8,
            linestyle="--"
        )

        axis.axvline(
            0.0,
            color="#94a3b8",
            linewidth=0.8,
            linestyle="--"
        )

        axis.grid(
            alpha=0.22
        )

        axis.set_xlim(
            -np.pi,
            np.pi
        )

        axis.set_title(
            (
                f"F = {drive_amplitude}, "
                f"{steps_per_period} steps/period\n"
                f"Estimated period = {estimated_period}"
            )
        )

        if column_index == 0:
            axis.set_ylabel(
                r"Angular velocity $\omega_n$"
            )

        if row_index == (
            len(drive_amplitudes) - 1
        ):
            axis.set_xlabel(
                r"Wrapped angle $\theta_n$"
            )

figure.suptitle(
    "Step-Size Convergence of the Poincare States",
    fontsize=18,
    y=0.995
)

figure.tight_layout()


# ============================================================
# Save the figure
# ============================================================

figure_directory = (
    Path(__file__).resolve().parent
    / "figures"
)

figure_directory.mkdir(
    parents=True,
    exist_ok=True
)

figure_path = (
    figure_directory
    / "part2_step_size_convergence.png"
)

figure.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(
    figure
)

print()
print(
    "Step-size convergence figure saved to:"
)
print(
    figure_path
)