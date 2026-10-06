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
# Section 2: Shared numerical parameters
# ============================================================

theta_initial = 0.2
omega_initial = 0.0

drive_frequency = 2.0 / 3.0

total_time = 60.0
step_size = 0.01


# ============================================================
# Section 3: Three dynamical cases
# ============================================================

case_parameters = {
    "undamped": {
        "label": "Undamped and Unforced",
        "damping": 0.0,
        "drive_amplitude": 0.0,
        "color": "#14b8a6"
    },

    "damped": {
        "label": "Damped and Unforced",
        "damping": 0.5,
        "drive_amplitude": 0.0,
        "color": "#f59e0b"
    },

    "forced_damped": {
        "label": "Forced and Damped",
        "damping": 0.5,
        "drive_amplitude": 1.2,
        "color": "#8b5cf6"
    }
}


# ============================================================
# Section 4: Run the three simulations
# ============================================================

simulation_results = {}

print(
    "Part II: Forced and damped pendulum analysis"
)

print(
    "-------------------------------------------"
)

for case_name, parameters in case_parameters.items():

    results = simulate_forced_damped_pendulum(
        theta_initial=theta_initial,
        omega_initial=omega_initial,
        damping=parameters["damping"],
        drive_amplitude=parameters[
            "drive_amplitude"
        ],
        drive_frequency=drive_frequency,
        total_time=total_time,
        step_size=step_size
    )

    simulation_results[case_name] = results

    print(
        parameters["label"],
        "| final theta =",
        results["theta"][-1],
        "| final omega =",
        results["omega"][-1]
    )


# ============================================================
# Section 5: Time-trajectory comparison figure
# ============================================================

figure, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(15, 12),
    sharex=True
)

figure.suptitle(
    "Time Trajectories of the Nonlinear Pendulum",
    fontsize=20,
    y=0.98
)

case_order = [
    "undamped",
    "damped",
    "forced_damped"
]

for row_index, case_name in enumerate(case_order):

    parameters = case_parameters[case_name]
    results = simulation_results[case_name]

    time_values = results["time"]
    theta_values = results["theta"]
    omega_values = results["omega"]

    color = parameters["color"]

    # --------------------------------------------------------
    # Angle-time trajectory
    # --------------------------------------------------------

    angle_axis = axes[row_index, 0]

    angle_axis.plot(
        time_values,
        theta_values,
        color=color,
        linewidth=1.6
    )

    angle_axis.axhline(
        0.0,
        color="#64748b",
        linewidth=0.8,
        linestyle="--"
    )

    angle_axis.set_ylabel(
        r"Angle $\theta$"
    )

    angle_axis.set_title(
        parameters["label"]
        + r": $\theta(\tau)$"
    )

    angle_axis.grid(
        alpha=0.25
    )

    # --------------------------------------------------------
    # Angular-velocity trajectory
    # --------------------------------------------------------

    velocity_axis = axes[row_index, 1]

    velocity_axis.plot(
        time_values,
        omega_values,
        color=color,
        linewidth=1.6
    )

    velocity_axis.axhline(
        0.0,
        color="#64748b",
        linewidth=0.8,
        linestyle="--"
    )

    velocity_axis.set_ylabel(
        r"Angular velocity $\omega$"
    )

    velocity_axis.set_title(
        parameters["label"]
        + r": $\omega(\tau)$"
    )

    velocity_axis.grid(
        alpha=0.25
    )


axes[-1, 0].set_xlabel(
    r"Dimensionless time $\tau$"
)

axes[-1, 1].set_xlabel(
    r"Dimensionless time $\tau$"
)

figure.text(
    0.5,
    0.01,
    (
        r"Shared initial condition: "
        r"$\theta(0)=0.2$, "
        r"$\omega(0)=0$; "
        r"driving frequency $\Omega=2/3$"
    ),
    ha="center",
    fontsize=11
)

figure.tight_layout(
    rect=[0.0, 0.035, 1.0, 0.95]
)

time_trajectory_path = (
    figure_directory
    / "part2_time_trajectories_comparison.png"
)

figure.savefig(
    time_trajectory_path,
    dpi=300,
    bbox_inches="tight"
)

print()
print(
    "Time-trajectory comparison saved to:"
)

print(
    time_trajectory_path
)

plt.show()


# ============================================================
# Section 6: Numerical summary
# ============================================================

print()
print(
    "Time-trajectory numerical summary"
)

print(
    "-------------------------------------------"
)

for case_name in case_order:

    parameters = case_parameters[case_name]
    results = simulation_results[case_name]

    theta_values = results["theta"]
    omega_values = results["omega"]
    energy_values = results["mechanical_energy"]

    print(
        parameters["label"]
    )

    print(
        "  theta range:",
        np.min(theta_values),
        "to",
        np.max(theta_values)
    )

    print(
        "  omega range:",
        np.min(omega_values),
        "to",
        np.max(omega_values)
    )

    print(
        "  initial mechanical energy:",
        energy_values[0]
    )

    print(
        "  final mechanical energy:",
        energy_values[-1]
    )
    
    # ============================================================
# Section 7: Phase-portrait comparison
# ============================================================

phase_figure, phase_axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(17, 5.5),
    sharex=True
)

phase_figure.suptitle(
    "Phase Portraits of the Nonlinear Pendulum",
    fontsize=20,
    y=1.02
)


def prepare_wrapped_phase_curve(
    wrapped_theta_values,
    omega_values
):
    """
    Break the plotted curve whenever the wrapped angle jumps
    between pi and -pi.

    This prevents artificial horizontal lines from appearing
    across the phase portrait.
    """

    theta_for_plot = np.array(
        wrapped_theta_values,
        copy=True
    )

    omega_for_plot = np.array(
        omega_values,
        copy=True
    )

    angle_jumps = (
        np.abs(
            np.diff(theta_for_plot)
        )
        > np.pi
    )

    theta_for_plot[1:][angle_jumps] = np.nan
    omega_for_plot[1:][angle_jumps] = np.nan

    return (
        theta_for_plot,
        omega_for_plot
    )


for axis, case_name in zip(
    phase_axes,
    case_order
):

    parameters = case_parameters[case_name]
    results = simulation_results[case_name]

    (
        phase_theta,
        phase_omega
    ) = prepare_wrapped_phase_curve(
        results["wrapped_theta"],
        results["omega"]
    )

    color = parameters["color"]

    # Complete phase-space trajectory.
    axis.plot(
        phase_theta,
        phase_omega,
        color=color,
        linewidth=1.3,
        alpha=0.9,
        label="Trajectory"
    )

    # Initial state.
    axis.scatter(
        [
            results["wrapped_theta"][0]
        ],
        [
            results["omega"][0]
        ],
        color="#22c55e",
        edgecolor="black",
        s=75,
        zorder=5,
        label="Initial state"
    )

    # Final state.
    axis.scatter(
        [
            results["wrapped_theta"][-1]
        ],
        [
            results["omega"][-1]
        ],
        color="#ef4444",
        edgecolor="black",
        marker="X",
        s=85,
        zorder=5,
        label="Final state"
    )

    # Stable equilibrium point.
    axis.scatter(
        [0.0],
        [0.0],
        color="#2563eb",
        edgecolor="white",
        s=55,
        zorder=4,
        label="Stable equilibrium"
    )

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

    axis.set_xlim(
        -np.pi,
        np.pi
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

    axis.set_title(
        parameters["label"],
        fontsize=14
    )

    axis.set_xlabel(
        r"Wrapped angle $\theta$"
    )

    axis.grid(
        alpha=0.25
    )


phase_axes[0].set_ylabel(
    r"Angular velocity $\omega$"
)

phase_axes[0].legend(
    loc="best",
    fontsize=8
)

phase_figure.tight_layout()

phase_portrait_path = (
    figure_directory
    / "part2_phase_portraits_comparison.png"
)

phase_figure.savefig(
    phase_portrait_path,
    dpi=300,
    bbox_inches="tight"
)

print()
print(
    "Phase-portrait comparison saved to:"
)

print(
    phase_portrait_path
)

plt.show()