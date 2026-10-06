# ============================================================
# Part I: Simple Pendulum — Order and Geometry
# Section 1: Imports and common parameters
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from single_pendulum_model import simulate_single_pendulum
from matplotlib.collections import LineCollection

# Physical parameters
g = 9.81
length = 1.0
mass = 1.0

# Numerical parameters
total_time = 12.0
step_size = 0.0025


# Critical angular velocity when theta(0) = 0
critical_omega = 2.0 * np.sqrt(g / length)

print("Critical angular velocity:")
print(critical_omega, "rad/s")

# ============================================================
# Section 2: Initial conditions for three motion regimes
# ============================================================

motion_cases = {
    "oscillation": {
        "theta_initial": 1.5,
        "omega_initial": 0.0,
        "color": "#4FD1C5"
    },

    "separatrix": {
        "theta_initial": 0.0,
        "omega_initial": critical_omega,
        "color": "#F6C85F"
    },

    "rotation": {
        "theta_initial": 0.0,
        "omega_initial": 7.0,
        "color": "#F97316"
    }
}


print("\nInitial conditions:")
print("-----------------------------------")

for case_name, parameters in motion_cases.items():

    print(
        case_name,
        "| theta(0) =",
        parameters["theta_initial"],
        "rad | omega(0) =",
        parameters["omega_initial"],
        "rad/s"
    )
    
# ============================================================
# Section 3: Run the three simulations
# ============================================================

simulation_results = {}

print("\nRunning simulations:")
print("-----------------------------------")

for case_name, parameters in motion_cases.items():

    results = simulate_single_pendulum(
        theta_initial=parameters["theta_initial"],
        omega_initial=parameters["omega_initial"],
        length=length,
        mass=mass,
        damping=0.0,
        total_time=total_time,
        step_size=step_size,
        g=g
    )

    simulation_results[case_name] = results

    print(
        case_name,
        "completed | number of time points =",
        len(results["time"])
    )


print("\nAvailable result names:")
print(simulation_results["oscillation"].keys())

# ============================================================
# Section 4: Angle-time comparison
# ============================================================

figure_directory = (
    Path(__file__).resolve().parent
    / "figures"
)

figure_directory.mkdir(
    parents=True,
    exist_ok=True
)


figure, axes = plt.subplots(
    3,
    1,
    figsize=(10, 11),
    sharex=True
)


case_titles = {
    "oscillation": "Oscillatory Motion: $E < 2mgL$",
    "separatrix": "Separatrix Motion: $E = 2mgL$",
    "rotation": "Rotational Motion: $E > 2mgL$"
}


for axis, case_name in zip(
    axes,
    [
        "oscillation",
        "separatrix",
        "rotation"
    ]
):

    results = simulation_results[case_name]
    color = motion_cases[case_name]["color"]

    axis.plot(
        results["time"],
        results["theta"],
        color=color,
        linewidth=2.0
    )

    axis.axhline(
        np.pi,
        color="#9CA3AF",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label=r"$\theta=\pi$"
    )

    axis.axhline(
        -np.pi,
        color="#9CA3AF",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label=r"$\theta=-\pi$"
    )

    axis.set_title(
        case_titles[case_name],
        fontsize=13
    )

    axis.set_ylabel(
        r"Angle $\theta$ (rad)"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend(
        loc="upper left"
    )


axes[-1].set_xlabel(
    "Time (s)"
)


figure.suptitle(
    "Three Motion Regimes of the Nonlinear Simple Pendulum",
    fontsize=16,
    y=0.995
)


figure.tight_layout()


angle_time_path = (
    figure_directory
    / "part1_angle_time_three_regimes.png"
)


figure.savefig(
    angle_time_path,
    dpi=300,
    bbox_inches="tight"
)


print("\nAngle-time figure saved to:")
print(angle_time_path)


plt.show()

# ============================================================
# Section 5: Phase portraits of the three motion regimes
# ============================================================

phase_figure, phase_axes = plt.subplots(
    1,
    3,
    figsize=(17, 5.5),
    sharey=True
)


# Theoretical separatrix
theta_boundary = np.linspace(
    -np.pi,
    np.pi,
    1000
)

omega_boundary_positive = (
    2.0
    * np.sqrt(g / length)
    * np.cos(theta_boundary / 2.0)
)

omega_boundary_negative = (
    -omega_boundary_positive
)


for axis, case_name in zip(
    phase_axes,
    [
        "oscillation",
        "separatrix",
        "rotation"
    ]
):

    results = simulation_results[case_name]
    color = motion_cases[case_name]["color"]

    # Wrap every angle into the interval [-pi, pi).
    theta_wrapped = (
        (results["theta"] + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )

    # Draw the theoretical separatrix.
    axis.plot(
        theta_boundary,
        omega_boundary_positive,
        color="#6B7280",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Theoretical separatrix"
    )

    axis.plot(
        theta_boundary,
        omega_boundary_negative,
        color="#6B7280",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8
    )

    # Draw the numerical trajectory.
    axis.scatter(
        theta_wrapped,
        results["omega"],
        color=color,
        s=4,
        alpha=0.75,
        label="Numerical trajectory"
    )

    # Stable equilibrium at theta = 0.
    axis.scatter(
        [0.0],
        [0.0],
        color="#2563EB",
        marker="o",
        s=80,
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
        label="Stable equilibrium"
    )

    # Unstable equilibria at theta = +/- pi.
    axis.scatter(
        [-np.pi, np.pi],
        [0.0, 0.0],
        color="#DC2626",
        marker="X",
        s=90,
        zorder=5,
        label="Unstable equilibrium"
    )

    axis.set_title(
        case_name.capitalize(),
        fontsize=14
    )

    axis.set_xlabel(
        r"Wrapped angle $\theta$ (rad)"
    )

    axis.set_xlim(
        -np.pi - 0.2,
        np.pi + 0.2
    )

    axis.set_xticks(
        [
            -np.pi,
            -np.pi / 2.0,
            0.0,
            np.pi / 2.0,
            np.pi
        ],
        [
            r"$-\pi$",
            r"$-\pi/2$",
            r"$0$",
            r"$\pi/2$",
            r"$\pi$"
        ]
    )

    axis.grid(
        alpha=0.25
    )


phase_axes[0].set_ylabel(
    r"Angular velocity $\omega$ (rad/s)"
)


phase_axes[0].legend(
    loc="upper right",
    fontsize=8
)


phase_figure.suptitle(
    "Phase-Space Geometry of the Nonlinear Simple Pendulum",
    fontsize=17,
    y=1.02
)


phase_figure.tight_layout()


phase_portrait_path = (
    figure_directory
    / "part1_phase_portraits_three_regimes.png"
)


phase_figure.savefig(
    phase_portrait_path,
    dpi=300,
    bbox_inches="tight"
)


print("\nPhase-portrait figure saved to:")
print(phase_portrait_path)


plt.show()

# ============================================================
# Section 6: Numerical equilibrium-stability experiment
# ============================================================

equilibrium_cases = {
    "near_stable": {
        "theta_initial": 0.10,
        "omega_initial": 0.0,
        "color": "#2563EB",
        "label": (
            r"Near stable equilibrium: "
            r"$\theta(0)=0.10$"
        )
    },

    "unstable_left": {
        "theta_initial": np.pi - 0.01,
        "omega_initial": 0.0,
        "color": "#DC2626",
        "label": (
            r"Left perturbation: "
            r"$\theta(0)=\pi-0.01$"
        )
    },

    "unstable_right": {
        "theta_initial": np.pi + 0.01,
        "omega_initial": 0.0,
        "color": "#F97316",
        "label": (
            r"Right perturbation: "
            r"$\theta(0)=\pi+0.01$"
        )
    }
}


equilibrium_results = {}


for case_name, parameters in equilibrium_cases.items():

    equilibrium_results[case_name] = (
        simulate_single_pendulum(
            theta_initial=parameters["theta_initial"],
            omega_initial=parameters["omega_initial"],
            length=length,
            mass=mass,
            damping=0.0,
            total_time=8.0,
            step_size=step_size,
            g=g
        )
    )


equilibrium_figure, equilibrium_axes = plt.subplots(
    2,
    1,
    figsize=(10, 8),
    sharex=True
)


# Stable-equilibrium perturbation
stable_results = equilibrium_results["near_stable"]

equilibrium_axes[0].plot(
    stable_results["time"],
    stable_results["theta"],
    color=equilibrium_cases["near_stable"]["color"],
    linewidth=2.0,
    label=equilibrium_cases["near_stable"]["label"]
)

equilibrium_axes[0].axhline(
    0.0,
    color="#111827",
    linestyle="--",
    linewidth=1.0,
    label=r"Stable equilibrium $\theta=0$"
)

equilibrium_axes[0].set_title(
    "Perturbation Near the Stable Equilibrium"
)

equilibrium_axes[0].set_ylabel(
    r"Angle $\theta$ (rad)"
)

equilibrium_axes[0].legend()

equilibrium_axes[0].grid(
    alpha=0.25
)


# Unstable-equilibrium perturbations
for case_name in [
    "unstable_left",
    "unstable_right"
]:

    results = equilibrium_results[case_name]
    parameters = equilibrium_cases[case_name]

    equilibrium_axes[1].plot(
        results["time"],
        results["theta"],
        color=parameters["color"],
        linewidth=2.0,
        label=parameters["label"]
    )


equilibrium_axes[1].axhline(
    np.pi,
    color="#111827",
    linestyle="--",
    linewidth=1.0,
    label=r"Unstable equilibrium $\theta=\pi$"
)

equilibrium_axes[1].set_title(
    "Perturbations Near the Unstable Equilibrium"
)

equilibrium_axes[1].set_xlabel(
    "Time (s)"
)

equilibrium_axes[1].set_ylabel(
    r"Angle $\theta$ (rad)"
)

equilibrium_axes[1].legend()

equilibrium_axes[1].grid(
    alpha=0.25
)


equilibrium_figure.suptitle(
    "Numerical Stability of the Simple-Pendulum Equilibria",
    fontsize=16,
    y=1.01
)


equilibrium_figure.tight_layout()


equilibrium_path = (
    figure_directory
    / "part1_equilibrium_stability.png"
)


equilibrium_figure.savefig(
    equilibrium_path,
    dpi=300,
    bbox_inches="tight"
)


print("\nEquilibrium-stability figure saved to:")
print(equilibrium_path)


plt.show()

# ============================================================
# Section 7: Energy-conservation validation
# ============================================================

print("\nEnergy-conservation validation:")
print("-----------------------------------")

for case_name in [
    "oscillation",
    "separatrix",
    "rotation"
]:

    results = simulation_results[case_name]

    print(
        case_name,
        "| initial energy =",
        results["initial_total_energy"],
        "J | maximum relative energy change =",
        results["maximum_relative_energy_change"]
    )


energy_figure, energy_axes = plt.subplots(
    3,
    1,
    figsize=(10, 10),
    sharex=True
)


for axis, case_name in zip(
    energy_axes,
    [
        "oscillation",
        "separatrix",
        "rotation"
    ]
):

    results = simulation_results[case_name]
    color = motion_cases[case_name]["color"]

    axis.plot(
        results["time"],
        results["relative_energy_change"],
        color=color,
        linewidth=1.8
    )

    axis.set_title(
        case_name.capitalize()
    )

    axis.set_ylabel(
        "Relative energy change"
    )

    axis.grid(
        alpha=0.25
    )

    axis.ticklabel_format(
        axis="y",
        style="scientific",
        scilimits=(0, 0)
    )


energy_axes[-1].set_xlabel(
    "Time (s)"
)


energy_figure.suptitle(
    "Energy-Conservation Validation for the Three Motion Regimes",
    fontsize=16,
    y=1.01
)


energy_figure.tight_layout()


energy_validation_path = (
    figure_directory
    / "part1_energy_validation_three_regimes.png"
)


energy_figure.savefig(
    energy_validation_path,
    dpi=300,
    bbox_inches="tight"
)


print("\nEnergy-validation figure saved to:")
print(energy_validation_path)


plt.show()

# ============================================================
# Section 8: Generative phase-space visualization
# ============================================================

generative_initial_omegas = [
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    -6.5,
    6.5,
    -7.0,
    7.0,
    -8.0,
    8.0
]


generative_results = []


for initial_omega in generative_initial_omegas:

    results = simulate_single_pendulum(
        theta_initial=0.0,
        omega_initial=initial_omega,
        length=length,
        mass=mass,
        damping=0.0,
        total_time=10.0,
        step_size=step_size,
        g=g
    )

    generative_results.append(
        results
    )


initial_energies = np.array(
    [
        results["initial_total_energy"]
        for results in generative_results
    ]
)


energy_normalization = plt.Normalize(
    vmin=np.min(initial_energies),
    vmax=np.max(initial_energies)
)


energy_color_map = plt.colormaps["plasma"]


generative_figure, generative_axis = plt.subplots(
    figsize=(12, 8)
)


background_color = "#070A12"

generative_figure.patch.set_facecolor(
    background_color
)

generative_axis.set_facecolor(
    background_color
)


for results in generative_results:

    theta_wrapped = (
        (results["theta"] + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )

    omega_values = results["omega"]

    phase_points = np.column_stack(
        [
            theta_wrapped,
            omega_values
        ]
    ).reshape(
        -1,
        1,
        2
    )

    phase_segments = np.concatenate(
        [
            phase_points[:-1],
            phase_points[1:]
        ],
        axis=1
    )

    # Remove artificial lines crossing from pi to -pi.
    valid_segments = (
        np.abs(
            np.diff(theta_wrapped)
        )
        < np.pi
    )

    phase_segments = phase_segments[
        valid_segments
    ]

    segment_speeds = (
        0.5
        * (
            np.abs(omega_values[:-1])
            + np.abs(omega_values[1:])
        )
    )[valid_segments]

    maximum_segment_speed = np.max(
        segment_speeds
    )

    if maximum_segment_speed == 0:
        maximum_segment_speed = 1.0

    segment_widths = (
        0.5
        + 2.8
        * segment_speeds
        / maximum_segment_speed
    )

    trajectory_energy = (
        results["initial_total_energy"]
    )

    trajectory_color = energy_color_map(
        energy_normalization(
            trajectory_energy
        )
    )

    trajectory_collection = LineCollection(
        phase_segments,
        colors=[trajectory_color],
        linewidths=segment_widths,
        alpha=0.82,
        zorder=3
    )

    generative_axis.add_collection(
        trajectory_collection
    )


# Draw the theoretical separatrix.
generative_axis.plot(
    theta_boundary,
    omega_boundary_positive,
    color="#E5E7EB",
    linestyle="--",
    linewidth=1.2,
    alpha=0.7,
    zorder=2
)

generative_axis.plot(
    theta_boundary,
    omega_boundary_negative,
    color="#E5E7EB",
    linestyle="--",
    linewidth=1.2,
    alpha=0.7,
    zorder=2
)


# Equilibrium points
generative_axis.scatter(
    [0.0],
    [0.0],
    color="#5EEAD4",
    s=90,
    edgecolor="white",
    linewidth=1.2,
    zorder=6
)

generative_axis.scatter(
    [-np.pi, np.pi],
    [0.0, 0.0],
    color="#FB7185",
    marker="X",
    s=100,
    zorder=6
)


generative_axis.set_xlim(
    -np.pi - 0.25,
    np.pi + 0.25
)

generative_axis.set_ylim(
    -8.7,
    8.7
)


generative_axis.set_xticks(
    [
        -np.pi,
        -np.pi / 2.0,
        0.0,
        np.pi / 2.0,
        np.pi
    ],
    [
        r"$-\pi$",
        r"$-\pi/2$",
        r"$0$",
        r"$\pi/2$",
        r"$\pi$"
    ]
)


generative_axis.set_title(
    "From Oscillation to Rotation",
    color="white",
    fontsize=20,
    pad=18
)

generative_axis.set_xlabel(
    r"Wrapped angle $\theta$ (rad)",
    color="white",
    fontsize=13
)

generative_axis.set_ylabel(
    r"Angular velocity $\omega$ (rad/s)",
    color="white",
    fontsize=13
)


generative_axis.tick_params(
    colors="white"
)

generative_axis.grid(
    color="white",
    alpha=0.10
)


for spine in generative_axis.spines.values():
    spine.set_color(
        "#9CA3AF"
    )


energy_scalar_map = plt.cm.ScalarMappable(
    norm=energy_normalization,
    cmap=energy_color_map
)

energy_scalar_map.set_array([])


energy_color_bar = generative_figure.colorbar(
    energy_scalar_map,
    ax=generative_axis,
    pad=0.03
)

energy_color_bar.set_label(
    "Initial total energy (J)",
    color="white",
    fontsize=12
)

energy_color_bar.ax.tick_params(
    colors="white"
)

energy_color_bar.outline.set_edgecolor(
    "#9CA3AF"
)


generative_figure.tight_layout()


generative_path = (
    figure_directory
    / "part1_generative_phase_space.png"
)


generative_figure.savefig(
    generative_path,
    dpi=300,
    bbox_inches="tight",
    facecolor=generative_figure.get_facecolor()
)


print("\nGenerative phase-space artwork saved to:")
print(generative_path)


plt.show()