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
# Section 2: Fixed model parameters
# ============================================================

damping = 0.5
drive_frequency = 2.0 / 3.0

theta_initial = 0.2
omega_initial = 0.0

drive_period = (
    2.0 * np.pi / drive_frequency
)


# ============================================================
# Section 3: Driving-amplitude parameter range
# ============================================================


drive_amplitudes = np.linspace(
    0.97,
    1.14,
    341
)

number_of_amplitudes = len(
    drive_amplitudes
)

# Use exactly 500 numerical steps during every driving period.
# This ensures that the end of each period lies exactly on the
# numerical time grid.

steps_per_period = 500

step_size = (
    drive_period / steps_per_period
)

# Remove transient motion before recording the attractor.
discarded_cycles = 300

# Record 100 Poincare points for each parameter value.
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
# Section 4: Initial states for the entire parameter sweep
# ============================================================

theta_values = np.full(
    number_of_amplitudes,
    theta_initial,
    dtype=float
)

omega_values = np.full(
    number_of_amplitudes,
    omega_initial,
    dtype=float
)

recorded_theta = np.zeros(
    (
        recorded_cycles,
        number_of_amplitudes
    ),
    dtype=float
)

recorded_omega = np.zeros(
    (
        recorded_cycles,
        number_of_amplitudes
    ),
    dtype=float
)


# ============================================================
# Section 5: Vectorized derivative function
# ============================================================

def calculate_derivatives(
    time,
    theta,
    omega
):
    """
    Calculate derivatives for every driving amplitude
    simultaneously.
    """

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
# Section 6: Vectorized RK4 parameter sweep
# ============================================================

print(
    "Driving-amplitude bifurcation experiment"
)

print(
    "----------------------------------------"
)

print(
    "Number of driving amplitudes:",
    number_of_amplitudes
)

print(
    "Driving-amplitude range:",
    drive_amplitudes[0],
    "to",
    drive_amplitudes[-1]
)

print(
    "Driving period:",
    drive_period
)

print(
    "Numerical step size:",
    step_size
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
    "Total RK4 steps:",
    total_steps
)

print()
print(
    "Running vectorized parameter sweep..."
)

current_time = 0.0
record_index = 0

for step_index in range(total_steps):

    # --------------------------------------------------------
    # RK4 stage 1
    # --------------------------------------------------------

    (
        k1_theta,
        k1_omega
    ) = calculate_derivatives(
        current_time,
        theta_values,
        omega_values
    )

    # --------------------------------------------------------
    # RK4 stage 2
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RK4 stage 3
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RK4 stage 4
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Complete RK4 update
    # --------------------------------------------------------

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

    # The equation depends on sin(theta), so angles that differ
    # by 2*pi represent the same physical state. Wrapping here
    # prevents the numerical angle from becoming unnecessarily
    # large during rotational motion.

    theta_values = (
        theta_values + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi

    current_time = (
        current_time + step_size
    )

    # --------------------------------------------------------
    # Poincare sampling
    # --------------------------------------------------------

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
    "Parameter sweep completed."
)

print(
    "Recorded Poincare rows:",
    record_index
)


# ============================================================
# Section 7: Prepare bifurcation points
# ============================================================

amplitude_plot_values = np.tile(
    drive_amplitudes,
    recorded_cycles
)

theta_plot_values = (
    recorded_theta.reshape(-1)
)

omega_plot_values = (
    recorded_omega.reshape(-1)
)


# ============================================================
# Section 8: Bifurcation diagram
# ============================================================

figure, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(13, 10),
    sharex=True
)

figure.suptitle(
    "Period-Doubling Region of the Forced Pendulum",
    fontsize=20,
    y=0.98
)


# ------------------------------------------------------------
# Angle bifurcation diagram
# ------------------------------------------------------------

axes[0].scatter(
    amplitude_plot_values,
    theta_plot_values,
    color="#6d28d9",
    s=1.0,
    alpha=0.55,
    edgecolors="none"
)

axes[0].set_ylabel(
    r"Sampled angle $\theta_n$"
)

axes[0].set_title(
    "Poincaré Angle versus Driving Amplitude"
)

axes[0].set_ylim(
    -np.pi,
    np.pi
)

axes[0].set_yticks(
    [
        -np.pi,
        -np.pi / 2.0,
        0.0,
        np.pi / 2.0,
        np.pi
    ]
)

axes[0].set_yticklabels(
    [
        r"$-\pi$",
        r"$-\pi/2$",
        r"$0$",
        r"$\pi/2$",
        r"$\pi$"
    ]
)

axes[0].grid(
    alpha=0.18
)


# ------------------------------------------------------------
# Angular-velocity bifurcation diagram
# ------------------------------------------------------------

axes[1].scatter(
    amplitude_plot_values,
    omega_plot_values,
    color="#ea580c",
    s=1.0,
    alpha=0.55,
    edgecolors="none"
)

axes[1].set_xlabel(
    r"Driving amplitude $F$"
)

axes[1].set_ylabel(
    r"Sampled angular velocity $\omega_n$"
)

axes[1].set_title(
    "Poincaré Angular Velocity versus Driving Amplitude"
)

axes[1].grid(
    alpha=0.18
)


figure.text(
    0.5,
    0.015,
    (
        r"$q=0.5$, "
        r"$\Omega=2/3$; "
        f"{recorded_cycles} Poincaré samples retained per amplitude"
    ),
    ha="center",
    fontsize=11
)

figure.tight_layout(
    rect=[0.0, 0.04, 1.0, 0.95]
)

bifurcation_figure_path = (
    figure_directory
    / "part2_period_doubling_zoom.png"
)

figure.savefig(
    bifurcation_figure_path,
    dpi=300,
    bbox_inches="tight"
)

print()
print(
    "Period-doubling zoom figure saved to:"
)

print(
    bifurcation_figure_path
)

plt.show()

# ============================================================
# Section 9: Parameter-encoded generative phase-space artwork
# ============================================================

artwork_theta = recorded_theta.T.reshape(-1)
artwork_omega = recorded_omega.T.reshape(-1)

artwork_amplitudes = np.repeat(
    drive_amplitudes,
    recorded_cycles
)

maximum_artwork_omega = np.max(
    np.abs(artwork_omega)
)

if maximum_artwork_omega == 0.0:
    maximum_artwork_omega = 1.0

# Angular speed controls point size.
artwork_point_sizes = (
    3.0
    + 18.0
    * np.abs(artwork_omega)
    / maximum_artwork_omega
)

artwork_figure, artwork_axis = plt.subplots(
    figsize=(13, 8),
    facecolor="#070914"
)

artwork_axis.set_facecolor(
    "#070914"
)

artwork_scatter = artwork_axis.scatter(
    artwork_theta,
    artwork_omega,
    c=artwork_amplitudes,
    s=artwork_point_sizes,
    cmap="turbo",
    alpha=0.38,
    edgecolors="none"
)

artwork_axis.axhline(
    0.0,
    color="white",
    linewidth=0.7,
    alpha=0.25
)

artwork_axis.axvline(
    0.0,
    color="white",
    linewidth=0.7,
    alpha=0.25
)

artwork_axis.set_xlim(
    -np.pi,
    np.pi
)

artwork_axis.set_xticks(
    [
        -np.pi,
        -np.pi / 2.0,
        0.0,
        np.pi / 2.0,
        np.pi
    ]
)

artwork_axis.set_xticklabels(
    [
        r"$-\pi$",
        r"$-\pi/2$",
        r"$0$",
        r"$\pi/2$",
        r"$\pi$"
    ],
    color="white"
)

artwork_axis.tick_params(
    axis="y",
    colors="white"
)

artwork_axis.set_xlabel(
    r"Stroboscopic angle $\theta_n$ (rad)",
    color="white",
    fontsize=13
)

artwork_axis.set_ylabel(
    r"Stroboscopic angular velocity $\omega_n$",
    color="white",
    fontsize=13
)

artwork_axis.set_title(
    "From Periodicity to Complexity",
    color="white",
    fontsize=24,
    pad=18
)

artwork_axis.grid(
    color="white",
    alpha=0.08
)

for spine in artwork_axis.spines.values():
    spine.set_color("#94a3b8")
    spine.set_alpha(0.65)

artwork_colorbar = artwork_figure.colorbar(
    artwork_scatter,
    ax=artwork_axis,
    pad=0.025
)

artwork_colorbar.set_label(
    r"Driving amplitude $F$",
    color="white",
    fontsize=12
)

artwork_colorbar.ax.tick_params(
    colors="white"
)

artwork_figure.text(
    0.5,
    0.015,
    (
        "Position encodes the Poincaré state; "
        "colour encodes driving amplitude; "
        "point size encodes angular speed."
    ),
    ha="center",
    color="#cbd5e1",
    fontsize=11
)

artwork_figure.tight_layout(
    rect=[0.0, 0.045, 1.0, 1.0]
)

artwork_path = (
    figure_directory
    / "part2_generative_phase_transition.png"
)

artwork_figure.savefig(
    artwork_path,
    dpi=300,
    bbox_inches="tight",
    facecolor=artwork_figure.get_facecolor()
)

print()
print("Generative phase-space artwork saved to:")
print(artwork_path)

plt.show()