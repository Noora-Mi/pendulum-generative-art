import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from pathlib import Path

# -----------------------------
# 1. Physical parameters
# -----------------------------

g = 9.81          # gravitational acceleration (m/s^2)

m1 = 1.0   # mass of the first bob (kg)
m2 = 1.0     # mass of the second bob (kg)

L1 =  1.0      # length of the first pendulum (m)
L2 = 1.0  # length of the second pendulum (m)


# -----------------------------
# 2. Initial conditions
# -----------------------------

theta1_initial = 3    # initial angle of the first pendulum (rad)
omega1_initial = 0.0     # initial angular velocity of pendulum 1

theta2_initial = -3    # initial angle of the second pendulum (rad)
omega2_initial = 0.0     # initial angular velocity of pendulum 2


# -----------------------------
# 3. Simulation settings
# -----------------------------

t_initial = 0.0
t_final = 20.0
h = 0.005

# Number of RK4 steps
number_of_steps = int(
    round(
        (t_final - t_initial) / h
    )
)

# Create exactly 2001 time points from 0 to 20 seconds
t_values = np.linspace(
    t_initial,
    t_final,
    number_of_steps + 1
)


# -----------------------------
# 4. State array
# -----------------------------

# Each row stores:
# [theta1, omega1, theta2, omega2]

y_values = np.zeros(
    (len(t_values), 4)
)

y_values[0] = [
    theta1_initial,
    omega1_initial,
    theta2_initial,
    omega2_initial
]


# -----------------------------
# 5. Output settings
# -----------------------------

# Label for the current parameter test
test_label = "A2_boundary_angles"

save_figures = True
save_animation = False

show_analysis_figures = True
show_artwork = True

# Save figures beside this Python file
script_directory = Path(__file__).resolve().parent
figure_directory = script_directory / "figures"

figure_directory.mkdir(
    parents=True,
    exist_ok=True
)
# -----------------------------
# 5. Convert angles to coordinates
# -----------------------------

def angles_to_coordinates(theta1, theta2):
    """
    Convert the two absolute pendulum angles
    into Cartesian coordinates.

    Both theta1 and theta2 are measured from
    the downward vertical direction.
    """

    # Position of the first bob
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)

    # Position of the second bob
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    return x1, y1, x2, y2


# Calculate the initial coordinates
x1_initial, y1_initial, x2_initial, y2_initial = (
    angles_to_coordinates(
        theta1_initial,
        theta2_initial
    )
)


print("Initial position of bob 1:")
print("x1 =", x1_initial)
print("y1 =", y1_initial)

print("Initial position of bob 2:")
print("x2 =", x2_initial)
print("y2 =", y2_initial)

# -----------------------------
# 6. Plot the initial configuration
# -----------------------------

fig, ax = plt.subplots(
    figsize=(7, 7)
)

fig.patch.set_facecolor("#070711")
ax.set_facecolor("#070711")


# Draw the two pendulum links
ax.plot(
    [0, x1_initial, x2_initial],
    [0, y1_initial, y2_initial],
    color="white",
    linewidth=2,
    marker="o",
    markersize=5,
    zorder=3
)


# Draw the fixed pivot
ax.scatter(
    0,
    0,
    color="white",
    s=80,
    zorder=5,
    label="Pivot"
)


# Draw the first bob
ax.scatter(
    x1_initial,
    y1_initial,
    color="#5eead4",
    s=130,
    zorder=5,
    label="Bob 1"
)


# Draw the second bob
ax.scatter(
    x2_initial,
    y2_initial,
    color="#f97316",
    s=130,
    zorder=5,
    label="Bob 2"
)


# Total possible spatial range
total_length = L1 + L2

ax.set_xlim(
    -1.1 * total_length,
    1.1 * total_length
)

ax.set_ylim(
    -1.1 * total_length,
    0.3 * total_length
)


# Equal scaling is necessary for correct geometry
ax.set_aspect(
    "equal",
    adjustable="box"
)


ax.set_xlabel(
    "Horizontal position x (m)",
    color="white"
)

ax.set_ylabel(
    "Vertical position y (m)",
    color="white"
)

ax.set_title(
    "Initial Configuration of the Double Pendulum",
    color="white"
)

ax.tick_params(colors="white")
ax.grid(True, alpha=0.12)

ax.legend(
    facecolor="#11111f",
    labelcolor="white"
)

plt.show()

# -----------------------------
# 7. Velocity components
# -----------------------------

def calculate_velocities(
    theta1,
    omega1,
    theta2,
    omega2
):
    """
    Calculate the Cartesian velocity components
    of the two pendulum bobs.
    """

    # Velocity of bob 1
    vx1 = (
        L1
        * np.cos(theta1)
        * omega1
    )

    vy1 = (
        L1
        * np.sin(theta1)
        * omega1
    )

    # Velocity of bob 2
    vx2 = (
        L1
        * np.cos(theta1)
        * omega1
        + L2
        * np.cos(theta2)
        * omega2
    )

    vy2 = (
        L1
        * np.sin(theta1)
        * omega1
        + L2
        * np.sin(theta2)
        * omega2
    )

    return vx1, vy1, vx2, vy2

# -----------------------------
# 8. Kinetic energy
# -----------------------------

def calculate_kinetic_energy(
    theta1,
    omega1,
    theta2,
    omega2
):
    """
    Calculate the total kinetic energy
    of the double pendulum.
    """

    vx1, vy1, vx2, vy2 = (
        calculate_velocities(
            theta1,
            omega1,
            theta2,
            omega2
        )
    )

    # Kinetic energy of bob 1
    kinetic_energy_1 = (
        0.5
        * m1
        * (vx1**2 + vy1**2)
    )

    # Kinetic energy of bob 2
    kinetic_energy_2 = (
        0.5
        * m2
        * (vx2**2 + vy2**2)
    )

    total_kinetic_energy = (
        kinetic_energy_1
        + kinetic_energy_2
    )

    return total_kinetic_energy

# -----------------------------
# 9. Potential energy
# -----------------------------

def calculate_potential_energy(
    theta1,
    theta2
):
    """
    Calculate the gravitational potential energy.

    The lowest configuration,
    theta1 = theta2 = 0,
    is chosen as the zero-potential reference.
    """

    potential_energy_1 = (
        (m1 + m2)
        * g
        * L1
        * (1 - np.cos(theta1))
    )

    potential_energy_2 = (
        m2
        * g
        * L2
        * (1 - np.cos(theta2))
    )

    total_potential_energy = (
        potential_energy_1
        + potential_energy_2
    )

    return total_potential_energy

# -----------------------------
# 10. Lagrangian
# -----------------------------

def calculate_lagrangian(
    theta1,
    omega1,
    theta2,
    omega2
):
    """
    Calculate the Lagrangian:

        Lagrangian = kinetic energy - potential energy
    """

    kinetic_energy = (
        calculate_kinetic_energy(
            theta1,
            omega1,
            theta2,
            omega2
        )
    )

    potential_energy = (
        calculate_potential_energy(
            theta1,
            theta2
        )
    )

    lagrangian = (
        kinetic_energy
        - potential_energy
    )

    return lagrangian

# -----------------------------
# 11. Total mechanical energy
# -----------------------------

def calculate_total_energy(
    theta1,
    omega1,
    theta2,
    omega2
):
    """
    Calculate the total mechanical energy:

        total energy = kinetic energy + potential energy
    """

    kinetic_energy = (
        calculate_kinetic_energy(
            theta1,
            omega1,
            theta2,
            omega2
        )
    )

    potential_energy = (
        calculate_potential_energy(
            theta1,
            theta2
        )
    )

    total_energy = (
        kinetic_energy
        + potential_energy
    )

    return total_energy

# -----------------------------
# 12. Double-pendulum differential equation
# -----------------------------

def double_pendulum_model(t, y):
    """
    Return the time derivative of the state:

    y = [theta1, omega1, theta2, omega2]

    dy/dt = [omega1, alpha1, omega2, alpha2]
    """

    theta1 = y[0]
    omega1 = y[1]
    theta2 = y[2]
    omega2 = y[3]

    # Difference between the two absolute angles
    delta = theta1 - theta2

    # ------------------------------------------
    # Coefficient matrix for angular acceleration
    # ------------------------------------------

    coefficient_matrix = np.array([
        [
            (m1 + m2) * L1**2,
            m2 * L1 * L2 * np.cos(delta)
        ],
        [
            m2 * L1 * L2 * np.cos(delta),
            m2 * L2**2
        ]
    ])

    # ------------------------------------------
    # Right-hand side of the equations
    # ------------------------------------------

    b1 = (
        -m2
        * L1
        * L2
        * omega2**2
        * np.sin(delta)
        - (m1 + m2)
        * g
        * L1
        * np.sin(theta1)
    )

    b2 = (
        m2
        * L1
        * L2
        * omega1**2
        * np.sin(delta)
        - m2
        * g
        * L2
        * np.sin(theta2)
    )

    right_hand_side = np.array([
        b1,
        b2
    ])

    # Solve the 2 x 2 linear system:
    #
    # coefficient_matrix @ [alpha1, alpha2]
    #     = right_hand_side

    angular_accelerations = np.linalg.solve(
        coefficient_matrix,
        right_hand_side
    )

    alpha1 = angular_accelerations[0]
    alpha2 = angular_accelerations[1]

    return np.array([
        omega1,
        alpha1,
        omega2,
        alpha2
    ])
    
    # -----------------------------
# 13. One RK4 step
# -----------------------------

def rk4_step(function, t, y, step_size):
    """
    Advance the state by one time step
    using the fourth-order Runge--Kutta method.
    """

    k1 = function(
        t,
        y
    )

    k2 = function(
        t + step_size / 2,
        y + step_size * k1 / 2
    )

    k3 = function(
        t + step_size / 2,
        y + step_size * k2 / 2
    )

    k4 = function(
        t + step_size,
        y + step_size * k3
    )

    y_next = (
        y
        + step_size
        * (
            k1
            + 2 * k2
            + 2 * k3
            + k4
        )
        / 6
    )

    return y_next

# -----------------------------
# 14. Run the simulation
# -----------------------------

for n in range(
    len(t_values) - 1
):
    y_values[n + 1] = rk4_step(
        double_pendulum_model,
        t_values[n],
        y_values[n],
        h
    )
    
# -----------------------------
# 15. Separate the state variables
# -----------------------------

theta1_values = y_values[:, 0]
omega1_values = y_values[:, 1]

theta2_values = y_values[:, 2]
omega2_values = y_values[:, 3]

# -----------------------------
# 16. Convert the simulation to coordinates
# -----------------------------

x1_values, y1_values, x2_values, y2_values = (
    angles_to_coordinates(
        theta1_values,
        theta2_values
    )
)

# -----------------------------
# 17. Basic numerical check
# -----------------------------

print("\nDouble-pendulum simulation completed.")

print(
    "Number of time points:",
    len(t_values)
)

print(
    "Final simulation time:",
    t_values[-1]
)

print(
    "Initial state:",
    y_values[0]
)

print(
    "Final state:",
    y_values[-1]
)

print(
    "Initial position of bob 2:",
    x2_values[0],
    y2_values[0]
)

print(
    "Final position of bob 2:",
    x2_values[-1],
    y2_values[-1]
)

# -----------------------------
# 18. Energy validation
# -----------------------------

# Calculate kinetic energy at every time point
kinetic_energy_values = (
    calculate_kinetic_energy(
        theta1_values,
        omega1_values,
        theta2_values,
        omega2_values
    )
)

# Calculate potential energy at every time point
potential_energy_values = (
    calculate_potential_energy(
        theta1_values,
        theta2_values
    )
)

# Total mechanical energy
total_energy_values = (
    kinetic_energy_values
    + potential_energy_values
)

# Initial total energy
initial_energy = total_energy_values[0]

# Difference between energy at each time
# and the initial energy
energy_difference = (
    total_energy_values
    - initial_energy
)

# Maximum absolute energy error
maximum_energy_error = np.max(
    np.abs(energy_difference)
)

# Maximum relative energy error
relative_energy_error = (
    maximum_energy_error
    / abs(initial_energy)
)


print("\nEnergy validation:")

print(
    "Initial total energy:",
    initial_energy,
    "J"
)

print(
    "Maximum absolute energy error:",
    maximum_energy_error,
    "J"
)

print(
    "Maximum relative energy error:",
    relative_energy_error
)

# -----------------------------
# 19. Energy plot
# -----------------------------

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.plot(
    t_values,
    kinetic_energy_values,
    color="darkred",
    label="Kinetic energy"
)

ax.plot(
    t_values,
    potential_energy_values,
    color="darkgreen",
    label="Potential energy"
)

ax.plot(
    t_values,
    total_energy_values,
    color="black",
    linestyle="--",
    linewidth=2,
    label="Total energy"
)

ax.set_xlabel(
    "Time (s)"
)

ax.set_ylabel(
    "Energy (J)"
)

ax.set_title(
    "Energy Conservation of the Double Pendulum"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()

if save_figures:
    plt.savefig(
        figure_directory
        / f"double_pendulum_energy_{test_label}.png",
        dpi=300,
        bbox_inches="tight"
    )
    
if show_analysis_figures:
    plt.show()
else:
    plt.close()
    
# -----------------------------
# 20. Angle-versus-time plot
# -----------------------------

fig, ax = plt.subplots(
    figsize=(10, 6)
)

# Angle of the first pendulum
ax.plot(
    t_values,
    theta1_values,
    color="#2563eb",
    linewidth=1.5,
    label=r"$\theta_1(t)$"
)

# Angle of the second pendulum
ax.plot(
    t_values,
    theta2_values,
    color="#f97316",
    linewidth=1.5,
    label=r"$\theta_2(t)$"
)

# Horizontal equilibrium line
ax.axhline(
    0,
    color="black",
    linewidth=0.8,
    alpha=0.5
)

ax.set_xlabel(
    "Time (s)"
)

ax.set_ylabel(
    "Angle (rad)"
)

ax.set_title(
    "Angular Motion of the Double Pendulum"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend()

if save_figures:
    plt.savefig(
        figure_directory
        / f"double_pendulum_angles_{test_label}.png",
        dpi=300,
        bbox_inches="tight"
    )

if show_analysis_figures:
    plt.show()
else:
    plt.close()
    
# -----------------------------
# 21. Physical trajectories
# -----------------------------

fig, ax = plt.subplots(
    figsize=(8, 8)
)

fig.patch.set_facecolor("#070711")
ax.set_facecolor("#070711")


# Trajectory of the first bob
ax.plot(
    x1_values,
    y1_values,
    color="#5eead4",
    linewidth=1.5,
    alpha=0.75,
    label="Trajectory of bob 1"
)


# Trajectory of the second bob
ax.plot(
    x2_values,
    y2_values,
    color="#f97316",
    linewidth=1.3,
    alpha=0.75,
    label="Trajectory of bob 2"
)


# Mark the fixed pivot
ax.scatter(
    0,
    0,
    color="white",
    s=80,
    zorder=5,
    label="Pivot"
)


# Mark the initial position of bob 1
ax.scatter(
    x1_values[0],
    y1_values[0],
    color="#5eead4",
    edgecolor="white",
    linewidth=0.8,
    s=100,
    zorder=5
)


# Mark the initial position of bob 2
ax.scatter(
    x2_values[0],
    y2_values[0],
    color="#f97316",
    edgecolor="white",
    linewidth=0.8,
    s=100,
    zorder=5
)


# Draw the initial double-pendulum configuration
ax.plot(
    [
        0,
        x1_values[0],
        x2_values[0]
    ],
    [
        0,
        y1_values[0],
        y2_values[0]
    ],
    color="white",
    linewidth=1.2,
    alpha=0.65,
    zorder=3
)


# Maximum possible distance from the pivot
total_length = L1 + L2

ax.set_xlim(
    -1.1 * total_length,
    1.1 * total_length
)

ax.set_ylim(
    -1.1 * total_length,
    1.1 * total_length
)


# Use equal scaling so that the geometry
# is not visually distorted
ax.set_aspect(
    "equal",
    adjustable="box"
)


ax.set_xlabel(
    "Horizontal position x (m)",
    color="white"
)

ax.set_ylabel(
    "Vertical position y (m)",
    color="white"
)

ax.set_title(
    "Physical Trajectories of the Double Pendulum",
    color="white"
)

ax.tick_params(
    colors="white"
)

ax.grid(
    True,
    alpha=0.12
)

ax.legend(
    facecolor="#11111f",
    labelcolor="white"
)


if save_figures:
    plt.savefig(
        figure_directory
        / (
            "double_pendulum_physical_trajectories_"
            f"{test_label}.png"
        ),
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )


if show_analysis_figures:
    plt.show()
else:
    plt.close()
    
# -----------------------------
# 22. Synchronized double-pendulum animation
# -----------------------------

# The trajectory remains visible after it is drawn.
# Time controls color.
# Speed controls line width.

# Target animation frame rate
target_fps = 30

# Automatically calculate the number of
# numerical points skipped between frames
animation_frame_step = max(
    1,
    round(
        1 / (target_fps * h)
    )
)

# Actual time interval between animation frames
animation_interval = (
    h
    * animation_frame_step
    * 1000
)

# Actual animation frame rate
actual_fps = (
    1
    / (
        h
        * animation_frame_step
    )
)

animation_frames = range(
    0,
    len(t_values),
    animation_frame_step
)


# ==================================================
# Calculate the Cartesian speeds of both bobs
# ==================================================

vx1_values, vy1_values, vx2_values, vy2_values = (
    calculate_velocities(
        theta1_values,
        omega1_values,
        theta2_values,
        omega2_values
    )
)

bob1_speed_values = np.sqrt(
    vx1_values**2
    + vy1_values**2
)

bob2_speed_values = np.sqrt(
    vx2_values**2
    + vy2_values**2
)


# Avoid division by zero
maximum_bob1_speed = np.max(
    bob1_speed_values
)

maximum_bob2_speed = np.max(
    bob2_speed_values
)

maximum_omega1 = np.max(
    np.abs(omega1_values)
)

maximum_omega2 = np.max(
    np.abs(omega2_values)
)

if maximum_bob1_speed == 0:
    maximum_bob1_speed = 1.0

if maximum_bob2_speed == 0:
    maximum_bob2_speed = 1.0

if maximum_omega1 == 0:
    maximum_omega1 = 1.0

if maximum_omega2 == 0:
    maximum_omega2 = 1.0


# ==================================================
# Color mapping
# ==================================================

animation_color_map = plt.colormaps["turbo"]

animation_color_normalization = plt.Normalize(
    vmin=t_values[0],
    vmax=t_values[-1]
)


# ==================================================
# Create the three-panel figure
# ==================================================

animation_fig = plt.figure(
    figsize=(15, 8),
    facecolor="#070711"
)

animation_grid = animation_fig.add_gridspec(
    2,
    2,
    width_ratios=[1.25, 1.0],
    height_ratios=[1.0, 1.0],
    wspace=0.22,
    hspace=0.30
)

# Left panel occupies both rows
physical_ax = animation_fig.add_subplot(
    animation_grid[:, 0]
)

# Right-top panel
phase1_ax = animation_fig.add_subplot(
    animation_grid[0, 1]
)

# Right-bottom panel
phase2_ax = animation_fig.add_subplot(
    animation_grid[1, 1]
)


for current_ax in [
    physical_ax,
    phase1_ax,
    phase2_ax
]:
    current_ax.set_facecolor("#070711")
    current_ax.tick_params(colors="white")
    current_ax.grid(
        True,
        alpha=0.12
    )


animation_fig.suptitle(
    "Double Pendulum: Physical Motion and Phase-Space Evolution",
    color="white",
    fontsize=16
)


# ==================================================
# Left panel: physical double pendulum
# ==================================================

total_length = L1 + L2

physical_ax.set_xlim(
    -1.1 * total_length,
    1.1 * total_length
)

physical_ax.set_ylim(
    -1.1 * total_length,
    1.1 * total_length
)

physical_ax.set_aspect(
    "equal",
    adjustable="box"
)

physical_ax.set_title(
    "Physical Space",
    color="white",
    fontsize=14
)

physical_ax.set_xlabel(
    "Horizontal position $x$ (m)",
    color="white"
)

physical_ax.set_ylabel(
    "Vertical position $y$ (m)",
    color="white"
)


# Fixed pivot
physical_ax.scatter(
    0,
    0,
    color="white",
    s=75,
    zorder=7
)


# Moving links
physical_links, = physical_ax.plot(
    [],
    [],
    color="white",
    linewidth=2,
    alpha=0.75,
    zorder=5
)


# Moving bob 1
physical_bob1, = physical_ax.plot(
    [],
    [],
    marker="o",
    markersize=12,
    color="#5eead4",
    markeredgecolor="white",
    markeredgewidth=0.7,
    linestyle="None",
    zorder=7
)


# Moving bob 2
physical_bob2, = physical_ax.plot(
    [],
    [],
    marker="o",
    markersize=13,
    color="#f8fafc",
    markeredgecolor="#f97316",
    markeredgewidth=2,
    linestyle="None",
    zorder=7
)


# Full trajectory of bob 1
physical_trail1 = LineCollection(
    [],
    cmap=animation_color_map,
    norm=animation_color_normalization,
    alpha=0.35,
    zorder=2
)

physical_ax.add_collection(
    physical_trail1
)


# Full trajectory of bob 2
physical_trail2 = LineCollection(
    [],
    cmap=animation_color_map,
    norm=animation_color_normalization,
    alpha=0.9,
    zorder=3
)

physical_ax.add_collection(
    physical_trail2
)


# ==================================================
# Right-top panel: phase space of pendulum 1
# ==================================================

theta1_margin = (
    0.08 * np.ptp(theta1_values)
    + 0.05
)

omega1_margin = (
    0.08 * np.ptp(omega1_values)
    + 0.05
)

phase1_ax.set_xlim(
    np.min(theta1_values) - theta1_margin,
    np.max(theta1_values) + theta1_margin
)

phase1_ax.set_ylim(
    np.min(omega1_values) - omega1_margin,
    np.max(omega1_values) + omega1_margin
)

phase1_ax.set_title(
    "Pendulum 1 Phase Space",
    color="white",
    fontsize=13
)

phase1_ax.set_xlabel(
    "Angle $\\theta_1$ (rad)",
    color="white"
)

phase1_ax.set_ylabel(
    "Angular velocity $\\omega_1$ (rad/s)",
    color="white"
)


phase1_trail = LineCollection(
    [],
    cmap=animation_color_map,
    norm=animation_color_normalization,
    alpha=0.9,
    zorder=2
)

phase1_ax.add_collection(
    phase1_trail
)


phase1_ball, = phase1_ax.plot(
    [],
    [],
    marker="o",
    markersize=10,
    color="#f8fafc",
    markeredgecolor="#5eead4",
    markeredgewidth=2,
    linestyle="None",
    zorder=5
)


# ==================================================
# Right-bottom panel: phase space of pendulum 2
# ==================================================

theta2_margin = (
    0.08 * np.ptp(theta2_values)
    + 0.05
)

omega2_margin = (
    0.08 * np.ptp(omega2_values)
    + 0.05
)

phase2_ax.set_xlim(
    np.min(theta2_values) - theta2_margin,
    np.max(theta2_values) + theta2_margin
)

phase2_ax.set_ylim(
    np.min(omega2_values) - omega2_margin,
    np.max(omega2_values) + omega2_margin
)

phase2_ax.set_title(
    "Pendulum 2 Phase Space",
    color="white",
    fontsize=13
)

phase2_ax.set_xlabel(
    "Angle $\\theta_2$ (rad)",
    color="white"
)

phase2_ax.set_ylabel(
    "Angular velocity $\\omega_2$ (rad/s)",
    color="white"
)


phase2_trail = LineCollection(
    [],
    cmap=animation_color_map,
    norm=animation_color_normalization,
    alpha=0.9,
    zorder=2
)

phase2_ax.add_collection(
    phase2_trail
)


phase2_ball, = phase2_ax.plot(
    [],
    [],
    marker="o",
    markersize=10,
    color="#f8fafc",
    markeredgecolor="#f97316",
    markeredgewidth=2,
    linestyle="None",
    zorder=5
)


# ==================================================
# Shared time label
# ==================================================

animation_time_text = animation_fig.text(
    0.5,
    0.02,
    "",
    ha="center",
    color="white",
    fontsize=13
)


# ==================================================
# Helper function: convert points into line segments
# ==================================================

def create_animation_segments(
    x_values,
    y_values,
    frame
):
    """
    Convert all points from the beginning to the
    current frame into consecutive line segments.
    """

    current_points = np.column_stack(
        (
            x_values[:frame + 1],
            y_values[:frame + 1]
        )
    )

    if len(current_points) < 2:
        return np.empty(
            (0, 2, 2)
        )

    current_segments = np.stack(
        [
            current_points[:-1],
            current_points[1:]
        ],
        axis=1
    )

    return current_segments


# ==================================================
# Animation update function
# ==================================================

def update_synchronized_animation(frame):

    # ----------------------------------------------
    # 1. Update the physical double pendulum
    # ----------------------------------------------

    current_x1 = x1_values[frame]
    current_y1 = y1_values[frame]

    current_x2 = x2_values[frame]
    current_y2 = y2_values[frame]

    physical_links.set_data(
        [
            0,
            current_x1,
            current_x2
        ],
        [
            0,
            current_y1,
            current_y2
        ]
    )

    physical_bob1.set_data(
        [current_x1],
        [current_y1]
    )

    physical_bob2.set_data(
        [current_x2],
        [current_y2]
    )


    # ----------------------------------------------
    # 2. Update the complete trajectory of bob 1
    # ----------------------------------------------

    bob1_segments = create_animation_segments(
        x1_values,
        y1_values,
        frame
    )

    physical_trail1.set_segments(
        bob1_segments
    )


    # ----------------------------------------------
    # 3. Update the complete trajectory of bob 2
    # ----------------------------------------------

    bob2_segments = create_animation_segments(
        x2_values,
        y2_values,
        frame
    )

    physical_trail2.set_segments(
        bob2_segments
    )


    # ----------------------------------------------
    # 4. Update pendulum 1 phase space
    # ----------------------------------------------

    phase1_segments = create_animation_segments(
        theta1_values,
        omega1_values,
        frame
    )

    phase1_trail.set_segments(
        phase1_segments
    )

    phase1_ball.set_data(
        [theta1_values[frame]],
        [omega1_values[frame]]
    )


    # ----------------------------------------------
    # 5. Update pendulum 2 phase space
    # ----------------------------------------------

    phase2_segments = create_animation_segments(
        theta2_values,
        omega2_values,
        frame
    )

    phase2_trail.set_segments(
        phase2_segments
    )

    phase2_ball.set_data(
        [theta2_values[frame]],
        [omega2_values[frame]]
    )


    # ----------------------------------------------
    # 6. Apply color and line-width mappings
    # ----------------------------------------------

    if frame >= 1:

        current_times = t_values[:frame]

        # Time controls color in all three panels.
        physical_trail1.set_array(
            current_times
        )

        physical_trail2.set_array(
            current_times
        )

        phase1_trail.set_array(
            current_times
        )

        phase2_trail.set_array(
            current_times
        )


        # Cartesian speed controls physical
        # trajectory line width.
        bob1_line_widths = (
            0.4
            + 2.0
            * bob1_speed_values[:frame]
            / maximum_bob1_speed
        )

        bob2_line_widths = (
            0.5
            + 3.5
            * bob2_speed_values[:frame]
            / maximum_bob2_speed
        )

        physical_trail1.set_linewidths(
            bob1_line_widths
        )

        physical_trail2.set_linewidths(
            bob2_line_widths
        )


        # Angular speed controls phase-space
        # trajectory line width.
        phase1_line_widths = (
            0.5
            + 3.0
            * np.abs(
                omega1_values[:frame]
            )
            / maximum_omega1
        )

        phase2_line_widths = (
            0.5
            + 3.0
            * np.abs(
                omega2_values[:frame]
            )
            / maximum_omega2
        )

        phase1_trail.set_linewidths(
            phase1_line_widths
        )

        phase2_trail.set_linewidths(
            phase2_line_widths
        )


    # ----------------------------------------------
    # 7. Update the time label
    # ----------------------------------------------

    animation_time_text.set_text(
        f"Time: {t_values[frame]:.2f} s"
    )


    return (
        physical_links,
        physical_bob1,
        physical_bob2,
        physical_trail1,
        physical_trail2,
        phase1_trail,
        phase1_ball,
        phase2_trail,
        phase2_ball,
        animation_time_text
    )


# ==================================================
# Create the synchronized animation
# ==================================================

synchronized_double_pendulum_animation = FuncAnimation(
    animation_fig,
    update_synchronized_animation,
    frames=animation_frames,
    interval=animation_interval,
    blit=False,
    repeat=True
)


# Leave space for the title and time label
plt.subplots_adjust(
    top=0.90,
    bottom=0.09
)


# ==================================================
# Save the animation
# ==================================================

if save_animation:
    synchronized_double_pendulum_animation.save(
        "synchronized_double_pendulum.gif",
        writer=PillowWriter(
            fps=actual_fps
        ),
        dpi=110
    )


# ==================================================
# Display the animation
# ==================================================

if show_artwork:
    plt.show()
else:
    plt.close()
    
# -----------------------------
# 23. Sensitivity to initial conditions
# -----------------------------

# A very small difference between the two initial angles
sensitivity_epsilon = 0.001

# System A:
# theta1(0) = 2.000 rad
initial_state_A = np.array([
    2.0,
    0.0,
    -1.2,
    0.0
])

# System B:
# theta1(0) = 2.001 rad
initial_state_B = np.array([
    2.0 + sensitivity_epsilon,
    0.0,
    -1.2,
    0.0
])

# Create storage arrays for both simulations
sensitivity_values_A = np.zeros(
    (len(t_values), 4)
)

sensitivity_values_B = np.zeros(
    (len(t_values), 4)
)

# Store the two initial states
sensitivity_values_A[0] = initial_state_A
sensitivity_values_B[0] = initial_state_B

# Run both simulations using the same model and step size
for n in range(len(t_values) - 1):

    sensitivity_values_A[n + 1] = rk4_step(
        double_pendulum_model,
        t_values[n],
        sensitivity_values_A[n],
        h
    )

    sensitivity_values_B[n + 1] = rk4_step(
        double_pendulum_model,
        t_values[n],
        sensitivity_values_B[n],
        h
    )

# Extract angles from system A
theta1_A = sensitivity_values_A[:, 0]
theta2_A = sensitivity_values_A[:, 2]

# Extract angles from system B
theta1_B = sensitivity_values_B[:, 0]
theta2_B = sensitivity_values_B[:, 2]

# Convert both simulations to physical coordinates
x1_A, y1_A, x2_A, y2_A = angles_to_coordinates(
    theta1_A,
    theta2_A
)

x1_B, y1_B, x2_B, y2_B = angles_to_coordinates(
    theta1_B,
    theta2_B
)

# Distance between the second bobs
separation_distance = np.sqrt(
    (x2_A - x2_B) ** 2
    + (y2_A - y2_B) ** 2
)

# Print numerical sensitivity results
print("\nSensitivity experiment:")

print(
    "Initial angular difference:",
    sensitivity_epsilon,
    "rad"
)

print(
    "Initial bob-2 separation:",
    separation_distance[0],
    "m"
)

print(
    "Maximum bob-2 separation:",
    np.max(separation_distance),
    "m"
)

print(
    "Final bob-2 separation:",
    separation_distance[-1],
    "m"
)

# -----------------------------
# Sensitivity comparison figure
# -----------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)

# Left: physical trajectories of bob 2
axes[0].plot(
    x2_A,
    y2_A,
    color="#22d3ee",
    linewidth=1.3,
    alpha=0.85,
    label=r"System A: $\theta_1(0)=2.000$"
)

axes[0].plot(
    x2_B,
    y2_B,
    color="#f97316",
    linewidth=1.3,
    alpha=0.85,
    label=r"System B: $\theta_1(0)=2.001$"
)

axes[0].scatter(
    0,
    0,
    color="white",
    s=80,
    zorder=5,
    label="Pivot"
)

maximum_reach = L1 + L2

axes[0].set_xlim(
    -1.1 * maximum_reach,
    1.1 * maximum_reach
)

axes[0].set_ylim(
    -1.1 * maximum_reach,
    1.1 * maximum_reach
)

axes[0].set_aspect(
    "equal",
    adjustable="box"
)

axes[0].set_xlabel(
    "Horizontal position x (m)"
)

axes[0].set_ylabel(
    "Vertical position y (m)"
)

axes[0].set_title(
    "Trajectories from Nearly Identical Initial Conditions"
)

axes[0].grid(
    True,
    alpha=0.2
)

axes[0].legend(
    fontsize=9
)

# Right: distance between the second bobs
axes[1].semilogy(
    t_values,
    np.maximum(
        separation_distance,
        1e-12
    ),
    color="#a855f7",
    linewidth=1.5
)

axes[1].set_xlabel(
    "Time (s)"
)

axes[1].set_ylabel(
    "Separation distance D(t) (m)"
)

axes[1].set_title(
    "Growth of Trajectory Separation"
)

axes[1].grid(
    True,
    alpha=0.2,
    which="both"
)

fig.suptitle(
    "Sensitivity to Initial Conditions in the Double Pendulum",
    fontsize=16
)

fig.tight_layout()

if save_figures:

    plt.savefig(
        figure_directory
        / "double_pendulum_sensitivity.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

if show_analysis_figures:

    plt.show()

else:

    plt.close()