import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter

# -----------------------------
# 1. Model parameters
# -----------------------------
g = 9.81          # gravitational acceleration (m/s^2)
L = 1.0           # pendulum length (m)
m = 1.0           # pendulum mass (kg)

# -----------------------------
# 2. Initial conditions
# -----------------------------
theta_initial = 1.5   # initial angle (radians)
omega_initial = 0.0   # initial angular velocity (radians/s)

angle_label = str(theta_initial).replace(".", "_")

# True: save figures
# False: display figures without saving
save_figures = False
save_animation = False

show_analysis_figures = False
show_artwork = True

# -----------------------------
# 3. Simulation settings
# -----------------------------
t_initial = 0.0
t_final = 13.0
h = 0.01

t_values = np.arange(t_initial, t_final + h, h)

# Each row stores [theta, omega]
y_values = np.zeros((len(t_values), 2))

# Store the initial state
y_values[0] = [theta_initial, omega_initial]


# -----------------------------
# 4. Differential equation
# -----------------------------
def pendulum_model(t, y):
    theta = y[0]
    omega = y[1]

    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)

    return np.array([dtheta_dt, domega_dt])


# -----------------------------
# 5. One RK4 step
# -----------------------------
def rk4_step(function, t, y, h):
    k1 = function(t, y)

    k2 = function(
        t + h / 2,
        y + (h / 2) * k1
    )

    k3 = function(
        t + h / 2,
        y + (h / 2) * k2
    )

    k4 = function(
        t + h,
        y + h * k3
    )

    y_next = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return y_next


# -----------------------------
# 6. Run the simulation
# -----------------------------
for n in range(len(t_values) - 1):
    y_values[n + 1] = rk4_step(
        pendulum_model,
        t_values[n],
        y_values[n],
        h
    )


# Separate theta and omega
theta_values = y_values[:, 0]
omega_values = y_values[:, 1]


# -----------------------------
# 7. Plot theta against time
# -----------------------------
plt.figure(figsize=(9, 5))

plt.plot(t_values, theta_values, color="navy")

plt.xlabel("Time (s)")
plt.ylabel("Angle θ (rad)")
plt.title("Nonlinear Simple Pendulum: Angle versus Time")
plt.grid(True, alpha=0.3)

if save_figures:
    plt.savefig(
        f"angle_time_theta_{angle_label}.png",
        dpi=300,
        bbox_inches="tight"
    )

if show_analysis_figures:
    plt.show()
else:
    plt.close()

# -----------------------------
# 8. Plot omega against time
# -----------------------------
plt.figure(figsize=(9, 5))

plt.plot(t_values, omega_values, color="darkred")

plt.xlabel("Time (s)")
plt.ylabel("Angular velocity ω (rad/s)")
plt.title("Nonlinear Simple Pendulum: Angular Velocity versus Time")
plt.grid(True, alpha=0.3)

if save_figures:
    plt.savefig(
        f"omega_time_theta_{angle_label}.png",
        dpi=300,
        bbox_inches="tight"
    )

if show_analysis_figures:
    plt.show()
else:
    plt.close()

# -----------------------------
# 9. Compare nonlinear and linear models
# -----------------------------

# Natural angular frequency
Omega = np.sqrt(g / L)

# Analytical solution of the linear small-angle model
theta_linear = (
    theta_initial * np.cos(Omega * t_values)
    + (omega_initial / Omega) * np.sin(Omega * t_values)
)

plt.figure(figsize=(9, 5))

plt.plot(
    t_values,
    theta_values,
    color="navy",
    label="Nonlinear numerical solution"
)

plt.plot(
    t_values,
    theta_linear,
    color="orange",
    linestyle="--",
    label="Linear small-angle solution"
)

plt.xlabel("Time (s)")
plt.ylabel("Angle θ (rad)")
plt.title("Linear and Nonlinear Simple Pendulum Models")
plt.grid(True, alpha=0.3)
plt.legend()

if save_figures:
    plt.savefig(
        f"linear_nonlinear_comparison_theta_{angle_label}.png",
        dpi=300,
        bbox_inches="tight"
    )

if show_analysis_figures:
    plt.show()
else:
    plt.close()

# -----------------------------
# 10. Phase-space plot
# -----------------------------
plt.figure(figsize=(7, 6))

plt.plot(
    theta_values,
    omega_values,
    color="purple"
)

# Mark the initial state
plt.scatter(
    theta_values[0],
    omega_values[0],
    color="red",
    label="Initial state",
    zorder=3
)

plt.xlabel("Angle θ (rad)")
plt.ylabel("Angular velocity ω (rad/s)")
plt.title("Phase-Space Plot of the Nonlinear Simple Pendulum")
plt.grid(True, alpha=0.3)
plt.legend()

if save_figures:
    plt.savefig(
        f"phase_space_theta_{angle_label}.png",
        dpi=300,
        bbox_inches="tight"
    )

if show_analysis_figures:
    plt.show()
else:
    plt.close()

# -----------------------------
# 11. Energy conservation
# -----------------------------

kinetic_energy = 0.5 * m * (L ** 2) * (omega_values ** 2)

potential_energy = (
    m * g * L * (1 - np.cos(theta_values))
)

total_energy = kinetic_energy + potential_energy


plt.figure(figsize=(9, 5))

plt.plot(
    t_values,
    kinetic_energy,
    color="darkred",
    label="Kinetic energy"
)

plt.plot(
    t_values,
    potential_energy,
    color="darkgreen",
    label="Potential energy"
)

plt.plot(
    t_values,
    total_energy,
    color="black",
    linestyle="--",
    label="Total energy"
)

plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Energy of the Nonlinear Simple Pendulum")
plt.grid(True, alpha=0.3)
plt.legend()

if save_figures:
    plt.savefig(
        f"energy_theta_{angle_label}.png",
        dpi=300,
        bbox_inches="tight"
    )

if show_analysis_figures:
    plt.show()
else:
    plt.close()

initial_energy = total_energy[0]

maximum_energy_error = np.max(
    np.abs(total_energy - initial_energy)
)

relative_energy_error = maximum_energy_error / abs(initial_energy)

print("Initial total energy:", initial_energy)
print("Maximum energy error:", maximum_energy_error)
print("Relative energy error:", relative_energy_error)

# -----------------------------
# 12. Step-size comparison
# -----------------------------

def run_simulation(step_size):
    time_points = np.arange(
        t_initial,
        t_final + step_size,
        step_size
    )

    states = np.zeros((len(time_points), 2))
    states[0] = [theta_initial, omega_initial]

    for n in range(len(time_points) - 1):
        states[n + 1] = rk4_step(
            pendulum_model,
            time_points[n],
            states[n],
            step_size
        )

    theta_result = states[:, 0]
    omega_result = states[:, 1]

    kinetic_result = (
        0.5 * m * (L ** 2) * (omega_result ** 2)
    )

    potential_result = (
        m * g * L * (1 - np.cos(theta_result))
    )

    energy_result = kinetic_result + potential_result

    maximum_error = np.max(
        np.abs(energy_result - energy_result[0])
    )

    relative_error = maximum_error / abs(energy_result[0])

    return relative_error


step_sizes = [0.1, 0.05, 0.02, 0.01]
energy_errors = []

for step_size in step_sizes:
    error = run_simulation(step_size)
    energy_errors.append(error)

    print(
        f"h = {step_size:.3f}, "
        f"relative energy error = {error:.6e}"
    )


plt.figure(figsize=(8, 5))

plt.loglog(
    step_sizes,
    energy_errors,
    marker="o",
    color="purple"
)

plt.xlabel("Step size h (s)")
plt.ylabel("Maximum relative energy error")
plt.title("Effect of Step Size on RK4 Energy Error")
plt.grid(True, which="both", alpha=0.3)

if save_figures:
    plt.savefig(
        f"step_size_comparison_theta_{angle_label}.png",
        dpi=300,
        bbox_inches="tight"
    )

if show_analysis_figures:
    plt.show()
else:
    plt.close()

# -----------------------------
# 13. Effect of initial angle on period
# -----------------------------

def simulate_for_angle(theta0, step_size=0.01):
    time_points = np.arange(
        t_initial,
        t_final + step_size,
        step_size
    )

    states = np.zeros((len(time_points), 2))
    states[0] = [theta0, 0.0]

    for n in range(len(time_points) - 1):
        states[n + 1] = rk4_step(
            pendulum_model,
            time_points[n],
            states[n],
            step_size
        )

    return time_points, states[:, 0], states[:, 1]


def estimate_period(time_points, theta_result):
    crossing_times = []

    # Find crossings from positive theta to negative theta
    for n in range(len(theta_result) - 1):
        if theta_result[n] > 0 and theta_result[n + 1] <= 0:

            # Linear interpolation for a more accurate crossing time
            fraction = (
                theta_result[n]
                / (theta_result[n] - theta_result[n + 1])
            )

            crossing_time = (
                time_points[n]
                + fraction * (time_points[n + 1] - time_points[n])
            )

            crossing_times.append(crossing_time)

    # Consecutive crossings in the same direction are one period apart
    measured_periods = np.diff(crossing_times)

    return np.mean(measured_periods)


initial_angles = [0.1, 0.5, 1.0, 1.5, 2.0]
numerical_periods = []

for theta0 in initial_angles:
    time_result, theta_result, omega_result = simulate_for_angle(theta0)

    period = estimate_period(time_result, theta_result)
    numerical_periods.append(period)

    print(
        f"Initial angle = {theta0:.1f} rad, "
        f"numerical period = {period:.6f} s"
    )


# Small-angle theoretical period
linear_period = 2 * np.pi * np.sqrt(L / g)

plt.figure(figsize=(8, 5))

plt.plot(
    initial_angles,
    numerical_periods,
    marker="o",
    color="navy",
    label="Nonlinear numerical period"
)

plt.axhline(
    linear_period,
    color="orange",
    linestyle="--",
    label="Linear small-angle period"
)

plt.xlabel("Initial angle θ₀ (rad)")
plt.ylabel("Period T (s)")
plt.title("Effect of Initial Angle on Pendulum Period")
plt.grid(True, alpha=0.3)
plt.legend()

if save_figures:
    plt.savefig(
        "period_vs_initial_angle.png",
        dpi=300,
        bbox_inches="tight"
    )

if show_analysis_figures:
    plt.show()
else:
    plt.close()

# -----------------------------
# 14. Damped pendulum comparison
# -----------------------------

def damped_pendulum_model(t, y, gamma):
    theta = y[0]
    omega = y[1]

    dtheta_dt = omega
    domega_dt = (
        -gamma * omega
        - (g / L) * np.sin(theta)
    )

    return np.array([dtheta_dt, domega_dt])


def simulate_damped_pendulum(gamma, step_size=0.01):
    time_points = np.arange(
        t_initial,
        t_final + step_size,
        step_size
    )

    states = np.zeros((len(time_points), 2))
    states[0] = [theta_initial, omega_initial]

    # Create a two-input function for the existing RK4 code
    def model(t, y):
        return damped_pendulum_model(t, y, gamma)

    for n in range(len(time_points) - 1):
        states[n + 1] = rk4_step(
            model,
            time_points[n],
            states[n],
            step_size
        )

    return time_points, states[:, 0], states[:, 1]


damping_values = [0.0, 0.2, 0.6]
colors = ["navy", "purple", "darkred"]

plt.figure(figsize=(8, 6))

for gamma, color in zip(damping_values, colors):
    time_result, theta_result, omega_result = (
        simulate_damped_pendulum(gamma)
    )

    plt.plot(
        theta_result,
        omega_result,
        color=color,
        label=fr"$\gamma={gamma}$"
    )

plt.xlabel("Angle θ (rad)")
plt.ylabel("Angular velocity ω (rad/s)")
plt.title("Effect of Damping on Pendulum Phase-Space Trajectories")
plt.grid(True, alpha=0.3)
plt.legend()

if save_figures:
    plt.savefig(
        f"damping_phase_space_theta_{angle_label}.png",
        dpi=300,
        bbox_inches="tight"
    )

if show_analysis_figures:
    plt.show()
else:
    plt.close()

# -----------------------------
# 15. Generative phase-space artwork
# -----------------------------

art_gamma = 0.08

art_time, art_theta, art_omega = simulate_damped_pendulum(
    art_gamma,
    step_size=0.01
)

# Combine theta and omega into phase-space points
points = np.column_stack((art_theta, art_omega))

# Convert consecutive points into line segments
segments = np.stack(
    [points[:-1], points[1:]],
    axis=1
)

# Use angular speed to control line width
speed = np.abs(art_omega[:-1])

normalized_speed = speed / np.max(speed)

line_widths = 0.5 + 3.0 * normalized_speed


# Create the artwork
fig, ax = plt.subplots(figsize=(8, 8))

fig.patch.set_facecolor("#070711")
ax.set_facecolor("#070711")

trajectory = LineCollection(
    segments,
    cmap="turbo",
    linewidths=line_widths,
    alpha=0.9
)

# Time controls color
trajectory.set_array(art_time[:-1])

ax.add_collection(trajectory)

ax.set_xlim(
    np.min(art_theta) - 0.15,
    np.max(art_theta) + 0.15
)

ax.set_ylim(
    np.min(art_omega) - 0.25,
    np.max(art_omega) + 0.25
)

# Remove axes for an artwork-style output
ax.axis("off")

if save_figures:
    plt.savefig(
        f"generative_phase_space_gamma_{str(art_gamma).replace('.', '_')}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

if show_artwork:
    plt.show()
else:
    plt.close()
    
# -----------------------------
# 16. Physical trajectory of damped pendulum
# -----------------------------

physical_gamma = 0.2

physical_time, physical_theta, physical_omega = (
    simulate_damped_pendulum(
        physical_gamma,
        step_size=0.01
    )
)

# Convert angular position to physical coordinates
physical_x = L * np.sin(physical_theta)
physical_y = -L * np.cos(physical_theta)

# Find turning points, where omega changes sign
turning_indices = np.where(
    np.sign(physical_omega[:-1])
    != np.sign(physical_omega[1:])
)[0] + 1

# Include the initial position
turning_indices = np.insert(turning_indices, 0, 0)


fig, ax = plt.subplots(figsize=(8, 8))

fig.patch.set_facecolor("#070711")
ax.set_facecolor("#070711")

# Draw the complete path of the pendulum bob
ax.plot(
    physical_x,
    physical_y,
    color="#5eead4",
    linewidth=1.5,
    alpha=0.55,
    label="Bob trajectory"
)

# Mark the successive turning points
turning_points = ax.scatter(
    physical_x[turning_indices],
    physical_y[turning_indices],
    c=physical_time[turning_indices],
    cmap="plasma",
    s=45,
    zorder=3,
    label="Turning points"
)

# Draw the initial pendulum rod
ax.plot(
    [0, physical_x[0]],
    [0, physical_y[0]],
    color="white",
    linewidth=1.2,
    alpha=0.7
)

# Draw the pivot
ax.scatter(
    0,
    0,
    color="white",
    s=70,
    zorder=4,
    label="Pivot"
)

# Draw the initial bob
ax.scatter(
    physical_x[0],
    physical_y[0],
    color="#5eead4",
    s=90,
    zorder=4
)

colorbar = plt.colorbar(
    turning_points,
    ax=ax,
    fraction=0.046,
    pad=0.04
)

colorbar.set_label(
    "Time (s)",
    color="white"
)

colorbar.ax.tick_params(colors="white")

ax.set_xlim(-1.1 * L, 1.1 * L)
ax.set_ylim(-1.1 * L, 0.15 * L)

# Essential for a physically correct circular trajectory
ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("Horizontal position x (m)", color="white")
ax.set_ylabel("Vertical position y (m)", color="white")
ax.set_title(
    "Physical Trajectory of a Damped Simple Pendulum",
    color="white"
)

ax.tick_params(colors="white")
ax.grid(True, alpha=0.12)
ax.legend(facecolor="#11111f", labelcolor="white")

if save_figures:
    plt.savefig(
        f"physical_trajectory_gamma_{str(physical_gamma).replace('.', '_')}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

if show_artwork:
    plt.show()
else:
    plt.close()
    
# -----------------------------
# 17. Synchronized physical-space and phase-space animation
# -----------------------------

save_animation = False

# Both panels must use the same damping value and simulation.
animation_gamma = 0.08

animation_time, animation_theta, animation_omega = (
    simulate_damped_pendulum(
        animation_gamma,
        step_size=0.01
    )
)

# -----------------------------
# Physical-space coordinates
# -----------------------------

physical_x = L * np.sin(animation_theta)
physical_y = -L * np.cos(animation_theta)

# -----------------------------
# Abstract phase-space coordinates
# -----------------------------

abstract_x = animation_theta
abstract_y = animation_omega

# Display every third numerical point.
frame_step = 3

animation_frames = range(
    0,
    len(animation_time),
    frame_step
)

# Create two panels.
fig, (ax_physical, ax_abstract) = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)

fig.patch.set_facecolor("#070711")

ax_physical.set_facecolor("#070711")
ax_abstract.set_facecolor("#070711")

# Main title
fig.suptitle(
    "One Pendulum in Physical Space and Abstract Phase Space",
    color="white",
    fontsize=16
)

# ==================================================
# Left panel: physical pendulum
# ==================================================

ax_physical.set_xlim(
    -1.15 * L,
    1.15 * L
)

ax_physical.set_ylim(
    -1.15 * L,
    0.20 * L
)

ax_physical.set_aspect(
    "equal",
    adjustable="box"
)

ax_physical.set_title(
    "Physical Space: $(x,y)$",
    color="white",
    fontsize=14
)

ax_physical.set_xlabel(
    "$x=L\\sin(\\theta)$",
    color="white"
)

ax_physical.set_ylabel(
    "$y=-L\\cos(\\theta)$",
    color="white"
)

ax_physical.tick_params(colors="white")
ax_physical.grid(True, alpha=0.12)

# Fixed pivot
ax_physical.scatter(
    0,
    0,
    color="white",
    s=70,
    zorder=5
)

# Moving pendulum rod
physical_rod, = ax_physical.plot(
    [],
    [],
    color="white",
    linewidth=2,
    alpha=0.75,
    zorder=3
)

# Moving physical bob
physical_bob, = ax_physical.plot(
    [],
    [],
    marker="o",
    markersize=13,
    color="#5eead4",
    markeredgecolor="white",
    markeredgewidth=0.6,
    linestyle="None",
    zorder=5
)

# Growing physical trajectory
physical_trail = LineCollection(
    [],
    zorder=2
)

ax_physical.add_collection(physical_trail)

# ==================================================
# Right panel: abstract phase space
# ==================================================

abstract_x_margin = 0.15
abstract_y_margin = 0.25

ax_abstract.set_xlim(
    np.min(abstract_x) - abstract_x_margin,
    np.max(abstract_x) + abstract_x_margin
)

ax_abstract.set_ylim(
    np.min(abstract_y) - abstract_y_margin,
    np.max(abstract_y) + abstract_y_margin
)

ax_abstract.set_title(
    "Abstract Phase Space: $(\\theta,\\omega)$",
    color="white",
    fontsize=14
)

ax_abstract.set_xlabel(
    "Angle $\\theta$ (rad)",
    color="white"
)

ax_abstract.set_ylabel(
    "Angular velocity $\\omega$ (rad/s)",
    color="white"
)

ax_abstract.tick_params(colors="white")
ax_abstract.grid(True, alpha=0.12)

# Moving abstract-space ball
abstract_bob, = ax_abstract.plot(
    [],
    [],
    marker="o",
    markersize=11,
    color="#f8fafc",
    markeredgecolor="#5eead4",
    markeredgewidth=2,
    linestyle="None",
    zorder=5
)

# Growing abstract generative trajectory
abstract_trail = LineCollection(
    [],
    zorder=2
)

ax_abstract.add_collection(abstract_trail)

# Time label shared by the two panels
time_text = fig.text(
    0.5,
    0.03,
    "",
    ha="center",
    color="white",
    fontsize=13
)

# ==================================================
# Color and line-width mappings
# ==================================================

color_map = plt.colormaps["turbo"]

color_normalization = plt.Normalize(
    vmin=animation_time[0],
    vmax=animation_time[-1]
)

maximum_speed = np.max(
    np.abs(animation_omega)
)

# Prevent division by zero.
if maximum_speed == 0:
    maximum_speed = 1.0


def create_segments(x_values, y_values):
    """
    Convert coordinate points into line segments
    for use in a LineCollection.
    """

    points = np.column_stack(
        (x_values, y_values)
    )

    if len(points) < 2:
        return np.empty((0, 2, 2))

    return np.stack(
        [points[:-1], points[1:]],
        axis=1
    )


def update_animation(frame):
    # ----------------------------------------------
    # Update the physical pendulum
    # ----------------------------------------------

    current_physical_x = physical_x[frame]
    current_physical_y = physical_y[frame]

    physical_rod.set_data(
        [0, current_physical_x],
        [0, current_physical_y]
    )

    physical_bob.set_data(
        [current_physical_x],
        [current_physical_y]
    )

    # Keep the complete physical path drawn so far.
    physical_segments = create_segments(
        physical_x[:frame + 1],
        physical_y[:frame + 1]
    )

    physical_trail.set_segments(
        physical_segments
    )

    # ----------------------------------------------
    # Update the abstract phase-space ball
    # ----------------------------------------------

    abstract_bob.set_data(
        [abstract_x[frame]],
        [abstract_y[frame]]
    )

    # Keep the complete abstract path drawn so far.
    abstract_segments = create_segments(
        abstract_x[:frame + 1],
        abstract_y[:frame + 1]
    )

    abstract_trail.set_segments(
        abstract_segments
    )

    # ----------------------------------------------
    # Apply identical visual mappings
    # ----------------------------------------------

    if frame >= 1:
        segment_times = animation_time[:frame]

        segment_colors = color_map(
            color_normalization(segment_times)
        )

        # Time controls color in both panels.
        physical_trail.set_color(
            segment_colors
        )

        abstract_trail.set_color(
            segment_colors
        )

        segment_speed = np.abs(
            animation_omega[:frame]
        )

        # Angular speed controls line width.
        segment_widths = (
            0.6
            + 3.0 * segment_speed / maximum_speed
        )

        physical_trail.set_linewidths(
            segment_widths
        )

        abstract_trail.set_linewidths(
            segment_widths
        )

    time_text.set_text(
        f"Time: {animation_time[frame]:.2f} s"
    )

    return (
        physical_rod,
        physical_bob,
        physical_trail,
        abstract_bob,
        abstract_trail,
        time_text
    )


synchronized_animation = FuncAnimation(
    fig,
    update_animation,
    frames=animation_frames,
    interval=0.01 * frame_step * 1000,
    blit=False,
    repeat=True
)

plt.tight_layout(
    rect=[0, 0.06, 1, 0.93]
)

if save_animation:
    synchronized_animation.save(
        "physical_and_abstract_pendulum.gif",
        writer=PillowWriter(fps=30),
        dpi=120
    )

if show_artwork:
    plt.show()
else:
    plt.close()
    
