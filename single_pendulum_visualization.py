# ============================================================
# Section 1: Imports
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


# ============================================================
# Section 2: Helper function
# ============================================================

def create_segments(
    x_values,
    y_values
):
    """
    Convert consecutive coordinate points into line segments.
    """

    points = np.column_stack(
        (
            x_values,
            y_values
        )
    )

    if len(points) < 2:
        return np.empty(
            (0, 2, 2)
        )

    return np.stack(
        (
            points[:-1],
            points[1:]
        ),
        axis=1
    )
    
    
# ============================================================
# Section 3: Static single-pendulum visualization
# ============================================================

def create_single_pendulum_figure(
    results,
    color_map_name="turbo",
    minimum_line_width=0.6,
    maximum_line_width=3.6,
    minimum_alpha=0.20,
    maximum_alpha=1.00
):
    """
    Create a static physical-space and phase-space figure.

    Time controls colour.
    Absolute angular velocity controls line thickness.
    """

    available_color_maps = [
        "turbo",
        "plasma",
        "viridis",
        "inferno",
        "magma",
        "cividis"
    ]

    if color_map_name not in available_color_maps:
        raise ValueError(
            "Unsupported colour map: "
            + color_map_name
        )

    time_values = results["time"]
    theta_values = results["theta"]
    omega_values = results["omega"]

    x_values = results["x"]
    y_values = results["y"]

    length = results["parameters"]["length"]

    color_map = plt.colormaps[
        color_map_name
    ]

    color_normalization = Normalize(
        vmin=time_values[0],
        vmax=time_values[-1]
    )

    maximum_speed = np.max(
        np.abs(omega_values)
    )

    if maximum_speed == 0:
        maximum_speed = 1.0

    segment_widths = (
        minimum_line_width
        + (maximum_line_width - minimum_line_width)
        * np.abs(omega_values[:-1])
        / maximum_speed
    )

    parameters = results["parameters"]
    critical_energy = (
        2.0
        * parameters["mass"]
        * parameters["g"]
        * parameters["length"]
    )
    energy_ratio = max(
        0.0,
        results["initial_total_energy"] / critical_energy
    )
    energy_intensity = energy_ratio / (1.0 + energy_ratio)
    trajectory_alpha = (
        minimum_alpha
        + (maximum_alpha - minimum_alpha) * energy_intensity
    )

    physical_segments = create_segments(
        x_values,
        y_values
    )

    phase_segments = create_segments(
        theta_values,
        omega_values
    )

    physical_trajectory = LineCollection(
        physical_segments,
        cmap=color_map,
        norm=color_normalization,
        linewidths=segment_widths,
        alpha=trajectory_alpha
    )

    physical_trajectory.set_array(
        time_values[:-1]
    )

    phase_trajectory = LineCollection(
        phase_segments,
        cmap=color_map,
        norm=color_normalization,
        linewidths=segment_widths,
        alpha=trajectory_alpha
    )

    phase_trajectory.set_array(
        time_values[:-1]
    )

    plt.style.use(
        "dark_background"
    )

    figure, (
        physical_axis,
        phase_axis
    ) = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    figure.patch.set_facecolor(
        "#070711"
    )

    physical_axis.set_facecolor(
        "#070711"
    )

    phase_axis.set_facecolor(
        "#070711"
    )

    # --------------------------------------------------------
    # Left panel: complete physical trajectory
    # --------------------------------------------------------

    physical_axis.add_collection(
        physical_trajectory
    )

    final_x = x_values[-1]
    final_y = y_values[-1]

    physical_axis.plot(
        [0.0, final_x],
        [0.0, final_y],
        color="white",
        linewidth=2.0,
        alpha=0.75,
        zorder=4
    )

    physical_axis.scatter(
        0.0,
        0.0,
        color="white",
        s=70,
        zorder=6
    )

    physical_axis.scatter(
        final_x,
        final_y,
        color="#5eead4",
        edgecolor="white",
        linewidth=0.8,
        s=100,
        zorder=6
    )

    physical_limit = 1.15 * length

    physical_axis.set_xlim(
        -physical_limit,
        physical_limit
    )

    physical_axis.set_ylim(
        -physical_limit,
        physical_limit
    )

    physical_axis.set_aspect(
        "equal",
        adjustable="box"
    )

    physical_axis.set_title(
        "Physical Space: $(x,y)$"
    )

    physical_axis.set_xlabel(
        r"$x=L\sin(\theta)$"
    )

    physical_axis.set_ylabel(
        r"$y=-L\cos(\theta)$"
    )

    physical_axis.grid(
        True,
        alpha=0.12
    )

    # --------------------------------------------------------
    # Right panel: complete phase-space trajectory
    # --------------------------------------------------------

    phase_axis.add_collection(
        phase_trajectory
    )

    phase_axis.scatter(
        theta_values[-1],
        omega_values[-1],
        color="#f8fafc",
        edgecolor="#5eead4",
        linewidth=2.0,
        s=80,
        zorder=6
    )

    theta_margin = (
        0.08 * np.ptp(theta_values)
        + 0.05
    )

    omega_margin = (
        0.08 * np.ptp(omega_values)
        + 0.05
    )

    phase_axis.set_xlim(
        np.min(theta_values) - theta_margin,
        np.max(theta_values) + theta_margin
    )

    phase_axis.set_ylim(
        np.min(omega_values) - omega_margin,
        np.max(omega_values) + omega_margin
    )

    phase_axis.set_title(
        r"Abstract Phase Space: $(\theta,\omega)$"
    )

    phase_axis.set_xlabel(
        r"Angle $\theta$ (rad)"
    )

    phase_axis.set_ylabel(
        r"Angular velocity $\omega$ (rad/s)"
    )

    phase_axis.axhline(
        0.0,
        color="white",
        linewidth=0.8,
        alpha=0.3
    )

    phase_axis.axvline(
        0.0,
        color="white",
        linewidth=0.8,
        alpha=0.3
    )

    phase_axis.grid(
        True,
        alpha=0.12
    )

    figure.suptitle(
        "Single Pendulum: Physical and Abstract Trajectories",
        fontsize=16
    )

    colour_bar = figure.colorbar(
        phase_trajectory,
        ax=[
            physical_axis,
            phase_axis
        ],
        fraction=0.025,
        pad=0.03
    )

    colour_bar.set_label(
        "Time (s)"
    )

    return (
        figure,
        physical_axis,
        phase_axis
    )

# ============================================================
# Section 4: Synchronized single-pendulum animation
# ============================================================

def create_single_pendulum_animation(
    results,
    frame_skip=3,
    interval=None,
    color_map_name="turbo",
    minimum_line_width=0.6,
    maximum_line_width=3.6,
    minimum_alpha=0.20,
    maximum_alpha=1.00
):
    """
    Create synchronized physical-space and phase-space
    animation for a single pendulum.

    Time controls colour.
    Absolute angular velocity controls line thickness.
    """

    if frame_skip < 1:
        raise ValueError(
            "frame_skip must be at least 1."
        )

    available_color_maps = [
        "turbo",
        "plasma",
        "viridis",
        "inferno",
        "magma",
        "cividis"
    ]

    if color_map_name not in available_color_maps:
        raise ValueError(
            "Unsupported colour map: "
            + color_map_name
        )

    time_values = results["time"]
    theta_values = results["theta"]
    omega_values = results["omega"]

    x_values = results["x"]
    y_values = results["y"]

    length = results["parameters"]["length"]

    if interval is None:
        interval = (
            results["step_size"]
            * frame_skip
            * 1000
        )

    color_map = plt.colormaps[
        color_map_name
    ]

    color_normalization = Normalize(
        vmin=time_values[0],
        vmax=time_values[-1]
    )

    maximum_speed = np.max(
        np.abs(omega_values)
    )

    if maximum_speed == 0:
        maximum_speed = 1.0

    # Create the full possible paths.
    physical_segments_all = create_segments(
        x_values,
        y_values
    )

    phase_segments_all = create_segments(
        theta_values,
        omega_values
    )

    segment_widths = (
        minimum_line_width
        + (maximum_line_width - minimum_line_width)
        * np.abs(omega_values[:-1])
        / maximum_speed
    )

    parameters = results["parameters"]
    critical_energy = (
        2.0
        * parameters["mass"]
        * parameters["g"]
        * parameters["length"]
    )
    energy_ratio = max(
        0.0,
        results["initial_total_energy"] / critical_energy
    )
    energy_intensity = energy_ratio / (1.0 + energy_ratio)
    trajectory_alpha = (
        minimum_alpha
        + (maximum_alpha - minimum_alpha) * energy_intensity
    )

    # Create the two-panel figure.
    plt.style.use(
        "dark_background"
    )

    figure, (
        physical_axis,
        phase_axis
    ) = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    figure.patch.set_facecolor(
        "#070711"
    )

    physical_axis.set_facecolor(
        "#070711"
    )

    phase_axis.set_facecolor(
        "#070711"
    )

    figure.suptitle(
        "One Pendulum in Physical Space and Abstract Phase Space",
        color="white",
        fontsize=16
    )

    # --------------------------------------------------------
    # Left panel: physical pendulum
    # --------------------------------------------------------

    physical_limit = 1.15 * length

    physical_axis.set_xlim(
        -physical_limit,
        physical_limit
    )

    physical_axis.set_ylim(
        -physical_limit,
        physical_limit
    )

    physical_axis.set_aspect(
        "equal",
        adjustable="box"
    )

    physical_axis.set_title(
        "Physical Space: $(x,y)$",
        fontsize=14
    )

    physical_axis.set_xlabel(
        r"$x=L\sin(\theta)$"
    )

    physical_axis.set_ylabel(
        r"$y=-L\cos(\theta)$"
    )

    physical_axis.grid(
        True,
        alpha=0.12
    )

    physical_axis.scatter(
        0.0,
        0.0,
        color="white",
        s=70,
        zorder=6
    )

    physical_rod, = physical_axis.plot(
        [],
        [],
        color="white",
        linewidth=2.0,
        alpha=0.75,
        zorder=4
    )

    physical_bob, = physical_axis.plot(
        [],
        [],
        marker="o",
        markersize=13,
        color="#5eead4",
        markeredgecolor="white",
        markeredgewidth=0.8,
        linestyle="None",
        zorder=6
    )

    physical_trail = LineCollection(
        [],
        cmap=color_map,
        norm=color_normalization,
        alpha=trajectory_alpha,
        zorder=2
    )

    physical_axis.add_collection(
        physical_trail
    )

    # --------------------------------------------------------
    # Right panel: abstract phase space
    # --------------------------------------------------------

    theta_margin = (
        0.08 * np.ptp(theta_values)
        + 0.05
    )

    omega_margin = (
        0.08 * np.ptp(omega_values)
        + 0.05
    )

    phase_axis.set_xlim(
        np.min(theta_values) - theta_margin,
        np.max(theta_values) + theta_margin
    )

    phase_axis.set_ylim(
        np.min(omega_values) - omega_margin,
        np.max(omega_values) + omega_margin
    )

    phase_axis.set_title(
        r"Abstract Phase Space: $(\theta,\omega)$",
        fontsize=14
    )

    phase_axis.set_xlabel(
        r"Angle $\theta$ (rad)"
    )

    phase_axis.set_ylabel(
        r"Angular velocity $\omega$ (rad/s)"
    )

    phase_axis.grid(
        True,
        alpha=0.12
    )

    phase_axis.axhline(
        0.0,
        color="white",
        linewidth=0.8,
        alpha=0.3
    )

    phase_axis.axvline(
        0.0,
        color="white",
        linewidth=0.8,
        alpha=0.3
    )

    phase_bob, = phase_axis.plot(
        [],
        [],
        marker="o",
        markersize=11,
        color="#f8fafc",
        markeredgecolor="#5eead4",
        markeredgewidth=2.0,
        linestyle="None",
        zorder=6
    )

    phase_trail = LineCollection(
        [],
        cmap=color_map,
        norm=color_normalization,
        alpha=trajectory_alpha,
        zorder=2
    )

    phase_axis.add_collection(
        phase_trail
    )

    time_text = figure.text(
        0.5,
        0.03,
        "",
        ha="center",
        color="white",
        fontsize=13
    )

    # --------------------------------------------------------
    # Animation frames
    # --------------------------------------------------------

    frame_indices = list(
        range(
            0,
            len(time_values),
            frame_skip
        )
    )

    if frame_indices[-1] != len(time_values) - 1:
        frame_indices.append(
            len(time_values) - 1
        )

    def update(frame_index):

        physical_rod.set_data(
            [0.0, x_values[frame_index]],
            [0.0, y_values[frame_index]]
        )

        physical_bob.set_data(
            [x_values[frame_index]],
            [y_values[frame_index]]
        )

        phase_bob.set_data(
            [theta_values[frame_index]],
            [omega_values[frame_index]]
        )

        physical_trail.set_segments(
            physical_segments_all[:frame_index]
        )

        phase_trail.set_segments(
            phase_segments_all[:frame_index]
        )

        if frame_index >= 1:
            segment_times = time_values[
                :frame_index
            ]

            current_widths = segment_widths[
                :frame_index
            ]

            physical_trail.set_array(
                segment_times
            )

            phase_trail.set_array(
                segment_times
            )

            physical_trail.set_linewidths(
                current_widths
            )

            phase_trail.set_linewidths(
                current_widths
            )

        time_text.set_text(
            f"Time: {time_values[frame_index]:.2f} s"
        )

        return (
            physical_rod,
            physical_bob,
            physical_trail,
            phase_bob,
            phase_trail,
            time_text
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_indices,
        interval=interval,
        blit=False,
        repeat=False
    )

    figure.tight_layout(
        rect=[0.0, 0.06, 1.0, 0.93]
    )

    return animation, figure
