# ============================================================
# Section 1: Imports
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


# ============================================================
# Section 2: Coloured physical trajectory
# ============================================================

def create_physical_trajectory_figure(
    results,
    save_path=None,
    color_map_name="turbo"
):
    """
    Create a physical double-pendulum trajectory figure.

    The trajectory of bob 2 changes colour and thickness
    according to its physical speed.

    Parameters
    ----------
    results : dictionary
        Results returned by simulate_double_pendulum().
    save_path : str, Path, or None
        Optional location for saving the figure.

    Returns
    -------
    figure, axis
        Matplotlib figure and axis objects.
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

    selected_color_map = plt.colormaps[
        color_map_name
    ]

    time_values = results["time"]

    x1_values = results["x1"]
    y1_values = results["y1"]

    x2_values = results["x2"]
    y2_values = results["y2"]

    parameters = results["parameters"]

    L1 = parameters["L1"]
    L2 = parameters["L2"]

    # Calculate the Cartesian speed of bob 2.
    x2_velocity = np.gradient(
        x2_values,
        time_values
    )

    y2_velocity = np.gradient(
        y2_values,
        time_values
    )

    bob2_speed = np.sqrt(
        x2_velocity ** 2
        + y2_velocity ** 2
    )

    # Convert the bob-2 trajectory into line segments.
    trajectory_points = np.column_stack(
        (
            x2_values,
            y2_values
        )
    )

    trajectory_segments = np.stack(
        (
            trajectory_points[:-1],
            trajectory_points[1:]
        ),
        axis=1
    )

    segment_speeds = bob2_speed[:-1]

    minimum_speed = np.min(segment_speeds)
    maximum_speed = np.max(segment_speeds)

    speed_range = maximum_speed - minimum_speed

    if speed_range > 1e-14:
        normalized_speed = (
            segment_speeds - minimum_speed
        ) / speed_range

    else:
        normalized_speed = np.zeros_like(
            segment_speeds
        )

    # Faster motion produces a thicker trajectory.
    line_widths = (
        0.8
        + 3.2 * normalized_speed
    )

    speed_normalization = Normalize(
        vmin=minimum_speed,
        vmax=maximum_speed
    )

    coloured_trajectory = LineCollection(
        trajectory_segments,
        cmap=selected_color_map,
        norm=speed_normalization,
        linewidths=line_widths,
        alpha=0.9
    )
    coloured_trajectory.set_array(
        segment_speeds
    )

    # Create the dark-background figure.
    plt.style.use("dark_background")

    figure, axis = plt.subplots(
        figsize=(10, 10)
    )

    figure.patch.set_facecolor("#070711")
    axis.set_facecolor("#070711")

    # Bob 1 follows a circular path around the pivot.
    axis.plot(
        x1_values,
        y1_values,
        color="#4DD6C8",
        linewidth=2.0,
        alpha=0.8,
        label="Trajectory of bob 1"
    )

    # Bob 2 creates the speed-dependent generative artwork.
    axis.add_collection(
        coloured_trajectory
    )

    # Draw the pendulum at the final time.
    final_x1 = x1_values[-1]
    final_y1 = y1_values[-1]

    final_x2 = x2_values[-1]
    final_y2 = y2_values[-1]

    axis.plot(
        [0.0, final_x1, final_x2],
        [0.0, final_y1, final_y2],
        color="#D7D7DF",
        linewidth=1.8,
        alpha=0.8,
        zorder=4
    )

    axis.scatter(
        0.0,
        0.0,
        s=120,
        color="white",
        edgecolor="white",
        zorder=6,
        label="Pivot"
    )

    axis.scatter(
        final_x1,
        final_y1,
        s=150,
        color="#4DD6C8",
        edgecolor="white",
        linewidth=1.2,
        zorder=6
    )

    axis.scatter(
        final_x2,
        final_y2,
        s=150,
        color="#FF7A18",
        edgecolor="white",
        linewidth=1.2,
        zorder=6
    )

    # Use the total pendulum length to determine the axes.
    maximum_distance = L1 + L2
    plot_limit = maximum_distance * 1.1

    axis.set_xlim(
        -plot_limit,
        plot_limit
    )

    axis.set_ylim(
        -plot_limit,
        plot_limit
    )

    axis.set_aspect(
        "equal",
        adjustable="box"
    )

    axis.set_title(
        "Physical Trajectory of the Double Pendulum",
        fontsize=18,
        pad=16
    )

    axis.set_xlabel(
        "Horizontal position x (m)",
        fontsize=12
    )

    axis.set_ylabel(
        "Vertical position y (m)",
        fontsize=12
    )

    axis.grid(
        color="white",
        alpha=0.12
    )

    # Add a colour bar explaining the visual mapping.
    colour_bar = figure.colorbar(
        coloured_trajectory,
        ax=axis,
        pad=0.02
    )

    colour_bar.set_label(
        "Speed of bob 2 (m/s)"
    )

    figure.tight_layout()

    # Save only when a path is supplied.
    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        figure.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=figure.get_facecolor()
        )

    return figure, axis


# ============================================================
# Section 3: Double-pendulum animation
# ============================================================

def create_double_pendulum_animation(
    results,
    frame_skip=10,
    interval=30,
    color_map_name="turbo"
):
    """
    Create an animation containing:

    1. The physical double pendulum.
    2. A persistent speed-dependent trajectory.
    3. The two angles as functions of time.
    4. The two angular velocities as functions of time.

    Parameters
    ----------
    results : dictionary
        Results returned by the simulation function.
    frame_skip : int
        Number of numerical points skipped between frames.
    interval : int
        Delay between animation frames in milliseconds.

    Returns
    -------
    animation, figure
        Matplotlib animation and figure objects.
    """

    if frame_skip < 1:
        raise ValueError(
            "frame_skip must be at least 1."
        )

    time_values = results["time"]

    theta1_values = results["theta1"]
    theta2_values = results["theta2"]

    omega1_values = results["omega1"]
    omega2_values = results["omega2"]

    x1_values = results["x1"]
    y1_values = results["y1"]

    x2_values = results["x2"]
    y2_values = results["y2"]

    parameters = results["parameters"]

    L1 = parameters["L1"]
    L2 = parameters["L2"]

    # Display angles in degrees on the side graph.
    theta1_degrees = np.rad2deg(
        theta1_values
    )

    theta2_degrees = np.rad2deg(
        theta2_values
    )
    
        # Use the original double.py colour mapping.
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

    animation_color_map = plt.colormaps[
        color_map_name
    ]

    animation_color_normalization = Normalize(
        vmin=time_values[0],
        vmax=time_values[-1]
    )

    maximum_omega1 = np.max(
        np.abs(omega1_values)
    )

    maximum_omega2 = np.max(
        np.abs(omega2_values)
    )

    if maximum_omega1 == 0:
        maximum_omega1 = 1.0

    if maximum_omega2 == 0:
        maximum_omega2 = 1.0

    phase1_points = np.column_stack(
        (
            theta1_values,
            omega1_values
        )
    )

    phase1_segments_all = np.stack(
        (
            phase1_points[:-1],
            phase1_points[1:]
        ),
        axis=1
    )

    phase2_points = np.column_stack(
        (
            theta2_values,
            omega2_values
        )
    )

    phase2_segments_all = np.stack(
        (
            phase2_points[:-1],
            phase2_points[1:]
        ),
        axis=1
    )
    
    # Calculate the physical speed of bob 2.
    x2_velocity = np.gradient(
        x2_values,
        time_values
    )

    y2_velocity = np.gradient(
        y2_values,
        time_values
    )

    bob2_speed = np.sqrt(
        x2_velocity ** 2
        + y2_velocity ** 2
    )

    # Construct all possible trajectory segments.
    trajectory_points = np.column_stack(
        (
            x2_values,
            y2_values
        )
    )

    trajectory_segments = np.stack(
        (
            trajectory_points[:-1],
            trajectory_points[1:]
        ),
        axis=1
    )

    segment_speeds = bob2_speed[:-1]

    minimum_speed = np.min(
        segment_speeds
    )

    maximum_speed = np.max(
        segment_speeds
    )

    speed_range = (
        maximum_speed
        - minimum_speed
    )

    if speed_range > 1e-14:
        normalized_speed = (
            segment_speeds
            - minimum_speed
        ) / speed_range

    else:
        normalized_speed = np.zeros_like(
            segment_speeds
        )

    line_widths = (
        0.8
        + 3.2 * normalized_speed
    )

    if speed_range > 1e-14:
        speed_normalization = Normalize(
            vmin=minimum_speed,
            vmax=maximum_speed
        )

    else:
        speed_normalization = Normalize(
            vmin=minimum_speed,
            vmax=minimum_speed + 1.0
        )

    # Create the three-panel layout.
    plt.style.use("dark_background")

    figure = plt.figure(
        figsize=(15, 8)
    )

    figure.patch.set_facecolor(
        "#070711"
    )

    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[1.55, 1.0],
        hspace=0.32,
        wspace=0.28
    )

    physical_axis = figure.add_subplot(
        grid[:, 0]
    )

    angle_axis = figure.add_subplot(
        grid[0, 1]
    )

    velocity_axis = figure.add_subplot(
        grid[1, 1]
    )

    for current_axis in (
        physical_axis,
        angle_axis,
        velocity_axis
    ):
        current_axis.set_facecolor(
            "#070711"
        )

        current_axis.grid(
            color="white",
            alpha=0.12
        )

    # --------------------------------------------------------
    # Left panel: physical trajectory
    # --------------------------------------------------------

    trajectory_collection = LineCollection(
        [],
        cmap=animation_color_map,
        norm=animation_color_normalization,
        alpha=0.9
    )

    physical_axis.add_collection(
        trajectory_collection
    )

    bob1_trail, = physical_axis.plot(
        [],
        [],
        color="#4DD6C8",
        linewidth=2.0,
        alpha=0.8,
        label="Trajectory of bob 1"
    )

    pendulum_line, = physical_axis.plot(
        [],
        [],
        color="#D7D7DF",
        linewidth=2.0,
        zorder=5
    )

    bob1_marker, = physical_axis.plot(
        [],
        [],
        marker="o",
        markersize=11,
        linestyle="None",
        color="#4DD6C8",
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=7
    )

    bob2_marker, = physical_axis.plot(
        [],
        [],
        marker="o",
        markersize=11,
        linestyle="None",
        color="#FF7A18",
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=7
    )

    physical_axis.scatter(
        0.0,
        0.0,
        s=110,
        color="white",
        edgecolor="white",
        zorder=8,
        label="Pivot"
    )

    time_text = physical_axis.text(
        0.03,
        0.96,
        "",
        transform=physical_axis.transAxes,
        verticalalignment="top",
        fontsize=12,
        color="white"
    )

    maximum_distance = L1 + L2
    plot_limit = 1.1 * maximum_distance

    physical_axis.set_xlim(
        -plot_limit,
        plot_limit
    )

    physical_axis.set_ylim(
        -plot_limit,
        plot_limit
    )

    physical_axis.set_aspect(
        "equal",
        adjustable="box"
    )

    physical_axis.set_title(
        "Physical Motion and Persistent Trajectory",
        fontsize=16,
        pad=12
    )

    physical_axis.set_xlabel(
        "Horizontal position x (m)"
    )

    physical_axis.set_ylabel(
        "Vertical position y (m)"
    )

    # --------------------------------------------------------
    # Right-top panel: phase space of pendulum 1
    # --------------------------------------------------------

    phase1_trail = LineCollection(
        [],
        cmap=animation_color_map,
        norm=animation_color_normalization,
        alpha=0.9,
        zorder=2
    )

    angle_axis.add_collection(
        phase1_trail
    )

    phase1_marker, = angle_axis.plot(
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

    theta1_margin = (
        0.08 * np.ptp(theta1_values)
        + 0.05
    )

    omega1_margin = (
        0.08 * np.ptp(omega1_values)
        + 0.05
    )

    angle_axis.set_xlim(
        np.min(theta1_values) - theta1_margin,
        np.max(theta1_values) + theta1_margin
    )

    angle_axis.set_ylim(
        np.min(omega1_values) - omega1_margin,
        np.max(omega1_values) + omega1_margin
    )

    angle_axis.set_title(
        "Pendulum 1 Phase Space"
    )

    angle_axis.set_xlabel(
        r"Angle $\theta_1$ (rad)"
    )

    angle_axis.set_ylabel(
        r"Angular velocity $\omega_1$ (rad/s)"
    )

    angle_axis.axhline(
        0.0,
        color="white",
        linewidth=0.8,
        alpha=0.35
    )

    angle_axis.axvline(
        0.0,
        color="white",
        linewidth=0.8,
        alpha=0.35
    )

    # --------------------------------------------------------
    # Right-bottom panel: phase space of pendulum 2
    # --------------------------------------------------------

    phase2_trail = LineCollection(
        [],
        cmap=animation_color_map,
        norm=animation_color_normalization,
        alpha=0.9,
        zorder=2
    )

    velocity_axis.add_collection(
        phase2_trail
    )

    phase2_marker, = velocity_axis.plot(
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

    theta2_margin = (
        0.08 * np.ptp(theta2_values)
        + 0.05
    )

    omega2_margin = (
        0.08 * np.ptp(omega2_values)
        + 0.05
    )

    velocity_axis.set_xlim(
        np.min(theta2_values) - theta2_margin,
        np.max(theta2_values) + theta2_margin
    )

    velocity_axis.set_ylim(
        np.min(omega2_values) - omega2_margin,
        np.max(omega2_values) + omega2_margin
    )

    velocity_axis.set_title(
        "Pendulum 2 Phase Space"
    )

    velocity_axis.set_xlabel(
        r"Angle $\theta_2$ (rad)"
    )

    velocity_axis.set_ylabel(
        r"Angular velocity $\omega_2$ (rad/s)"
    )

    velocity_axis.axhline(
        0.0,
        color="white",
        linewidth=0.8,
        alpha=0.35
    )

    velocity_axis.axvline(
        0.0,
        color="white",
        linewidth=0.8,
        alpha=0.35
    )

    
    # --------------------------------------------------------
    # Animation frames
    # --------------------------------------------------------

    frame_indices = list(
        range(
            1,
            len(time_values),
            frame_skip
        )
    )

    if frame_indices[-1] != len(time_values) - 1:
        frame_indices.append(
            len(time_values) - 1
        )

    def update(frame_index):

        # Keep every previous bob-2 trajectory segment.
        trajectory_collection.set_segments(
            trajectory_segments[:frame_index]
        )

        trajectory_collection.set_array(
            time_values[:frame_index]
        )

        trajectory_collection.set_linewidths(
            line_widths[:frame_index]
        )

        # Keep the complete bob-1 trajectory up to this time.
        bob1_trail.set_data(
            x1_values[:frame_index + 1],
            y1_values[:frame_index + 1]
        )

        # Update the pendulum and both bobs.
        pendulum_line.set_data(
            [
                0.0,
                x1_values[frame_index],
                x2_values[frame_index]
            ],
            [
                0.0,
                y1_values[frame_index],
                y2_values[frame_index]
            ]
        )

        bob1_marker.set_data(
            [x1_values[frame_index]],
            [y1_values[frame_index]]
        )

        bob2_marker.set_data(
            [x2_values[frame_index]],
            [y2_values[frame_index]]
        )

        current_time = time_values[
            frame_index
        ]

        time_text.set_text(
            f"Time: {current_time:.2f} s"
        )

        # Update pendulum 1 phase space.
        phase1_trail.set_segments(
            phase1_segments_all[:frame_index]
        )

        phase1_trail.set_array(
            time_values[:frame_index]
        )

        phase1_line_widths = (
            0.5
            + 3.0
            * np.abs(
                omega1_values[:frame_index]
            )
            / maximum_omega1
        )

        phase1_trail.set_linewidths(
            phase1_line_widths
        )

        phase1_marker.set_data(
            [theta1_values[frame_index]],
            [omega1_values[frame_index]]
        )

        # Update pendulum 2 phase space.
        phase2_trail.set_segments(
            phase2_segments_all[:frame_index]
        )

        phase2_trail.set_array(
            time_values[:frame_index]
        )

        phase2_line_widths = (
            0.5
            + 3.0
            * np.abs(
                omega2_values[:frame_index]
            )
            / maximum_omega2
        )

        phase2_trail.set_linewidths(
            phase2_line_widths
        )

        phase2_marker.set_data(
            [theta2_values[frame_index]],
            [omega2_values[frame_index]]
        )
        
        return (
            trajectory_collection,
            bob1_trail,
            pendulum_line,
            bob1_marker,
            bob2_marker,
            time_text,
            phase1_trail,
            phase1_marker,
            phase2_trail,
            phase2_marker
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_indices,
        interval=interval,
        blit=False,
        repeat=False
    )

    figure.suptitle(
        "Double-Pendulum Generative Art and Physical Motion",
        fontsize=19,
        y=0.98
    )

    return animation, figure