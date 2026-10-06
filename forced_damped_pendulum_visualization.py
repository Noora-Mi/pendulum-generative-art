import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


AVAILABLE_COLOR_MAPS = (
    "turbo",
    "plasma",
    "viridis",
    "inferno",
    "magma",
    "cividis",
)


def _validate_visual_parameters(
    color_map_name,
    minimum_line_width,
    maximum_line_width,
    minimum_alpha,
    maximum_alpha,
):
    if color_map_name not in AVAILABLE_COLOR_MAPS:
        raise ValueError(
            "Unsupported colour map: " + color_map_name
        )

    if minimum_line_width <= 0.0:
        raise ValueError(
            "minimum_line_width must be positive."
        )

    if maximum_line_width < minimum_line_width:
        raise ValueError(
            "maximum_line_width must not be smaller than "
            "minimum_line_width."
        )

    if not 0.0 <= minimum_alpha <= 1.0:
        raise ValueError(
            "minimum_alpha must lie between 0 and 1."
        )

    if not minimum_alpha <= maximum_alpha <= 1.0:
        raise ValueError(
            "maximum_alpha must lie between minimum_alpha "
            "and 1."
        )


def _normalize(values):
    values = np.asarray(values, dtype=float)
    minimum = np.min(values)
    maximum = np.max(values)

    if np.isclose(maximum, minimum):
        return np.full_like(values, 0.5)

    return (values - minimum) / (maximum - minimum)


def _segments(x_values, y_values, break_wrapping=False):
    points = np.column_stack((x_values, y_values))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    indices = np.arange(len(segments))

    if break_wrapping:
        valid = np.abs(np.diff(x_values)) < np.pi
        segments = segments[valid]
        indices = indices[valid]

    return segments, indices


def _visual_arrays(
    results,
    color_map_name,
    minimum_line_width,
    maximum_line_width,
    minimum_alpha,
    maximum_alpha,
):
    time_values = results["time"]
    omega_values = results["omega"]
    energy_values = results["mechanical_energy"]

    speed_normalized = _normalize(
        np.abs(omega_values[:-1])
    )
    energy_normalized = _normalize(
        energy_values[:-1]
    )

    line_widths = (
        minimum_line_width
        + (maximum_line_width - minimum_line_width)
        * speed_normalized
    )

    alpha_values = (
        minimum_alpha
        + (maximum_alpha - minimum_alpha)
        * energy_normalized
    )

    color_normalization = Normalize(
        vmin=time_values[0],
        vmax=time_values[-1],
    )
    color_map = plt.colormaps[color_map_name]
    colors = color_map(
        color_normalization(time_values[:-1])
    )
    colors[:, 3] = alpha_values

    return (
        color_map,
        color_normalization,
        colors,
        line_widths,
    )


def _prepare_data(results):
    time_values = results["time"]
    theta_values = results["theta"]
    wrapped_theta_values = results["wrapped_theta"]
    omega_values = results["omega"]

    x_values = np.sin(theta_values)
    y_values = -np.cos(theta_values)

    physical_segments, physical_indices = _segments(
        x_values,
        y_values,
    )
    phase_segments, phase_indices = _segments(
        wrapped_theta_values,
        omega_values,
        break_wrapping=True,
    )

    return {
        "time": time_values,
        "theta": theta_values,
        "wrapped_theta": wrapped_theta_values,
        "omega": omega_values,
        "energy": results["mechanical_energy"],
        "x": x_values,
        "y": y_values,
        "physical_segments": physical_segments,
        "physical_indices": physical_indices,
        "phase_segments": phase_segments,
        "phase_indices": phase_indices,
    }


def _poincare_indices(results):
    """Return indices sampled once per driving period."""

    frequency = results["parameters"]["drive_frequency"]
    if frequency <= 0.0:
        return np.empty(0, dtype=int)

    time_values = results["time"]
    driving_period = 2.0 * np.pi / frequency
    sample_times = np.arange(
        driving_period,
        time_values[-1] + 0.5 * results["step_size"],
        driving_period,
    )
    indices = np.searchsorted(time_values, sample_times)
    return np.clip(indices, 0, len(time_values) - 1)


def _style_axes(figure, physical_axis, phase_axis):
    figure.patch.set_facecolor("#070711")

    for axis in (physical_axis, phase_axis):
        axis.set_facecolor("#070711")
        axis.grid(True, alpha=0.13)


def create_forced_damped_pendulum_figure(
    results,
    color_map_name="turbo",
    minimum_line_width=0.6,
    maximum_line_width=3.6,
    minimum_alpha=0.20,
    maximum_alpha=1.00,
):
    """Create physical-space and phase-space static artwork.

    Colour represents dimensionless time, line width represents
    absolute angular velocity, and opacity represents instantaneous
    dimensionless mechanical energy.
    """

    _validate_visual_parameters(
        color_map_name,
        minimum_line_width,
        maximum_line_width,
        minimum_alpha,
        maximum_alpha,
    )

    data = _prepare_data(results)
    (
        color_map,
        color_normalization,
        colors,
        line_widths,
    ) = _visual_arrays(
        results,
        color_map_name,
        minimum_line_width,
        maximum_line_width,
        minimum_alpha,
        maximum_alpha,
    )

    plt.style.use("dark_background")
    figure, (physical_axis, phase_axis) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )
    _style_axes(figure, physical_axis, phase_axis)

    figure.suptitle(
        "Forced and Damped Pendulum: Physical and Phase Space",
        fontsize=17,
        color="white",
    )

    physical_collection = LineCollection(
        data["physical_segments"],
        colors=colors[data["physical_indices"]],
        linewidths=line_widths[data["physical_indices"]],
        zorder=2,
    )
    physical_axis.add_collection(physical_collection)
    physical_axis.scatter(0.0, 0.0, color="white", s=65, zorder=5)
    physical_axis.set_xlim(-1.15, 1.15)
    physical_axis.set_ylim(-1.15, 1.15)
    physical_axis.set_aspect("equal", adjustable="box")
    physical_axis.set_title(r"Physical Space: $(x,y)$")
    physical_axis.set_xlabel(r"$x=\sin\theta$")
    physical_axis.set_ylabel(r"$y=-\cos\theta$")

    phase_collection = LineCollection(
        data["phase_segments"],
        colors=colors[data["phase_indices"]],
        linewidths=line_widths[data["phase_indices"]],
        zorder=2,
    )
    phase_axis.add_collection(phase_collection)
    omega_margin = 0.08 * np.ptp(data["omega"]) + 0.10
    phase_axis.set_xlim(-np.pi, np.pi)
    phase_axis.set_ylim(
        np.min(data["omega"]) - omega_margin,
        np.max(data["omega"]) + omega_margin,
    )
    phase_axis.axhline(0.0, color="white", alpha=0.25, linewidth=0.8)
    phase_axis.axvline(0.0, color="white", alpha=0.25, linewidth=0.8)
    phase_axis.set_title(r"Phase Space: $(\theta_{\mathrm{w}},\omega)$")
    phase_axis.set_xlabel(r"Wrapped angle $\theta_{\mathrm{w}}$")
    phase_axis.set_ylabel(r"Dimensionless angular velocity $\omega$")

    poincare_indices = _poincare_indices(results)
    if len(poincare_indices) > 0:
        phase_axis.scatter(
            data["wrapped_theta"][poincare_indices],
            data["omega"][poincare_indices],
            s=38,
            facecolor="#f8fafc",
            edgecolor="#ef4444",
            linewidth=1.2,
            zorder=7,
            label="Poincaré samples",
        )
        phase_axis.legend(loc="best")

    scalar_mappable = ScalarMappable(
        norm=color_normalization,
        cmap=color_map,
    )
    scalar_mappable.set_array([])
    color_bar = figure.colorbar(
        scalar_mappable,
        ax=[physical_axis, phase_axis],
        fraction=0.035,
        pad=0.04,
    )
    color_bar.set_label(r"Dimensionless time $\tau$")

    figure.text(
        0.5,
        0.015,
        "Colour: time   |   Width: |omega|   |   Opacity: mechanical energy",
        ha="center",
        color="white",
        fontsize=10,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.88,
        bottom=0.13,
        top=0.88,
        wspace=0.24,
    )

    return figure, physical_axis, phase_axis


def create_forced_damped_pendulum_animation(
    results,
    frame_skip=3,
    interval=None,
    color_map_name="turbo",
    minimum_line_width=0.6,
    maximum_line_width=3.6,
    minimum_alpha=0.20,
    maximum_alpha=1.00,
):
    """Create synchronized physical and phase-space animation."""

    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1.")

    _validate_visual_parameters(
        color_map_name,
        minimum_line_width,
        maximum_line_width,
        minimum_alpha,
        maximum_alpha,
    )

    data = _prepare_data(results)
    (
        color_map,
        color_normalization,
        colors,
        line_widths,
    ) = _visual_arrays(
        results,
        color_map_name,
        minimum_line_width,
        maximum_line_width,
        minimum_alpha,
        maximum_alpha,
    )

    if interval is None:
        interval = results["step_size"] * frame_skip * 1000.0

    plt.style.use("dark_background")
    figure, (physical_axis, phase_axis) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )
    _style_axes(figure, physical_axis, phase_axis)
    figure.suptitle(
        "Forced and Damped Pendulum: Dynamic Evolution",
        fontsize=17,
        color="white",
    )

    physical_axis.set_xlim(-1.15, 1.15)
    physical_axis.set_ylim(-1.15, 1.15)
    physical_axis.set_aspect("equal", adjustable="box")
    physical_axis.set_title(r"Physical Space: $(x,y)$")
    physical_axis.set_xlabel(r"$x=\sin\theta$")
    physical_axis.set_ylabel(r"$y=-\cos\theta$")
    physical_axis.scatter(0.0, 0.0, color="white", s=65, zorder=6)

    rod, = physical_axis.plot(
        [], [], color="white", linewidth=2.0, alpha=0.75, zorder=4
    )
    bob, = physical_axis.plot(
        [], [], marker="o", markersize=13, color="#5eead4",
        markeredgecolor="white", markeredgewidth=0.8,
        linestyle="None", zorder=6,
    )
    physical_trail = LineCollection([], zorder=2)
    physical_axis.add_collection(physical_trail)

    omega_margin = 0.08 * np.ptp(data["omega"]) + 0.10
    phase_axis.set_xlim(-np.pi, np.pi)
    phase_axis.set_ylim(
        np.min(data["omega"]) - omega_margin,
        np.max(data["omega"]) + omega_margin,
    )
    phase_axis.set_title(r"Phase Space: $(\theta_{\mathrm{w}},\omega)$")
    phase_axis.set_xlabel(r"Wrapped angle $\theta_{\mathrm{w}}$")
    phase_axis.set_ylabel(r"Dimensionless angular velocity $\omega$")
    phase_axis.axhline(0.0, color="white", alpha=0.25, linewidth=0.8)
    phase_axis.axvline(0.0, color="white", alpha=0.25, linewidth=0.8)
    phase_trail = LineCollection([], zorder=2)
    phase_axis.add_collection(phase_trail)
    phase_point, = phase_axis.plot(
        [], [], marker="o", markersize=10, color="#f8fafc",
        markeredgecolor="#5eead4", markeredgewidth=2.0,
        linestyle="None", zorder=6,
    )
    poincare_points = phase_axis.scatter(
        [], [], s=42, facecolor="#f8fafc",
        edgecolor="#ef4444", linewidth=1.2,
        zorder=7, label="Poincaré samples",
    )
    phase_axis.legend(loc="best")
    poincare_indices = _poincare_indices(results)

    scalar_mappable = ScalarMappable(
        norm=color_normalization,
        cmap=color_map,
    )
    scalar_mappable.set_array([])
    color_bar = figure.colorbar(
        scalar_mappable,
        ax=[physical_axis, phase_axis],
        fraction=0.035,
        pad=0.04,
    )
    color_bar.set_label(r"Dimensionless time $\tau$")

    time_text = figure.text(
        0.5,
        0.035,
        "",
        ha="center",
        color="white",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.012,
        "Colour: time   |   Width: |omega|   |   Opacity: mechanical energy",
        ha="center",
        color="white",
        fontsize=9,
    )

    frame_indices = list(
        range(0, len(data["time"]), frame_skip)
    )
    if frame_indices[-1] != len(data["time"]) - 1:
        frame_indices.append(len(data["time"]) - 1)

    def update(frame_index):
        rod.set_data(
            [0.0, data["x"][frame_index]],
            [0.0, data["y"][frame_index]],
        )
        bob.set_data(
            [data["x"][frame_index]],
            [data["y"][frame_index]],
        )
        phase_point.set_data(
            [data["wrapped_theta"][frame_index]],
            [data["omega"][frame_index]],
        )

        physical_mask = data["physical_indices"] < frame_index
        physical_indices = data["physical_indices"][physical_mask]
        physical_trail.set_segments(
            data["physical_segments"][physical_mask]
        )
        physical_trail.set_color(colors[physical_indices])
        physical_trail.set_linewidths(line_widths[physical_indices])

        phase_mask = data["phase_indices"] < frame_index
        phase_indices = data["phase_indices"][phase_mask]
        phase_trail.set_segments(
            data["phase_segments"][phase_mask]
        )
        phase_trail.set_color(colors[phase_indices])
        phase_trail.set_linewidths(line_widths[phase_indices])

        visible_poincare = poincare_indices[
            poincare_indices <= frame_index
        ]
        if len(visible_poincare) > 0:
            poincare_points.set_offsets(
                np.column_stack((
                    data["wrapped_theta"][visible_poincare],
                    data["omega"][visible_poincare],
                ))
            )
        else:
            poincare_points.set_offsets(np.empty((0, 2)))

        time_text.set_text(
            rf"Dimensionless time: $\tau={data['time'][frame_index]:.2f}$"
        )

        return (
            rod, bob, physical_trail, phase_trail,
            phase_point, poincare_points, time_text
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_indices,
        interval=interval,
        blit=False,
        repeat=False,
    )

    figure.subplots_adjust(
        left=0.07,
        right=0.88,
        bottom=0.14,
        top=0.88,
        wspace=0.24,
    )

    return animation, figure


def create_damping_comparison_animation(
    undamped_results,
    damped_results,
    frame_skip=3,
    interval=None,
    color_map_name="turbo",
    minimum_line_width=0.6,
    maximum_line_width=3.6,
    minimum_alpha=0.20,
    maximum_alpha=1.00,
):
    """Compare unforced undamped and damped motion side by side."""

    _validate_visual_parameters(
        color_map_name, minimum_line_width, maximum_line_width,
        minimum_alpha, maximum_alpha,
    )
    left = _prepare_data(undamped_results)
    right = _prepare_data(damped_results)
    left_visual = _visual_arrays(
        undamped_results, color_map_name,
        minimum_line_width, maximum_line_width,
        minimum_alpha, maximum_alpha,
    )
    right_visual = _visual_arrays(
        damped_results, color_map_name,
        minimum_line_width, maximum_line_width,
        minimum_alpha, maximum_alpha,
    )
    if interval is None:
        interval = undamped_results["step_size"] * frame_skip * 1000.0

    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    figure.patch.set_facecolor("#070711")
    figure.suptitle(
        "Effect of Damping: Same Initial State, No External Force",
        fontsize=17,
    )

    configurations = (
        (left, left_visual, axes[0, 0], axes[1, 0], "Undamped: q = 0"),
        (right, right_visual, axes[0, 1], axes[1, 1], "Damped: q = 0.2"),
    )
    artists = []
    for data, visual, physical_axis, phase_axis, title in configurations:
        physical_axis.set_facecolor("#070711")
        phase_axis.set_facecolor("#070711")
        physical_axis.grid(True, alpha=0.13)
        phase_axis.grid(True, alpha=0.13)
        physical_axis.set_xlim(-1.15, 1.15)
        physical_axis.set_ylim(-1.15, 1.15)
        physical_axis.set_aspect("equal", adjustable="box")
        physical_axis.set_title(title + " — Physical space")
        physical_axis.set_xlabel(r"$x=\sin\theta$")
        physical_axis.set_ylabel(r"$y=-\cos\theta$")
        physical_axis.scatter(0.0, 0.0, color="white", s=55, zorder=6)
        rod, = physical_axis.plot(
            [], [], color="white", linewidth=2.0, alpha=0.75, zorder=4
        )
        bob, = physical_axis.plot(
            [], [], marker="o", markersize=11, color="#5eead4",
            markeredgecolor="white", linestyle="None", zorder=6,
        )
        physical_trail = LineCollection([], zorder=2)
        physical_axis.add_collection(physical_trail)

        omega_margin = 0.08 * np.ptp(data["omega"]) + 0.10
        phase_axis.set_xlim(-np.pi, np.pi)
        phase_axis.set_ylim(
            np.min(data["omega"]) - omega_margin,
            np.max(data["omega"]) + omega_margin,
        )
        phase_axis.set_title(title + " — Phase space")
        phase_axis.set_xlabel(r"Wrapped angle $\theta_{\mathrm{w}}$")
        phase_axis.set_ylabel(r"Angular velocity $\omega$")
        phase_axis.axhline(0.0, color="white", alpha=0.25, linewidth=0.8)
        phase_axis.axvline(0.0, color="white", alpha=0.25, linewidth=0.8)
        phase_trail = LineCollection([], zorder=2)
        phase_axis.add_collection(phase_trail)
        phase_point, = phase_axis.plot(
            [], [], marker="o", markersize=8, color="#f8fafc",
            markeredgecolor="#5eead4", linestyle="None", zorder=6,
        )
        artists.append({
            "data": data,
            "colors": visual[2],
            "widths": visual[3],
            "rod": rod,
            "bob": bob,
            "physical_trail": physical_trail,
            "phase_trail": phase_trail,
            "phase_point": phase_point,
        })

    time_text = figure.text(0.5, 0.018, "", ha="center", fontsize=12)
    frame_indices = list(range(0, len(left["time"]), frame_skip))
    if frame_indices[-1] != len(left["time"]) - 1:
        frame_indices.append(len(left["time"]) - 1)

    def update(frame_index):
        returned = []
        for item in artists:
            data = item["data"]
            item["rod"].set_data(
                [0.0, data["x"][frame_index]],
                [0.0, data["y"][frame_index]],
            )
            item["bob"].set_data(
                [data["x"][frame_index]], [data["y"][frame_index]]
            )
            item["phase_point"].set_data(
                [data["wrapped_theta"][frame_index]],
                [data["omega"][frame_index]],
            )
            physical_mask = data["physical_indices"] < frame_index
            p_indices = data["physical_indices"][physical_mask]
            item["physical_trail"].set_segments(
                data["physical_segments"][physical_mask]
            )
            item["physical_trail"].set_color(item["colors"][p_indices])
            item["physical_trail"].set_linewidths(item["widths"][p_indices])
            phase_mask = data["phase_indices"] < frame_index
            s_indices = data["phase_indices"][phase_mask]
            item["phase_trail"].set_segments(
                data["phase_segments"][phase_mask]
            )
            item["phase_trail"].set_color(item["colors"][s_indices])
            item["phase_trail"].set_linewidths(item["widths"][s_indices])
            returned.extend((
                item["rod"], item["bob"], item["physical_trail"],
                item["phase_trail"], item["phase_point"],
            ))
        time_text.set_text(
            rf"Dimensionless time: $\tau={left['time'][frame_index]:.2f}$"
        )
        returned.append(time_text)
        return tuple(returned)

    animation = FuncAnimation(
        figure, update, frames=frame_indices,
        interval=interval, blit=False, repeat=False,
    )
    figure.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
    return animation, figure
