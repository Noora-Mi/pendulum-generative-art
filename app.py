# ============================================================
# Section 1: Imports
# ============================================================

from math import ceil
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock

from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from single_pendulum_model import (
    simulate_single_pendulum
)

from single_pendulum_visualization import (
    create_single_pendulum_animation,
    create_single_pendulum_figure
)

from double_pendulum_model import (
    simulate_double_pendulum_degrees
)

from double_pendulum_visualization import (
    create_double_pendulum_animation,
    create_physical_trajectory_figure
)

from forced_damped_pendulum_model import (
    simulate_forced_damped_pendulum
)

from forced_damped_pendulum_visualization import (
    create_damping_comparison_animation,
    create_forced_damped_pendulum_animation,
    create_forced_damped_pendulum_figure
)


matplotlib_lock = RLock()

# ============================================================
# Section 2: Animation helpers
# ============================================================

def choose_frame_skip(
    number_of_points,
    maximum_frames=240
):
    """
    Select a frame skip that limits the GIF size and
    website generation time.
    """

    return max(
        1,
        ceil(
            number_of_points
            / maximum_frames
        )
    )


def animation_to_gif_bytes(
    animation,
    fps=30,
    dpi=90
):
    """
    Save a Matplotlib animation temporarily and return
    the GIF as bytes for display and download.
    """

    temporary_path = None

    try:
        with NamedTemporaryFile(
            suffix=".gif",
            delete=False
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

        animation.save(
            temporary_path,
            writer="pillow",
            fps=fps,
            dpi=dpi
        )

        gif_bytes = temporary_path.read_bytes()

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()

    return gif_bytes


# ============================================================
# Section 2: Page configuration
# ============================================================

st.set_page_config(
    page_title="Pendulum Generative Art",
    page_icon="🎨",
    layout="wide"
)

st.title(
    "Pendulum Generative Art"
)

st.markdown(
    """
    Explore how physical parameters and initial conditions
    generate different visual trajectories.

    Colour represents a changing physical quantity, while
    line thickness responds to motion speed.
    """
)


# ============================================================
# Section 3: Model selection
# ============================================================

model_choice = st.sidebar.radio(
    "Choose a model",
    [
        "Part I: Simple Pendulum",
        "Part II: Forced and Damped Pendulum",
        "Part III: Double Pendulum"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Angles are entered in degrees. "
    "The numerical RK4 step size is selected automatically."
)


# ============================================================
# Section 4: Single-pendulum interface
# ============================================================

if model_choice == "Part I: Simple Pendulum":

    st.header(
        "Part I: Simple Pendulum"
    )

    st.markdown(
        r"""
        The ideal nonlinear single-pendulum model is

        $$
        \ddot{\theta}
        + \frac{g}{L}\sin(\theta)=0.
        $$
        """
    )

    regime_defaults = {
        "Oscillation": (
            85.94, 0.0, 1.0, 1.0, 12.0
        ),
        "Separatrix": (
            0.0, 6.26418, 1.0, 1.0, 12.0
        ),
        "Rotation": (
            0.0, 7.0, 1.0, 1.0, 12.0
        ),
        "Custom": (
            45.0, 0.0, 1.0, 1.0, 12.0
        )
    }

    def load_part1_preset():
        selected_regime = st.session_state[
            "part1_preset"
        ]

        if selected_regime == "Custom":
            return

        (
            preset_theta,
            preset_omega,
            preset_length,
            preset_mass,
            preset_time
        ) = regime_defaults[selected_regime]

        st.session_state["part1_theta"] = preset_theta
        st.session_state["part1_omega"] = preset_omega
        st.session_state["part1_length"] = preset_length
        st.session_state["part1_mass"] = preset_mass
        st.session_state["part1_time"] = preset_time

    regime_choice = st.sidebar.selectbox(
        "Starting motion-regime preset",
        [
            "Oscillation",
            "Separatrix",
            "Rotation",
            "Custom"
        ],
        key="part1_preset",
        on_change=load_part1_preset
    )

    (
        default_theta,
        default_omega,
        default_length,
        default_mass,
        default_time
    ) = regime_defaults[regime_choice]

    with st.sidebar.form(
        "single_pendulum_parameters"
    ):

        st.subheader(
            "Single-Pendulum Parameters"
        )

        theta_degrees = st.slider(
            "Initial angle θ₀ (degrees)",
            min_value=-180.0,
            max_value=180.0,
            value=default_theta,
            step=0.01,
            key="part1_theta",
            disabled=(regime_choice != "Custom")
        )

        omega_initial = st.slider(
            "Initial angular velocity ω₀ (rad/s)",
            min_value=-8.0,
            max_value=8.0,
            value=default_omega,
            step=0.01,
            key="part1_omega",
            disabled=(regime_choice != "Custom")
        )

        length = st.slider(
            "Pendulum length L (m)",
            min_value=0.5,
            max_value=2.0,
            value=default_length,
            step=0.1,
            key="part1_length",
            disabled=(regime_choice != "Custom")
        )

        mass = st.slider(
            "Pendulum mass m (kg)",
            min_value=0.4,
            max_value=1.6,
            value=default_mass,
            step=0.1,
            key="part1_mass",
            disabled=(regime_choice != "Custom")
        )

        total_time = st.slider(
            "Simulation time (s)",
            min_value=1.0,
            max_value=20.0,
            value=default_time,
            step=1.0,
            key="part1_time"
        )

        color_map_name = st.selectbox(
            "Trajectory colour map",
            [
                "turbo",
                "plasma",
                "viridis",
                "inferno",
                "magma",
                "cividis"
            ]
        )

        minimum_line_width = st.slider(
            "Minimum trajectory width",
            min_value=0.2,
            max_value=3.0,
            value=0.6,
            step=0.1
        )

        maximum_line_width = st.slider(
            "Maximum trajectory width",
            min_value=1.0,
            max_value=8.0,
            value=3.6,
            step=0.1
        )

        minimum_alpha = st.slider(
            "Minimum trajectory opacity",
            min_value=0.05,
            max_value=1.0,
            value=0.20,
            step=0.05
        )

        maximum_alpha = st.slider(
            "Maximum trajectory opacity",
            min_value=0.05,
            max_value=1.0,
            value=1.0,
            step=0.05
        )

        generate_single_static = (
            st.form_submit_button(
                "Generate Static Artwork",
                type="primary"
            )
        )

        generate_single_animation = (
            st.form_submit_button(
                "Generate Animation"
            )
        )

    if (
        generate_single_static
        or generate_single_animation
    ):

        if maximum_line_width < minimum_line_width:
            st.error(
                "Maximum width must be at least the minimum width."
            )
            st.stop()

        if maximum_alpha < minimum_alpha:
            st.error(
                "Maximum opacity must be at least the minimum opacity."
            )
            st.stop()

        with st.spinner(
            "Running the single-pendulum simulation..."
        ):

            single_results = (
                simulate_single_pendulum(
                    theta_initial=np.deg2rad(theta_degrees),
                    omega_initial=omega_initial,
                    length=length,
                    mass=mass,
                    damping=0.0,
                    total_time=total_time,
                    step_size=0.0025
                )
            )

        # ----------------------------------------------------
        # Static artwork
        # ----------------------------------------------------

        if generate_single_static:

            with matplotlib_lock:
                single_figure, _, _ = (
                    create_single_pendulum_figure(
                        results=single_results,
                        color_map_name=color_map_name,
                        minimum_line_width=minimum_line_width,
                        maximum_line_width=maximum_line_width,
                        minimum_alpha=minimum_alpha,
                        maximum_alpha=maximum_alpha
                    )
                )

                single_png_buffer = BytesIO()

                single_figure.savefig(
                    single_png_buffer,
                    format="png",
                    dpi=300,
                    bbox_inches="tight"
                )

                single_png_bytes = (
                    single_png_buffer.getvalue()
                )

                st.pyplot(
                    single_figure,
                    width="stretch"
                )

                st.download_button(
                    label="Download Single-Pendulum PNG",
                    data=single_png_bytes,
                    file_name=(
                        "single_pendulum_artwork.png"
                    ),
                    mime="image/png"
                )

                single_png_buffer.close()

                plt.close(
                    single_figure
                )

        # ----------------------------------------------------
        # Animated artwork
        # ----------------------------------------------------

        if generate_single_animation:

            single_frame_skip = choose_frame_skip(
                len(single_results["time"])
            )

            with st.spinner(
                "Rendering the GIF animation. "
                "This may take a few minutes..."
            ):

                with matplotlib_lock:
                    (
                        single_animation,
                        single_animation_figure
                    ) = create_single_pendulum_animation(
                        results=single_results,
                        frame_skip=single_frame_skip,
                        color_map_name=color_map_name,
                        minimum_line_width=minimum_line_width,
                        maximum_line_width=maximum_line_width,
                        minimum_alpha=minimum_alpha,
                        maximum_alpha=maximum_alpha
                    )

                    single_gif_bytes = (
                        animation_to_gif_bytes(
                            single_animation,
                            fps=30,
                            dpi=90
                        )
                    )

                    plt.close(
                        single_animation_figure
                    )

            st.image(
                single_gif_bytes
            )

            st.download_button(
                label="Download Single-Pendulum GIF",
                data=single_gif_bytes,
                file_name=(
                    "single_pendulum_animation.gif"
                ),
                mime="image/gif"
            )

        # ----------------------------------------------------
        # Physical and numerical results
        # ----------------------------------------------------

        metric1, metric2, metric3 = st.columns(
            3
        )

        metric1.metric(
            "Initial energy",
            (
                f"{single_results['initial_total_energy']:.5f} J"
            )
        )

        metric2.metric(
            "Final energy",
            (
                f"{single_results['final_total_energy']:.5f} J"
            )
        )

        metric3.metric(
            "Maximum relative energy error",
            (
                f"{single_results['maximum_relative_energy_change']:.3e}"
            )
        )

        critical_energy = 2.0 * mass * 9.81 * length
        initial_energy = single_results["initial_total_energy"]
        energy_tolerance = 1.0e-6 * max(1.0, critical_energy)

        if abs(initial_energy - critical_energy) <= energy_tolerance:
            detected_regime = "Separatrix"
        elif initial_energy < critical_energy:
            detected_regime = "Oscillation"
        else:
            detected_regime = "Rotation"

        st.success(
            "Detected motion regime: " + detected_regime
        )

        st.info(
            "Changing mass changes the energy values, "
            "but it does not change the ideal pendulum trajectory."
        )

# ============================================================
# Section 5: Double-pendulum interface
# ============================================================

elif model_choice == "Part III: Double Pendulum":

    st.header(
        "Double Pendulum"
    )

    st.markdown(
        """
        The double pendulum is a nonlinear coupled system.
        Small changes in its initial conditions can eventually
        generate very different trajectories.
        """
    )

    with st.sidebar.form(
        "double_pendulum_parameters"
    ):

        st.subheader(
            "Double-Pendulum Parameters"
        )

        theta1_degrees = st.slider(
            "Initial angle θ₁ (degrees)",
            min_value=-180.0,
            max_value=180.0,
            value=120.0,
            step=1.0
        )

        theta2_degrees = st.slider(
            "Initial angle θ₂ (degrees)",
            min_value=-180.0,
            max_value=180.0,
            value=-70.0,
            step=1.0
        )

        omega1_initial = st.slider(
            "Initial angular velocity ω₁ (rad/s)",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1
        )

        omega2_initial = st.slider(
            "Initial angular velocity ω₂ (rad/s)",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1
        )

        m1 = st.slider(
            "Mass m₁ (kg)",
            min_value=0.4,
            max_value=1.6,
            value=1.0,
            step=0.1
        )

        m2 = st.slider(
            "Mass m₂ (kg)",
            min_value=0.4,
            max_value=1.6,
            value=1.0,
            step=0.1
        )

        L1 = st.slider(
            "Length L₁ (m)",
            min_value=2.0 / 3.0,
            max_value=4.0 / 3.0,
            value=1.0,
            step=1.0 / 30.0,
            format="%.2f"
        )

        L2 = st.slider(
            "Length L₂ (m)",
            min_value=2.0 / 3.0,
            max_value=4.0 / 3.0,
            value=1.0,
            step=1.0 / 30.0,
            format="%.2f"
        )

        double_total_time = st.slider(
            "Simulation time (s)",
            min_value=1.0,
            max_value=20.0,
            value=20.0,
            step=1.0
        )

        double_color_map = st.selectbox(
            "Trajectory colour map",
            [
                "turbo",
                "plasma",
                "viridis",
                "inferno",
                "magma",
                "cividis"
            ],
            key="double_colour"
        )

        generate_double_static = (
            st.form_submit_button(
                "Generate Static Artwork",
                type="primary"
            )
        )

        generate_double_animation = (
            st.form_submit_button(
                "Generate Animation"
            )
        )

    if (
        generate_double_static
        or generate_double_animation
    ):

        with st.spinner(
            "Running the double-pendulum simulation..."
        ):

            double_results = (
                simulate_double_pendulum_degrees(
                    theta1_degrees=theta1_degrees,
                    theta2_degrees=theta2_degrees,
                    omega1_initial=omega1_initial,
                    omega2_initial=omega2_initial,
                    m1=m1,
                    m2=m2,
                    L1=L1,
                    L2=L2,
                    total_time=double_total_time
                )
            )

        # ----------------------------------------------------
        # Static artwork
        # ----------------------------------------------------

        if generate_double_static:

            with matplotlib_lock:
                double_figure, _ = (
                    create_physical_trajectory_figure(
                        results=double_results,
                        color_map_name=double_color_map
                    )
                )

                double_png_buffer = BytesIO()

                double_figure.savefig(
                    double_png_buffer,
                    format="png",
                    dpi=300,
                    bbox_inches="tight"
                )

                double_png_bytes = (
                    double_png_buffer.getvalue()
                )

                st.pyplot(
                    double_figure,
                    width="stretch"
                )

                st.download_button(
                    label="Download Double-Pendulum PNG",
                    data=double_png_bytes,
                    file_name=(
                        "double_pendulum_artwork.png"
                    ),
                    mime="image/png"
                )

                double_png_buffer.close()

                plt.close(
                    double_figure
                )
        # ----------------------------------------------------
        # Animated artwork
        # ----------------------------------------------------

        if generate_double_animation:

            double_frame_skip = choose_frame_skip(
                len(double_results["time"])
            )

            with st.spinner(
                "Rendering the double-pendulum GIF. "
                "This may take a few minutes..."
            ):

                with matplotlib_lock:
                    (
                        double_animation,
                        double_animation_figure
                    ) = create_double_pendulum_animation(
                        results=double_results,
                        frame_skip=double_frame_skip,
                        color_map_name=double_color_map
                    )

                    double_gif_bytes = (
                        animation_to_gif_bytes(
                            double_animation,
                            fps=30,
                            dpi=90
                        )
                    )

                    plt.close(
                        double_animation_figure
                    )

            st.image(
                double_gif_bytes
            )

            st.download_button(
                label="Download Double-Pendulum GIF",
                data=double_gif_bytes,
                file_name=(
                    "double_pendulum_animation.gif"
                ),
                mime="image/gif"
            )

        # ----------------------------------------------------
        # Numerical validation
        # ----------------------------------------------------

        metric1, metric2, metric3 = st.columns(
            3
        )

        metric1.metric(
            "Initial total energy",
            (
                f"{double_results['initial_total_energy']:.5f} J"
            )
        )

        metric2.metric(
            "Maximum absolute error",
            (
                f"{double_results['maximum_absolute_energy_error']:.3e} J"
            )
        )

        metric3.metric(
            "Maximum relative error",
            (
                f"{double_results['maximum_relative_energy_error']:.3e}"
            )
        )

        st.caption(
            "The RK4 step size is fixed internally at 0.005 s. "
            "It is hidden from public website controls."
        )

# ============================================================
# Section 6: Forced-and-damped-pendulum interface
# ============================================================

elif model_choice == "Part II: Forced and Damped Pendulum":

    st.header(
        "Part II: Forced and Damped Pendulum"
    )

    st.markdown(
        r"""
        The dimensionless model is

        $$
        \frac{d^2\theta}{d\tau^2}
        +q\frac{d\theta}{d\tau}
        +\sin\theta
        =F\cos(\Omega\tau).
        $$

        All Part II times, angular velocities, and frequencies are
        dimensionless. Colour represents time, line width represents
        \(\lvert\omega\rvert\), and opacity represents instantaneous
        mechanical energy.
        """
    )

    response_defaults = {
        "Reference complex response": (
            0.2, 0.0, 0.5, 1.20, 2.0 / 3.0
        ),
        "Period 1": (
            0.2, 0.0, 0.5, 1.05, 2.0 / 3.0
        ),
        "Period 2": (
            0.2, 0.0, 0.5, 1.07, 2.0 / 3.0
        ),
        "Period 4": (
            0.2, 0.0, 0.5, 1.0815, 2.0 / 3.0
        ),
        "Custom": (
            0.2, 0.0, 0.5, 1.20, 2.0 / 3.0
        )
    }

    def load_part2_preset():
        selected_preset = st.session_state[
            "part2_preset"
        ]

        if selected_preset == "Custom":
            return

        (
            preset_theta,
            preset_initial_omega,
            preset_q,
            preset_f,
            preset_drive_omega
        ) = response_defaults[selected_preset]

        st.session_state["part2_theta"] = preset_theta
        st.session_state["part2_initial_omega"] = (
            preset_initial_omega
        )
        st.session_state["part2_q"] = preset_q
        st.session_state["part2_f"] = preset_f
        st.session_state["part2_drive_omega"] = (
            preset_drive_omega
        )

    response_preset = st.sidebar.selectbox(
        "Starting parameter preset",
        [
            "Reference complex response",
            "Period 1",
            "Period 2",
            "Period 4",
            "Custom"
        ],
        key="part2_preset",
        on_change=load_part2_preset
    )

    (
        default_theta,
        default_initial_omega,
        default_q,
        default_f,
        default_omega_drive
    ) = response_defaults[response_preset]

    with st.sidebar.form(
        "forced_damped_parameters"
    ):

        st.subheader(
            "Part II Parameters"
        )

        forced_theta_initial = st.slider(
            "Initial angle θ₀ (rad)",
            min_value=-3.1416,
            max_value=3.1416,
            value=float(default_theta),
            step=0.01,
            format="%.4f",
            key="part2_theta",
            disabled=(response_preset != "Custom")
        )

        forced_omega_initial = st.slider(
            "Initial angular velocity ω₀",
            min_value=-4.0,
            max_value=4.0,
            value=float(default_initial_omega),
            step=0.05,
            key="part2_initial_omega",
            disabled=(response_preset != "Custom")
        )

        forced_damping = st.slider(
            "Dimensionless damping q",
            min_value=0.0,
            max_value=1.0,
            value=float(default_q),
            step=0.01,
            key="part2_q",
            disabled=(response_preset != "Custom")
        )

        forced_amplitude = st.slider(
            "Forcing amplitude F",
            min_value=0.0,
            max_value=2.0,
            value=float(default_f),
            step=0.005,
            format="%.4f",
            key="part2_f",
            disabled=(response_preset != "Custom")
        )

        forced_frequency = st.slider(
            "Driving frequency Ω",
            min_value=0.1,
            max_value=2.0,
            value=float(default_omega_drive),
            step=0.01,
            key="part2_drive_omega",
            disabled=(response_preset != "Custom")
        )

        forced_total_time = st.slider(
            "Dimensionless simulation time τ",
            min_value=2.0,
            max_value=60.0,
            value=20.0,
            step=1.0
        )

        forced_color_map = st.selectbox(
            "Trajectory colour map",
            [
                "turbo",
                "plasma",
                "viridis",
                "inferno",
                "magma",
                "cividis"
            ],
            key="forced_colour"
        )

        minimum_line_width = st.slider(
            "Minimum line width",
            min_value=0.1,
            max_value=2.0,
            value=0.6,
            step=0.1
        )

        maximum_line_width = st.slider(
            "Maximum line width",
            min_value=2.0,
            max_value=8.0,
            value=4.0,
            step=0.2
        )

        minimum_alpha = st.slider(
            "Minimum opacity",
            min_value=0.05,
            max_value=0.8,
            value=0.20,
            step=0.05
        )

        maximum_alpha = st.slider(
            "Maximum opacity",
            min_value=0.2,
            max_value=1.0,
            value=1.0,
            step=0.05
        )

        generate_forced_static = st.form_submit_button(
            "Generate Static Artwork",
            type="primary"
        )

        generate_forced_animation = st.form_submit_button(
            "Generate Animation"
        )

        generate_damping_comparison = st.form_submit_button(
            "Generate Damping Comparison"
        )

    if (
        generate_forced_static
        or generate_forced_animation
        or generate_damping_comparison
    ):

        if maximum_line_width < minimum_line_width:
            st.error(
                "Maximum width must be at least the minimum width."
            )
            st.stop()

        if maximum_alpha < minimum_alpha:
            st.error(
                "Maximum opacity must be at least the minimum opacity."
            )
            st.stop()

        if generate_forced_static or generate_forced_animation:
            with st.spinner(
                "Running the forced and damped simulation..."
            ):
                forced_results = simulate_forced_damped_pendulum(
                    theta_initial=forced_theta_initial,
                    omega_initial=forced_omega_initial,
                    damping=forced_damping,
                    drive_amplitude=forced_amplitude,
                    drive_frequency=forced_frequency,
                    total_time=forced_total_time,
                    step_size=0.01
                )

        visual_settings = {
            "color_map_name": forced_color_map,
            "minimum_line_width": minimum_line_width,
            "maximum_line_width": maximum_line_width,
            "minimum_alpha": minimum_alpha,
            "maximum_alpha": maximum_alpha
        }

        if generate_forced_static or generate_forced_animation:
            visual_arguments = {
                "results": forced_results,
                **visual_settings
            }

        if generate_forced_static:
            with matplotlib_lock:
                forced_figure, _, _ = (
                    create_forced_damped_pendulum_figure(
                        **visual_arguments
                    )
                )

                forced_png_buffer = BytesIO()
                forced_figure.savefig(
                    forced_png_buffer,
                    format="png",
                    dpi=300,
                    bbox_inches="tight"
                )
                forced_png_bytes = forced_png_buffer.getvalue()

                st.pyplot(
                    forced_figure,
                    width="stretch"
                )
                st.download_button(
                    label="Download Part II PNG",
                    data=forced_png_bytes,
                    file_name="forced_damped_pendulum_artwork.png",
                    mime="image/png"
                )

                forced_png_buffer.close()
                plt.close(forced_figure)

        if generate_forced_animation:
            forced_frame_skip = choose_frame_skip(
                len(forced_results["time"])
            )

            with st.spinner(
                "Rendering the Part II GIF. This may take a few minutes..."
            ):
                with matplotlib_lock:
                    forced_animation, forced_animation_figure = (
                        create_forced_damped_pendulum_animation(
                            frame_skip=forced_frame_skip,
                            **visual_arguments
                        )
                    )
                    forced_gif_bytes = animation_to_gif_bytes(
                        forced_animation,
                        fps=30,
                        dpi=90
                    )
                    plt.close(forced_animation_figure)

            st.image(forced_gif_bytes)
            st.download_button(
                label="Download Part II GIF",
                data=forced_gif_bytes,
                file_name="forced_damped_pendulum_animation.gif",
                mime="image/gif"
            )

            st.caption(
                "Red markers are Poincaré samples recorded once per "
                "driving period. Longer simulations display more samples."
            )

        if generate_damping_comparison:
            with st.spinner(
                "Rendering the damping-comparison GIF..."
            ):
                undamped_results = simulate_forced_damped_pendulum(
                    theta_initial=1.5,
                    omega_initial=0.0,
                    damping=0.0,
                    drive_amplitude=0.0,
                    drive_frequency=2.0 / 3.0,
                    total_time=forced_total_time,
                    step_size=0.01
                )
                damped_results = simulate_forced_damped_pendulum(
                    theta_initial=1.5,
                    omega_initial=0.0,
                    damping=0.2,
                    drive_amplitude=0.0,
                    drive_frequency=2.0 / 3.0,
                    total_time=forced_total_time,
                    step_size=0.01
                )
                comparison_frame_skip = choose_frame_skip(
                    len(undamped_results["time"])
                )
                with matplotlib_lock:
                    comparison_animation, comparison_figure = (
                        create_damping_comparison_animation(
                            undamped_results=undamped_results,
                            damped_results=damped_results,
                            frame_skip=comparison_frame_skip,
                            **visual_settings
                        )
                    )
                    comparison_gif_bytes = animation_to_gif_bytes(
                        comparison_animation,
                        fps=30,
                        dpi=85
                    )
                    plt.close(comparison_figure)

            st.image(comparison_gif_bytes)
            st.download_button(
                label="Download Damping Comparison GIF",
                data=comparison_gif_bytes,
                file_name="part2_damping_comparison.gif",
                mime="image/gif"
            )
            st.caption(
                "Both systems start from θ₀ = 1.5 and ω₀ = 0 with "
                "F = 0. The left system has q = 0; the right system "
                "has q = 0.2."
            )

        if generate_forced_static or generate_forced_animation:
            energy_values = forced_results["mechanical_energy"]
            metric1, metric2, metric3 = st.columns(3)
            metric1.metric(
                "Minimum mechanical energy",
                f"{np.min(energy_values):.5f}"
            )
            metric2.metric(
                "Maximum mechanical energy",
                f"{np.max(energy_values):.5f}"
            )
            metric3.metric(
                "Actual numerical step",
                f"{forced_results['step_size']:.5f}"
            )

        if response_preset != "Custom":
            st.caption(
                "The locked initial state and dynamical parameters define "
                "this published response preset. Simulation time and visual "
                "settings affect only how much of it is displayed."
            )
