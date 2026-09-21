# ============================================================
# Section 1: Imports
# ============================================================

from math import ceil
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock

import matplotlib.pyplot as plt
import streamlit as st

from single_pendulum_model import (
    simulate_single_pendulum_degrees
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
        "Single Pendulum",
        "Double Pendulum"
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

if model_choice == "Single Pendulum":

    st.header(
        "Single Pendulum"
    )

    st.markdown(
        r"""
        The nonlinear damped single-pendulum model is

        $$
        \ddot{\theta}
        + \gamma\dot{\theta}
        + \frac{g}{L}\sin(\theta)=0.
        $$
        """
    )

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
            value=85.94,
            step=1.0
        )

        omega_initial = st.slider(
            "Initial angular velocity ω₀ (rad/s)",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1
        )

        length = st.slider(
            "Pendulum length L (m)",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1
        )

        mass = st.slider(
            "Pendulum mass m (kg)",
            min_value=0.4,
            max_value=1.6,
            value=1.0,
            step=0.1
        )

        damping = st.slider(
            "Damping γ",
            min_value=0.0,
            max_value=0.6,
            value=0.08,
            step=0.01
        )

        total_time = st.slider(
            "Simulation time (s)",
            min_value=1.0,
            max_value=20.0,
            value=13.0,
            step=1.0
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

        with st.spinner(
            "Running the single-pendulum simulation..."
        ):

            single_results = (
                simulate_single_pendulum_degrees(
                    theta_degrees=theta_degrees,
                    omega_initial=omega_initial,
                    length=length,
                    mass=mass,
                    damping=damping,
                    total_time=total_time
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
                        color_map_name=color_map_name
                    )
                )

                st.pyplot(
                    single_figure,
                    width="stretch"
                )

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
                        color_map_name=color_map_name
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

        if damping == 0.0:
            metric3.metric(
                "Maximum relative energy error",
                (
                    f"{single_results['maximum_relative_energy_change']:.3e}"
                )
            )

        else:
            metric3.metric(
                "Energy dissipated",
                (
                    f"{100 * single_results['final_relative_energy_loss']:.2f}%"
                )
            )

        st.info(
            "Changing mass changes the energy values, "
            "but it does not change the ideal pendulum trajectory."
        )

# ============================================================
# Section 5: Double-pendulum interface
# ============================================================

else:

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

                st.pyplot(
                    double_figure,
                    width="stretch"
                )

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