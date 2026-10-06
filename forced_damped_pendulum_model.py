import numpy as np


# ============================================================
# Section 1: Forced and damped pendulum equations
# ============================================================

def forced_damped_pendulum_derivatives(
    time,
    state,
    damping,
    drive_amplitude,
    drive_frequency
):
    """
    Return the derivatives of the dimensionless forced and
    damped pendulum system.

    State order:
        state[0] = theta
        state[1] = omega

    Governing equation:
        theta'' + q theta' + sin(theta)
        = F cos(Omega t)

    Parameters:
        damping = q
        drive_amplitude = F
        drive_frequency = Omega
    """

    theta, omega = state

    theta_derivative = omega

    omega_derivative = (
        -np.sin(theta)
        - damping * omega
        + drive_amplitude
        * np.cos(drive_frequency * time)
    )

    return np.array(
        [
            theta_derivative,
            omega_derivative
        ],
        dtype=float
    )


# ============================================================
# Section 2: Fourth-order Runge--Kutta step
# ============================================================

def rk4_step(
    derivative_function,
    current_time,
    current_state,
    step_size
):
    """Perform one fourth-order Runge--Kutta step."""

    k1 = derivative_function(
        current_time,
        current_state
    )

    k2 = derivative_function(
        current_time + 0.5 * step_size,
        current_state + 0.5 * step_size * k1
    )

    k3 = derivative_function(
        current_time + 0.5 * step_size,
        current_state + 0.5 * step_size * k2
    )

    k4 = derivative_function(
        current_time + step_size,
        current_state + step_size * k3
    )

    return current_state + (
        step_size / 6.0
    ) * (
        k1
        + 2.0 * k2
        + 2.0 * k3
        + k4
    )


# ============================================================
# Section 3: Complete simulation function
# ============================================================

def simulate_forced_damped_pendulum(
    theta_initial,
    omega_initial=0.0,
    damping=0.5,
    drive_amplitude=1.2,
    drive_frequency=2.0 / 3.0,
    total_time=60.0,
    step_size=0.01
):
    """
    Simulate the dimensionless forced and damped pendulum.

    Equation:
        theta'' + q theta' + sin(theta)
        = F cos(Omega t)

    The time variable is dimensionless:
        tau = t * sqrt(g / L)

    Consequently, the angular velocity is measured with
    respect to dimensionless time.
    """

    # --------------------------------------------------------
    # Parameter validation
    # --------------------------------------------------------

    if damping < 0.0:
        raise ValueError(
            "The damping coefficient must be non-negative."
        )

    if drive_amplitude < 0.0:
        raise ValueError(
            "The driving amplitude must be non-negative."
        )

    if drive_frequency <= 0.0:
        raise ValueError(
            "The driving frequency must be greater than zero."
        )

    if total_time <= 0.0:
        raise ValueError(
            "The total simulation time must be greater than zero."
        )

    if step_size <= 0.0:
        raise ValueError(
            "The step size must be greater than zero."
        )

    # --------------------------------------------------------
    # Time values
    # --------------------------------------------------------

    number_of_steps = int(
        np.ceil(total_time / step_size)
    )

    time_values = np.linspace(
        0.0,
        total_time,
        number_of_steps + 1
    )

    actual_step_size = (
        time_values[1] - time_values[0]
    )

    # --------------------------------------------------------
    # State values
    # --------------------------------------------------------

    initial_state = np.array(
        [
            theta_initial,
            omega_initial
        ],
        dtype=float
    )

    state_values = np.zeros(
        (number_of_steps + 1, 2),
        dtype=float
    )

    state_values[0] = initial_state

    # Connect the selected parameters to the derivative model.
    def model(current_time, current_state):

        return forced_damped_pendulum_derivatives(
            current_time,
            current_state,
            damping,
            drive_amplitude,
            drive_frequency
        )

    # --------------------------------------------------------
    # RK4 integration
    # --------------------------------------------------------

    for index in range(number_of_steps):

        state_values[index + 1] = rk4_step(
            model,
            time_values[index],
            state_values[index],
            actual_step_size
        )

    theta_values = state_values[:, 0]
    omega_values = state_values[:, 1]

    # Wrap the angle into [-pi, pi).
    wrapped_theta_values = (
        theta_values + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi

    # --------------------------------------------------------
    # Dimensionless mechanical energy
    # --------------------------------------------------------

    mechanical_energy = (
        0.5 * omega_values**2
        + 1.0
        - np.cos(theta_values)
    )

    # The phase of the external driving force.
    drive_phase = (
        drive_frequency * time_values
    ) % (
        2.0 * np.pi
    )

    return {
        "time": time_values,
        "states": state_values,
        "theta": theta_values,
        "wrapped_theta": wrapped_theta_values,
        "omega": omega_values,
        "mechanical_energy": mechanical_energy,
        "drive_phase": drive_phase,
        "step_size": actual_step_size,
        "parameters": {
            "theta_initial": theta_initial,
            "omega_initial": omega_initial,
            "damping": damping,
            "drive_amplitude": drive_amplitude,
            "drive_frequency": drive_frequency,
            "total_time": total_time,
            "step_size": actual_step_size
        }
    }


# ============================================================
# Section 4: Basic model test
# ============================================================

if __name__ == "__main__":

    results = simulate_forced_damped_pendulum(
        theta_initial=0.2,
        omega_initial=0.0,
        damping=0.5,
        drive_amplitude=1.2,
        drive_frequency=2.0 / 3.0,
        total_time=60.0,
        step_size=0.01
    )

    print(
        "Forced and damped pendulum model test"
    )

    print(
        "-----------------------------------"
    )

    print(
        "Number of time points:",
        len(results["time"])
    )

    print(
        "Final simulation time:",
        results["time"][-1]
    )

    print(
        "Initial state:",
        results["states"][0]
    )

    print(
        "Final state:",
        results["states"][-1]
    )

    print(
        "Minimum wrapped angle:",
        np.min(results["wrapped_theta"])
    )

    print(
        "Maximum wrapped angle:",
        np.max(results["wrapped_theta"])
    )

    print(
        "Minimum mechanical energy:",
        np.min(results["mechanical_energy"])
    )

    print(
        "Maximum mechanical energy:",
        np.max(results["mechanical_energy"])
    )

    # Basic numerical checks.
    assert np.all(
        np.isfinite(results["states"])
    )

    assert np.isclose(
        results["time"][-1],
        60.0
    )

    assert np.allclose(
        results["states"][0],
        [0.2, 0.0]
    )

    assert np.all(
        results["wrapped_theta"] >= -np.pi
    )

    assert np.all(
        results["wrapped_theta"] < np.pi
    )

    print()
    print("All basic model checks passed.")