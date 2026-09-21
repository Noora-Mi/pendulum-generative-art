"""
Reusable mathematical model for the double pendulum.

This file contains the physical equations and numerical solver.
It does not create plots or animations.
"""

import numpy as np

def angles_to_coordinates(theta1, theta2, L1, L2):
    """
    Convert the two pendulum angles into Cartesian coordinates.

    Angles are measured from the downward vertical direction.

    Parameters
    ----------
    theta1 : float or numpy array
        Angle of the first pendulum in radians.
    theta2 : float or numpy array
        Angle of the second pendulum in radians.
    L1 : float
        Length of the first pendulum.
    L2 : float
        Length of the second pendulum.

    Returns
    -------
    x1, y1, x2, y2
        Cartesian coordinates of the two bobs.
    """

    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)

    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    return x1, y1, x2, y2

def double_pendulum_derivatives(
    t,
    state,
    m1,
    m2,
    L1,
    L2,
    g=9.81
):
    """
    Calculate the time derivatives of the double-pendulum state.

    State order:
    [theta1, omega1, theta2, omega2]

    Parameters
    ----------
    t : float
        Current simulation time.
    state : numpy array
        Current angles and angular velocities.
    m1, m2 : float
        Masses of the two bobs.
    L1, L2 : float
        Lengths of the two pendulum segments.
    g : float
        Gravitational acceleration.

    Returns
    -------
    numpy array
        [omega1, alpha1, omega2, alpha2]
    """

    theta1, omega1, theta2, omega2 = state

    angle_difference = theta1 - theta2

    cosine_difference = np.cos(angle_difference)
    sine_difference = np.sin(angle_difference)

    # Mass matrix from the coupled equations of motion
    mass_matrix = np.array([
        [
            (m1 + m2) * L1,
            m2 * L2 * cosine_difference
        ],
        [
            L1 * cosine_difference,
            L2
        ]
    ])

    # Right-hand side of the coupled equations
    force_vector = np.array([
        (
            -(m1 + m2) * g * np.sin(theta1)
            - m2
            * L2
            * omega2 ** 2
            * sine_difference
        ),
        (
            -g * np.sin(theta2)
            + L1
            * omega1 ** 2
            * sine_difference
        )
    ])

    # Solve for the two angular accelerations
    alpha1, alpha2 = np.linalg.solve(
        mass_matrix,
        force_vector
    )

    return np.array([
        omega1,
        alpha1,
        omega2,
        alpha2
    ])
    
def rk4_step(function, t, state, step_size):
    """
    Advance the system by one time step using
    the fourth-order Runge-Kutta method.

    Parameters
    ----------
    function : callable
        Function that calculates the state derivatives.
    t : float
        Current simulation time.
    state : numpy array
        Current state of the system.
    step_size : float
        Numerical integration step size.

    Returns
    -------
    numpy array
        State of the system after one time step.
    """

    k1 = function(
        t,
        state
    )

    k2 = function(
        t + step_size / 2,
        state + step_size * k1 / 2
    )

    k3 = function(
        t + step_size / 2,
        state + step_size * k2 / 2
    )

    k4 = function(
        t + step_size,
        state + step_size * k3
    )

    next_state = state + (
        step_size / 6
    ) * (
        k1
        + 2 * k2
        + 2 * k3
        + k4
    )

    return next_state


# ============================================================
# Section 4: Complete double-pendulum simulation
# ============================================================

def simulate_double_pendulum(
    theta1_initial,
    theta2_initial,
    omega1_initial=0.0,
    omega2_initial=0.0,
    m1=1.0,
    m2=1.0,
    L1=1.0,
    L2=1.0,
    total_time=20.0,
    step_size=0.005,
    g=9.81
):
    """
    Simulate a double pendulum using the RK4 method.

    All angles must be supplied in radians.
    Angular velocities are measured in radians per second.
    """

    # Check that the physical parameters are valid.
    if m1 <= 0 or m2 <= 0:
        raise ValueError("Both masses must be greater than zero.")

    if L1 <= 0 or L2 <= 0:
        raise ValueError("Both lengths must be greater than zero.")

    if total_time <= 0:
        raise ValueError("The total simulation time must be greater than zero.")

    if step_size <= 0:
        raise ValueError("The step size must be greater than zero.")

    # Construct the time array.
    number_of_steps = int(np.ceil(total_time / step_size))

    time_values = np.linspace(
        0.0,
        total_time,
        number_of_steps + 1
    )

    actual_step_size = time_values[1] - time_values[0]

    # State order:
    # [theta1, omega1, theta2, omega2]
    initial_state = np.array(
        [
            theta1_initial,
            omega1_initial,
            theta2_initial,
            omega2_initial
        ],
        dtype=float
    )

    state_values = np.zeros(
        (number_of_steps + 1, 4)
    )

    state_values[0] = initial_state

    # Connect the selected parameters to the derivative function.
    def model(current_time, current_state):
        return double_pendulum_derivatives(
            current_time,
            current_state,
            m1,
            m2,
            L1,
            L2,
            g
        )

    # Perform the full RK4 simulation.
    for index in range(number_of_steps):
        state_values[index + 1] = rk4_step(
            model,
            time_values[index],
            state_values[index],
            actual_step_size
        )

    theta1_values = state_values[:, 0]
    omega1_values = state_values[:, 1]
    theta2_values = state_values[:, 2]
    omega2_values = state_values[:, 3]

    x1_values, y1_values, x2_values, y2_values = (
        angles_to_coordinates(
            theta1_values,
            theta2_values,
            L1,
            L2
        )
    )

    # Calculate energy throughout the simulation.
    (
        kinetic_energy,
        potential_energy,
        total_energy
    ) = calculate_energies(
        theta1_values,
        omega1_values,
        theta2_values,
        omega2_values,
        m1,
        m2,
        L1,
        L2,
        g
    )

    initial_total_energy = total_energy[0]

    absolute_energy_error = np.abs(
        total_energy - initial_total_energy
    )

    maximum_absolute_energy_error = np.max(
        absolute_energy_error
    )

    if np.abs(initial_total_energy) > 1e-14:
        relative_energy_error = (
            absolute_energy_error
            / np.abs(initial_total_energy)
        )

        maximum_relative_energy_error = np.max(
            relative_energy_error
        )

    else:
        relative_energy_error = np.full_like(
            total_energy,
            np.nan
        )

        maximum_relative_energy_error = np.nan

    return {
        "time": time_values,
        "states": state_values,
        "theta1": theta1_values,
        "omega1": omega1_values,
        "theta2": theta2_values,
        "omega2": omega2_values,
        "x1": x1_values,
        "y1": y1_values,
        "x2": x2_values,
        "y2": y2_values,
        "kinetic_energy": kinetic_energy,
        "potential_energy": potential_energy,
        "total_energy": total_energy,
        "absolute_energy_error": absolute_energy_error,
        "relative_energy_error": relative_energy_error,
        "initial_total_energy": initial_total_energy,
        "maximum_absolute_energy_error": (
            maximum_absolute_energy_error
        ),
        "maximum_relative_energy_error": (
            maximum_relative_energy_error
        ),
        "parameters": {
            "theta1_initial": theta1_initial,
            "theta2_initial": theta2_initial,
            "omega1_initial": omega1_initial,
            "omega2_initial": omega2_initial,
            "m1": m1,
            "m2": m2,
            "L1": L1,
            "L2": L2,
            "total_time": total_time,
            "g": g
        },
        "step_size": actual_step_size
    }

# ============================================================
# Section 5: Energy calculation
# ============================================================

def calculate_energies(
    theta1,
    omega1,
    theta2,
    omega2,
    m1,
    m2,
    L1,
    L2,
    g=9.81
):
    """
    Calculate the kinetic, potential, and total energy
    of the double pendulum.

    The lowest vertical configuration is used as the
    zero level of potential energy.
    """

    angle_difference = theta1 - theta2

    kinetic_energy = (
        0.5
        * (m1 + m2)
        * L1 ** 2
        * omega1 ** 2
        + 0.5
        * m2
        * L2 ** 2
        * omega2 ** 2
        + m2
        * L1
        * L2
        * omega1
        * omega2
        * np.cos(angle_difference)
    )

    potential_energy = (
        (m1 + m2)
        * g
        * L1
        * (1 - np.cos(theta1))
        + m2
        * g
        * L2
        * (1 - np.cos(theta2))
    )

    total_energy = (
        kinetic_energy
        + potential_energy
    )

    return (
        kinetic_energy,
        potential_energy,
        total_energy
    )
    
    
# ============================================================
# Section 6: User-facing simulation in degrees
# ============================================================

def simulate_double_pendulum_degrees(
    theta1_degrees,
    theta2_degrees,
    omega1_initial=0.0,
    omega2_initial=0.0,
    m1=1.0,
    m2=1.0,
    L1=1.0,
    L2=1.0,
    total_time=20.0,
    g=9.81
):
    """
    Run a double-pendulum simulation using initial angles
    supplied in degrees.

    The website will use this function so that users do not
    need to enter angles in radians.

    The numerical step size is fixed internally at 0.005 s.
    """

    # Validate the parameter ranges used by the public website.
    if not -180.0 <= theta1_degrees <= 180.0:
        raise ValueError(
            "theta1 must be between -180 and 180 degrees."
        )

    if not -180.0 <= theta2_degrees <= 180.0:
        raise ValueError(
            "theta2 must be between -180 and 180 degrees."
        )

    if not -2.0 <= omega1_initial <= 2.0:
        raise ValueError(
            "omega1 must be between -2 and 2 rad/s."
        )

    if not -2.0 <= omega2_initial <= 2.0:
        raise ValueError(
            "omega2 must be between -2 and 2 rad/s."
        )

    if not 0.4 <= m1 <= 1.6:
        raise ValueError(
            "m1 must be between 0.4 and 1.6 kg."
        )

    if not 0.4 <= m2 <= 1.6:
        raise ValueError(
            "m2 must be between 0.4 and 1.6 kg."
        )

    minimum_length = 2.0 / 3.0
    maximum_length = 4.0 / 3.0

    if not minimum_length <= L1 <= maximum_length:
        raise ValueError(
            "L1 must be between 2/3 and 4/3 metres."
        )

    if not minimum_length <= L2 <= maximum_length:
        raise ValueError(
            "L2 must be between 2/3 and 4/3 metres."
        )

    if not 1.0 <= total_time <= 20.0:
        raise ValueError(
            "total_time must be between 1 and 20 seconds."
        )

    theta1_radians = np.deg2rad(
        theta1_degrees
    )

    theta2_radians = np.deg2rad(
        theta2_degrees
    )

    results = simulate_double_pendulum(
        theta1_initial=theta1_radians,
        theta2_initial=theta2_radians,
        omega1_initial=omega1_initial,
        omega2_initial=omega2_initial,
        m1=m1,
        m2=m2,
        L1=L1,
        L2=L2,
        total_time=total_time,
        step_size=0.005,
        g=g
    )

    results["theta1_initial_degrees"] = (
        theta1_degrees
    )

    results["theta2_initial_degrees"] = (
        theta2_degrees
    )

    return results