"""
Reusable mathematical model for a single pendulum.

This file contains the physical equations and numerical solver.
It does not create plots or animations.
"""

import numpy as np


# ============================================================
# Section 1: Convert angle to physical coordinates
# ============================================================

def angle_to_coordinates(
    theta,
    length
):
    """
    Convert pendulum angle into Cartesian coordinates.

    The angle is measured from the downward vertical.
    """

    x = length * np.sin(theta)
    y = -length * np.cos(theta)

    return x, y


# ============================================================
# Section 2: Single-pendulum differential equation
# ============================================================

def single_pendulum_derivatives(
    t,
    state,
    length,
    damping=0.0,
    g=9.81
):
    """
    Calculate the derivatives of a single pendulum.

    State order:
    [theta, omega]

    The equation is:

    theta'' + damping * theta'
    + (g / length) * sin(theta) = 0
    """

    theta, omega = state

    angular_acceleration = (
        -damping * omega
        - (g / length) * np.sin(theta)
    )

    return np.array(
        [
            omega,
            angular_acceleration
        ]
    )


# ============================================================
# Section 3: One RK4 integration step
# ============================================================

def rk4_step(
    function,
    t,
    state,
    step_size
):
    """
    Advance the system by one RK4 time step.
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
# Section 4: Energy calculation
# ============================================================

def calculate_single_pendulum_energies(
    theta,
    omega,
    mass,
    length,
    g=9.81
):
    """
    Calculate kinetic, potential, and total energy.
    """

    kinetic_energy = (
        0.5
        * mass
        * length ** 2
        * omega ** 2
    )

    potential_energy = (
        mass
        * g
        * length
        * (1 - np.cos(theta))
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
# Section 5: Complete single-pendulum simulation
# ============================================================

def simulate_single_pendulum(
    theta_initial,
    omega_initial=0.0,
    length=1.0,
    mass=1.0,
    damping=0.0,
    total_time=13.0,
    step_size=0.01,
    g=9.81
):
    """
    Simulate a nonlinear single pendulum using RK4.

    The initial angle must be supplied in radians.
    """

    if length <= 0:
        raise ValueError(
            "Pendulum length must be greater than zero."
        )

    if mass <= 0:
        raise ValueError(
            "Pendulum mass must be greater than zero."
        )

    if damping < 0:
        raise ValueError(
            "Damping cannot be negative."
        )

    if total_time <= 0:
        raise ValueError(
            "Total time must be greater than zero."
        )

    if step_size <= 0:
        raise ValueError(
            "Step size must be greater than zero."
        )

    number_of_steps = int(
        np.ceil(total_time / step_size)
    )

    time_values = np.linspace(
        0.0,
        total_time,
        number_of_steps + 1
    )

    actual_step_size = (
        time_values[1]
        - time_values[0]
    )

    state_values = np.zeros(
        (number_of_steps + 1, 2)
    )

    state_values[0] = [
        theta_initial,
        omega_initial
    ]

    def model(
        current_time,
        current_state
    ):
        return single_pendulum_derivatives(
            current_time,
            current_state,
            length=length,
            damping=damping,
            g=g
        )

    for index in range(number_of_steps):
        state_values[index + 1] = rk4_step(
            model,
            time_values[index],
            state_values[index],
            actual_step_size
        )

    theta_values = state_values[:, 0]
    omega_values = state_values[:, 1]

    x_values, y_values = angle_to_coordinates(
        theta_values,
        length
    )

    (
        kinetic_energy,
        potential_energy,
        total_energy
    ) = calculate_single_pendulum_energies(
        theta_values,
        omega_values,
        mass,
        length,
        g
    )

    initial_total_energy = total_energy[0]
    final_total_energy = total_energy[-1]

    absolute_energy_change = np.abs(
        total_energy - initial_total_energy
    )

    maximum_absolute_energy_change = np.max(
        absolute_energy_change
    )

    if np.abs(initial_total_energy) > 1e-14:
        relative_energy_change = (
            absolute_energy_change
            / np.abs(initial_total_energy)
        )

        maximum_relative_energy_change = np.max(
            relative_energy_change
        )

        final_relative_energy_loss = (
            initial_total_energy
            - final_total_energy
        ) / np.abs(initial_total_energy)

    else:
        relative_energy_change = np.full_like(
            total_energy,
            np.nan
        )

        maximum_relative_energy_change = np.nan
        final_relative_energy_loss = np.nan

    return {
        "time": time_values,
        "states": state_values,
        "theta": theta_values,
        "omega": omega_values,
        "x": x_values,
        "y": y_values,
        "kinetic_energy": kinetic_energy,
        "potential_energy": potential_energy,
        "total_energy": total_energy,
        "initial_total_energy": initial_total_energy,
        "final_total_energy": final_total_energy,
        "absolute_energy_change": absolute_energy_change,
        "relative_energy_change": relative_energy_change,
        "maximum_absolute_energy_change": (
            maximum_absolute_energy_change
        ),
        "maximum_relative_energy_change": (
            maximum_relative_energy_change
        ),
        "final_relative_energy_loss": (
            final_relative_energy_loss
        ),
        "step_size": actual_step_size,
        "parameters": {
            "theta_initial": theta_initial,
            "omega_initial": omega_initial,
            "length": length,
            "mass": mass,
            "damping": damping,
            "total_time": total_time,
            "g": g
        }
    }
    
    
# ============================================================
# Section 6: User-facing simulation in degrees
# ============================================================

def simulate_single_pendulum_degrees(
    theta_degrees,
    omega_initial=0.0,
    length=1.0,
    mass=1.0,
    damping=0.08,
    total_time=13.0,
    g=9.81
):
    """
    Run a single-pendulum simulation using an initial angle
    supplied in degrees.

    This function is intended for the interactive website.
    The numerical step size is fixed internally at 0.01 s.
    """

    # Validate the public website parameter ranges.
    if not -180.0 <= theta_degrees <= 180.0:
        raise ValueError(
            "Initial angle must be between "
            "-180 and 180 degrees."
        )

    if not -2.0 <= omega_initial <= 2.0:
        raise ValueError(
            "Initial angular velocity must be between "
            "-2 and 2 rad/s."
        )

    if not 0.5 <= length <= 2.0:
        raise ValueError(
            "Pendulum length must be between "
            "0.5 and 2.0 metres."
        )

    if not 0.4 <= mass <= 1.6:
        raise ValueError(
            "Pendulum mass must be between "
            "0.4 and 1.6 kg."
        )

    if not 0.0 <= damping <= 0.6:
        raise ValueError(
            "Damping must be between 0.0 and 0.6."
        )

    if not 1.0 <= total_time <= 20.0:
        raise ValueError(
            "Total time must be between "
            "1 and 20 seconds."
        )

    theta_radians = np.deg2rad(
        theta_degrees
    )

    results = simulate_single_pendulum(
        theta_initial=theta_radians,
        omega_initial=omega_initial,
        length=length,
        mass=mass,
        damping=damping,
        total_time=total_time,
        step_size=0.01,
        g=g
    )

    results["theta_initial_degrees"] = (
        theta_degrees
    )

    return results