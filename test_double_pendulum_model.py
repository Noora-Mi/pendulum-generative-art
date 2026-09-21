# ============================================================
# Section 1: Import the reusable double-pendulum model
# ============================================================

import numpy as np

from double_pendulum_model import (
    simulate_double_pendulum,
    simulate_double_pendulum_degrees
)


# ============================================================
# Section 2: Define one baseline test
# ============================================================

theta1_initial = 2.0
theta2_initial = -1.2

omega1_initial = 0.0
omega2_initial = 0.0

m1 = 1.0
m2 = 1.0

L1 = 1.0
L2 = 1.0

total_time = 20.0
step_size = 0.005


# ============================================================
# Section 3: Run the simulation
# ============================================================

results = simulate_double_pendulum(
    theta1_initial=theta1_initial,
    theta2_initial=theta2_initial,
    omega1_initial=omega1_initial,
    omega2_initial=omega2_initial,
    m1=m1,
    m2=m2,
    L1=L1,
    L2=L2,
    total_time=total_time,
    step_size=step_size
)


# ============================================================
# Section 4: Display the results
# ============================================================

print("Reusable double-pendulum model test")
print("-----------------------------------")

print("Number of time points:")
print(len(results["time"]))

print("\nFinal simulation time:")
print(results["time"][-1])

print("\nInitial state:")
print(results["states"][0])

print("\nFinal state:")
print(results["states"][-1])

print("\nInitial position of bob 1:")
print(
    results["x1"][0],
    results["y1"][0]
)

print("\nInitial position of bob 2:")
print(
    results["x2"][0],
    results["y2"][0]
)

print("\nActual numerical step size:")
print(results["step_size"])


# ============================================================
# Section 5: Basic automatic checks
# ============================================================

assert len(results["time"]) == 4001

assert np.isclose(
    results["time"][-1],
    total_time
)

assert results["states"].shape == (4001, 4)

assert np.all(
    np.isfinite(results["states"])
)

# ============================================================
# Section 6: Energy validation
# ============================================================

print("\nEnergy validation:")
print(
    "Initial total energy:",
    results["initial_total_energy"],
    "J"
)

print(
    "Maximum absolute energy error:",
    results["maximum_absolute_energy_error"],
    "J"
)

print(
    "Maximum relative energy error:",
    results["maximum_relative_energy_error"]
)


# The initial energy should match the previously tested result.
assert np.isclose(
    results["initial_total_energy"],
    34.040071361638766,
    rtol=1e-10,
    atol=1e-10
)

# The energy arrays should have one value at every time point.
assert len(results["kinetic_energy"]) == 4001
assert len(results["potential_energy"]) == 4001
assert len(results["total_energy"]) == 4001

# All calculated energy values should be finite.
assert np.all(
    np.isfinite(results["total_energy"])
)

# For this baseline test, the relative energy error should
# remain below 0.01%.
assert (
    results["maximum_relative_energy_error"]
    < 1e-4
)

print("\nAll basic and energy checks passed.")


# ============================================================
# Section 7: Test all angle-sign combinations
# ============================================================

angle_sign_tests = [
    {
        "name": "positive_positive",
        "theta1": 1.0,
        "theta2": 1.5
    },
    {
        "name": "positive_negative",
        "theta1": 1.0,
        "theta2": -1.5
    },
    {
        "name": "negative_positive",
        "theta1": -1.0,
        "theta2": 1.5
    },
    {
        "name": "negative_negative",
        "theta1": -1.0,
        "theta2": -1.5
    }
]

print("\nAngle-sign combination tests:")
print("-----------------------------------")

for test_case in angle_sign_tests:

    test_results = simulate_double_pendulum(
        theta1_initial=test_case["theta1"],
        theta2_initial=test_case["theta2"],
        omega1_initial=0.0,
        omega2_initial=0.0,
        m1=1.0,
        m2=1.0,
        L1=1.0,
        L2=1.0,
        total_time=5.0,
        step_size=0.005
    )

    assert np.all(
        np.isfinite(test_results["states"])
    )

    assert (
        test_results["maximum_relative_energy_error"]
        < 1e-4
    )

    print(
        test_case["name"],
        "| theta1 =",
        test_case["theta1"],
        "| theta2 =",
        test_case["theta2"],
        "| maximum relative energy error =",
        test_results["maximum_relative_energy_error"]
    )

print("\nAll angle-sign combination tests passed.")



# ============================================================
# Section 8: Combined boundary and step-size test
# ============================================================

combined_test_step_sizes = [
    0.01,
    0.005,
    0.0025,
    0.001
]

print("\nCombined boundary and step-size test:")
print("-----------------------------------")

for tested_step_size in combined_test_step_sizes:

    combined_results = simulate_double_pendulum(
        theta1_initial=3.0,
        theta2_initial=-3.0,
        omega1_initial=2.0,
        omega2_initial=-2.0,
        m1=1.6,
        m2=0.4,
        L1=4.0 / 3.0,
        L2=2.0 / 3.0,
        total_time=10.0,
        step_size=tested_step_size
    )

    print(
        "step size =",
        tested_step_size,
        "| time points =",
        len(combined_results["time"]),
        "| initial energy =",
        combined_results["initial_total_energy"],
        "| maximum relative energy error =",
        combined_results["maximum_relative_energy_error"]
    )

    assert np.all(
        np.isfinite(combined_results["states"])
    )

print("\nCombined boundary simulations completed.")


# ============================================================
# Section 9: Degree-input interface test
# ============================================================

degree_results = simulate_double_pendulum_degrees(
    theta1_degrees=90.0,
    theta2_degrees=-90.0,
    omega1_initial=0.0,
    omega2_initial=0.0,
    m1=1.0,
    m2=1.0,
    L1=1.0,
    L2=1.0,
    total_time=1.0
)

expected_theta1_radians = np.pi / 2
expected_theta2_radians = -np.pi / 2

print("\nDegree-input interface test:")
print("-----------------------------------")

print(
    "theta1:",
    degree_results["theta1_initial_degrees"],
    "degrees =",
    degree_results["theta1"][0],
    "radians"
)

print(
    "theta2:",
    degree_results["theta2_initial_degrees"],
    "degrees =",
    degree_results["theta2"][0],
    "radians"
)

print(
    "Automatically selected step size:",
    degree_results["step_size"]
)


# Check that 90 degrees was converted to pi / 2.
assert np.isclose(
    degree_results["theta1"][0],
    expected_theta1_radians
)

# Check that -90 degrees was converted to -pi / 2.
assert np.isclose(
    degree_results["theta2"][0],
    expected_theta2_radians
)

# Check that the hidden website step size is 0.005 seconds.
assert np.isclose(
    degree_results["step_size"],
    0.005
)

# Check that the simulation produced finite results.
assert np.all(
    np.isfinite(degree_results["states"])
)

print("\nDegree-input interface test passed.")


# ============================================================
# Section 10: Public parameter-range validation
# ============================================================

print("\nPublic parameter-range validation:")
print("-----------------------------------")


# Test that the allowed boundary values can run successfully.
boundary_results = simulate_double_pendulum_degrees(
    theta1_degrees=180.0,
    theta2_degrees=-180.0,
    omega1_initial=2.0,
    omega2_initial=-2.0,
    m1=1.6,
    m2=0.4,
    L1=4.0 / 3.0,
    L2=2.0 / 3.0,
    total_time=1.0
)

assert np.all(
    np.isfinite(boundary_results["states"])
)

print("Allowed boundary values: passed")


# Each dictionary contains one deliberately invalid parameter.
invalid_parameter_tests = [
    {
        "name": "theta1 above maximum",
        "changes": {
            "theta1_degrees": 181.0
        }
    },
    {
        "name": "theta2 below minimum",
        "changes": {
            "theta2_degrees": -181.0
        }
    },
    {
        "name": "omega1 above maximum",
        "changes": {
            "omega1_initial": 2.1
        }
    },
    {
        "name": "omega2 below minimum",
        "changes": {
            "omega2_initial": -2.1
        }
    },
    {
        "name": "mass below minimum",
        "changes": {
            "m1": 0.3
        }
    },
    {
        "name": "length below minimum",
        "changes": {
            "L2": 0.5
        }
    },
    {
        "name": "simulation time above maximum",
        "changes": {
            "total_time": 21.0
        }
    }
]


# These are valid default website parameters.
base_website_parameters = {
    "theta1_degrees": 90.0,
    "theta2_degrees": -90.0,
    "omega1_initial": 0.0,
    "omega2_initial": 0.0,
    "m1": 1.0,
    "m2": 1.0,
    "L1": 1.0,
    "L2": 1.0,
    "total_time": 5.0
}


for test_case in invalid_parameter_tests:

    test_parameters = (
        base_website_parameters.copy()
    )

    test_parameters.update(
        test_case["changes"]
    )

    try:
        simulate_double_pendulum_degrees(
            **test_parameters
        )

    except ValueError as error:
        print(
            test_case["name"],
            ": correctly rejected ->",
            error
        )

    else:
        raise AssertionError(
            test_case["name"]
            + " was not rejected."
        )


print("\nAll public parameter-range tests passed.")