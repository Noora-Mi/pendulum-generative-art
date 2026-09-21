# Pendulum Generative Art

## Live Demo

Try the interactive application here:

https://pendulum-generative-art.streamlit.app

An interactive generative-art project that transforms the motion of nonlinear pendulum systems into visual trajectories.

The project combines mathematical modeling, numerical simulation, and visual design. Users can modify physical parameters and observe how the motion and resulting artwork change.

## Features

- Interactive Streamlit interface
- Single-pendulum simulation
- Double-pendulum simulation
- Adjustable initial angles and angular velocities
- Adjustable pendulum lengths and masses
- Optional damping for the single pendulum
- Multiple color-map options
- Static trajectory visualization
- Animated GIF generation
- Phase-space visualization
- Energy-based numerical validation
- Downloadable artwork and animations

## Mathematical Models

### Single Pendulum

The nonlinear single-pendulum equation is

\[
\ddot{\theta}
+
\gamma\dot{\theta}
+
\frac{g}{L}\sin(\theta)
=
0,
\]

where:

- \(\theta\) is the angular displacement;
- \(\dot{\theta}\) is the angular velocity;
- \(L\) is the pendulum length;
- \(g\) is gravitational acceleration;
- \(\gamma\) is the damping coefficient.

The bob position is calculated using

\[
x=L\sin(\theta),
\qquad
y=-L\cos(\theta).
\]

### Double Pendulum

The double pendulum is modeled as two connected pendulums with angles
\(\theta_1\) and \(\theta_2\).

Their Cartesian coordinates are

\[
x_1=L_1\sin(\theta_1),
\qquad
y_1=-L_1\cos(\theta_1),
\]

\[
x_2=x_1+L_2\sin(\theta_2),
\qquad
y_2=y_1-L_2\cos(\theta_2).
\]

The nonlinear equations of motion are solved numerically using the
fourth-order Runge–Kutta method.

## Visual Mapping

The simulations convert physical data into visual properties:

| Simulation quantity | Visual property |
|---|---|
| Time | Trajectory color |
| Linear speed | Physical-trajectory thickness |
| Angular speed | Phase-space trajectory thickness |
| Pendulum position | Moving bob position |
| Angle and angular velocity | Phase-space coordinates |

## Public Parameter Ranges

| Parameter | Available range |
|---|---:|
| Initial angle | −180° to 180° |
| Initial angular velocity | −2 to 2 rad/s |
| Mass | 0.4 to 1.6 kg |
| Double-pendulum length | 2/3 to 4/3 m |
| Single-pendulum length | 0.5 to 2.0 m |
| Single-pendulum damping | 0 to 0.6 |
| Simulation time | 1 to 20 s |

## Numerical Method

The simulations use the fourth-order Runge–Kutta method.

Default internal step sizes:

- Single pendulum: \(h=0.01\) s
- Double pendulum: \(h=0.005\) s

Energy conservation tests are included to evaluate numerical accuracy
for undamped simulations.

## Project Structure

```text
generative-art-model/
├── app.py
├── single_pendulum_model.py
├── single_pendulum_visualization.py
├── double_pendulum_model.py
├── double_pendulum_visualization.py
├── test_single_pendulum_model.py
├── test_single_pendulum_animation.py
├── test_double_pendulum_model.py
├── test_visualization.py
├── test_animation.py
├── requirements.txt
└── README.md