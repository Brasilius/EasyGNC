# EasyGNC

A 2-D drone swarm simulation for comparing guidance, navigation, and control (GNC) strategies. Five drones start at staggered X positions and fly along a hallway, with the system testing every combination of three fusion methods and three control methods.

---

## What it does

- **Environment** – a straight hallway (configurable width × length). Two parallel walls extend along the Y axis. Drones enter at Y = 0 and fly toward increasing Y.
- **Sensors** – each drone casts 8 optical rays at 45° intervals. Each ray returns the distance to the nearest wall (with Gaussian noise). No GPS; lateral position is estimated entirely from these distance vectors.
- **Fusion methods** (how sensor readings are combined into a position estimate):
  | Method | Description |
  |---|---|
  | EKF | Extended Kalman Filter – maintains a full probabilistic state [x, y, vx, vy] with a constant-velocity process model and wall-distance measurement model |
  | Random Weighting | Converts each ray to an x estimate, then takes a randomly-weighted average |
  | OWA | Ordered Weighted Average – sorts the x estimates and applies Gaussian-shaped weights that favour the median, trimming outliers |
- **Control methods** (how velocity commands are generated from state estimates):
  | Method | Description |
  |---|---|
  | Leader-Follower | Centre drone tracks the hallway centreline; others hold fixed X offsets relative to the leader |
  | Consensus | Drones rank themselves by X position and drive toward evenly-spaced desired positions, with wall-avoidance override |
  | Behavior-Based | Each drone independently blends centering, wall-avoidance, and inter-drone separation behaviours |
- **Comparison** – all 9 combinations run at the same random seed and are evaluated on three metrics: position RMSE, minimum wall clearance, and average formation spread.

---

## Project layout

```
EasyGNC/
├── simulation/
│   └── hallway.py          # Hallway geometry and ray intersection
├── drone/
│   ├── drone.py            # Drone state (true + estimated)
│   └── sensors.py          # Ray casting and x-estimate extraction
├── fusion/
│   ├── ekf.py              # Extended Kalman Filter
│   ├── random_weighting.py # Random Weighting
│   └── owa.py              # Ordered Weighted Average
├── control/
│   ├── leader_follower.py  # Leader-Follower
│   ├── consensus.py        # Consensus-based
│   └── behavior.py         # Behavior-based
├── swarm/
│   ├── swarm.py            # Single simulation step (sense → fuse → control → move)
│   └── runner.py           # Runs all 9 combinations and collects metrics
├── visualization/
│   ├── animator.py         # 3×3 trajectory plot
│   └── metrics.py          # 3-panel heatmap of performance metrics
├── main.py                 # Entry point
└── pyproject.toml
```

---

## Setup

### Requirements
- Python 3.10 or later
- pip

### Install

```bash
# 1. Clone / enter the project directory
cd EasyGNC

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e .
```

Dependencies (`numpy`, `matplotlib`) are installed automatically from `pyproject.toml`.

### Run

```bash
python main.py
```

This runs all 9 fusion × control combinations (≈ 5 seconds) and opens two matplotlib windows:

1. **Trajectory plot** – 3 × 3 grid showing each drone's X–Y path through the hallway. Circles mark starting positions, squares mark final positions.
2. **Metrics heatmap** – colour-coded table comparing Position RMSE, minimum wall clearance, and average formation spread across all combinations.

A text summary is also printed to the terminal.

---

## Configuration

Key parameters are defined as module-level constants and can be edited directly:

| File | Constant | Default | Meaning |
|---|---|---|---|
| `main.py` | `Hallway(width=..., length=...)` | 12 m × 100 m | Hallway dimensions |
| `swarm/swarm.py` | `DT` | 0.1 s | Simulation timestep |
| `swarm/swarm.py` | `INITIAL_X` | [-4, -2, 0, 2, 4] | Starting X positions |
| `swarm/swarm.py` | `PROCESS_NOISE_STD` | 0.03 m | Actuator noise |
| `swarm/runner.py` | `NUM_STEPS` | 500 | Steps per run (50 s) |
| `drone/sensors.py` | `SENSOR_NOISE_STD` | 0.08 m | Optical sensor noise |
| `drone/sensors.py` | `NUM_RAYS` | 8 | Rays per drone |

---

## Metrics

| Metric | Ideal | Notes |
|---|---|---|
| Position RMSE | low | Lateral estimation error averaged over all drones and all timesteps |
| Min wall clearance | high | Closest any drone got to either wall during the run |
| Avg formation spread | context-dependent | Standard deviation of drone X positions; low = tight formation |

---

## Roadmap

- Stage 2: obstacles inside the hallway (pillars, doorways)
- Stage 3: full 3-D simulation with altitude control
- Stage 4: Monte Carlo runner across N seeds with confidence intervals
- Stage 5: Gazebo / ROS integration for hardware-in-the-loop testing
