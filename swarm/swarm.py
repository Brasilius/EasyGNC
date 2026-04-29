import numpy as np
from drone.drone import Drone
from drone.sensors import cast_rays
from simulation.hallway import Hallway

INITIAL_X = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
INITIAL_Y = 0.0
BASE_VY = 2.0
DT = 0.1
PROCESS_NOISE_STD = 0.03


def create_swarm(hallway: Hallway) -> list[Drone]:
    return [Drone(i, float(x), INITIAL_Y, BASE_VY) for i, x in enumerate(INITIAL_X)]


def step(drones: list[Drone], hallway: Hallway, fusion_step, control_fn) -> None:
    """One simulation tick: sense → fuse → control → move."""
    # Sense and fuse for each drone
    for drone in drones:
        distances = cast_rays(drone.true_pos, hallway)
        drone.state_est, drone.cov = fusion_step(
            drone.state_est, drone.cov, distances, hallway, DT
        )
        # y is not observable from lateral wall sensors; sync from true position
        drone.state_est[1] = drone.true_pos[1]

    # Control then move
    velocities = control_fn(drones, hallway)
    for drone, vel in zip(drones, velocities):
        noise = np.random.normal(0.0, PROCESS_NOISE_STD, 2)
        drone.apply_velocity(vel + noise, DT, hallway)
        drone.state_est[3] = vel[1]  # keep vy in state aligned with command
        drone.record()
