import numpy as np

from drone.drone import Drone
from drone.sensors import cast_rays
from simulation.hallway import Hallway

INITIAL_X = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
INITIAL_Y = 0.0
BASE_VY = 2.0
DT = 0.1
PROCESS_NOISE_STD = 0.03

_K_OBS = 8.0  # repulsion gain
_OBS_RADIUS = 6.0  # influence radius from obstacle surface (m)
_OBS_Y_AHEAD = 10.0  # vy decel look-ahead distance (m)


def create_swarm(hallway: Hallway) -> list[Drone]:
    return [Drone(i, float(x), INITIAL_Y, BASE_VY) for i, x in enumerate(INITIAL_X)]


def _obstacle_avoidance(
    vel: np.ndarray, pos: np.ndarray, hallway: Hallway
) -> np.ndarray:
    """Add repulsion from obstacles into the commanded velocity."""
    vx, vy = float(vel[0]), float(vel[1])
    for obs in hallway.obstacles:
        cx = float(np.clip(pos[0], obs.x_min, obs.x_max))
        cy = float(np.clip(pos[1], obs.y_min, obs.y_max))
        diff = pos - np.array([cx, cy])
        dist = float(np.linalg.norm(diff))
        if 1e-6 < dist < _OBS_RADIUS:
            strength = _K_OBS * (1.0 / dist - 1.0 / _OBS_RADIUS)
            lat = diff[0]
            # When the drone is within the obstacle's x-span, diff[0] == 0 so
            # normal repulsion has no lateral component.  Steer using the
            # drone's offset from the obstacle centre; if exactly centred,
            # choose the side with the wider open gap.
            if abs(lat) < 1e-6:
                offset = pos[0] - obs.x_center
                if abs(offset) > 1e-6:
                    lat = np.sign(offset) * dist
                else:
                    gap_right = hallway.x_right - obs.x_max
                    gap_left = obs.x_min - hallway.x_left
                    lat = dist if gap_right >= gap_left else -dist
            vx += strength * lat / dist

        # Decelerate vy if directly in front of obstacle and still approaching
        in_x_range = obs.x_min - 1.0 < pos[0] < obs.x_max + 1.0
        dy_ahead = obs.y_min - pos[1]
        if in_x_range and 0.0 < dy_ahead < _OBS_Y_AHEAD:
            vy *= dy_ahead / _OBS_Y_AHEAD

    return np.array([vx, vy])


def step(drones: list[Drone], hallway: Hallway, fusion_step, control_fn) -> None:
    """One simulation tick: sense → fuse → control → obstacle-avoid → move."""
    for drone in drones:
        distances = cast_rays(drone.true_pos, hallway)
        drone.state_est, drone.cov = fusion_step(
            drone.state_est, drone.cov, distances, hallway, DT
        )
        # y is not observable from lateral wall sensors; sync from true position
        drone.state_est[1] = drone.true_pos[1]

    velocities = control_fn(drones, hallway)
    for drone, vel in zip(drones, velocities):
        vel = _obstacle_avoidance(vel, drone.true_pos, hallway)
        noise = np.random.normal(0.0, PROCESS_NOISE_STD, 2)
        drone.apply_velocity(vel + noise, DT, hallway)
        drone.state_est[3] = vel[1]
        drone.record()
