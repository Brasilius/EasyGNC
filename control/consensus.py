"""
Consensus-based control.

Drones rank themselves by estimated x position and drive toward evenly-spaced
desired positions. Wall-avoidance terms override consensus when close to a wall.
"""
import numpy as np

K_CONSENSUS = 0.8
K_WALL = 3.0
WALL_MARGIN = 1.5
DESIRED_SPACING = 2.0
BASE_VY = 2.0


def compute_velocities(drones: list, env) -> list:
    n = len(drones)
    xs = [d.est_pos[0] for d in drones]
    sorted_idx = np.argsort(xs)

    velocities = []
    for i, drone in enumerate(drones):
        rank = int(np.where(sorted_idx == i)[0][0])
        desired_x = (rank - (n - 1) / 2.0) * DESIRED_SPACING
        vx = K_CONSENSUS * (desired_x - drone.est_pos[0])

        d_left = drone.est_pos[0] - env.x_left
        d_right = env.x_right - drone.est_pos[0]
        if d_left < WALL_MARGIN:
            vx += K_WALL * (WALL_MARGIN - d_left)
        if d_right < WALL_MARGIN:
            vx -= K_WALL * (WALL_MARGIN - d_right)

        velocities.append(np.array([vx, BASE_VY]))
    return velocities
