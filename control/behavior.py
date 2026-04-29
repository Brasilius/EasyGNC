"""
Behavior-based control.

Each drone independently blends three behaviours with fixed weights:
  1. Centering  – weak attraction toward x = 0
  2. Wall avoid – repulsion when within WALL_THRESH of a wall
  3. Separation – repulsion from neighbours within SEP_THRESH
"""
import numpy as np

W_CENTER = 0.3
W_WALL = 3.5
W_SEP = 1.5
WALL_THRESH = 2.0
SEP_THRESH = 2.5
BASE_VY = 2.0


def compute_velocities(drones: list, env) -> list:
    velocities = []
    for drone in drones:
        px = drone.est_pos[0]
        vx = W_CENTER * (-px)

        d_left = px - env.x_left
        d_right = env.x_right - px
        if d_left < WALL_THRESH:
            vx += W_WALL * (WALL_THRESH - d_left) / WALL_THRESH
        if d_right < WALL_THRESH:
            vx -= W_WALL * (WALL_THRESH - d_right) / WALL_THRESH

        for other in drones:
            if other.id == drone.id:
                continue
            dx = px - other.est_pos[0]
            dist = abs(dx)
            if 1e-6 < dist < SEP_THRESH:
                vx += W_SEP * (SEP_THRESH - dist) / SEP_THRESH * np.sign(dx)

        velocities.append(np.array([vx, BASE_VY]))
    return velocities
