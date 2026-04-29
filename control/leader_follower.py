"""
Leader-Follower control.

Drone 2 (centre) is the designated leader and tracks the hallway centreline.
The four followers maintain fixed x-offsets relative to the leader's estimated position.
"""
import numpy as np

LEADER_IDX = 2
# Desired x position of each drone relative to the leader
_OFFSETS = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
KP = 1.5
BASE_VY = 2.0


def compute_velocities(drones: list, env) -> list:
    leader = drones[LEADER_IDX]
    ldr_vx = -KP * leader.est_pos[0]  # proportional pull toward centreline

    velocities = []
    for i, drone in enumerate(drones):
        if i == LEADER_IDX:
            velocities.append(np.array([ldr_vx, BASE_VY]))
        else:
            desired_x = leader.est_pos[0] + _OFFSETS[i]
            vx = ldr_vx + KP * (desired_x - drone.est_pos[0])
            velocities.append(np.array([vx, BASE_VY]))
    return velocities
