import numpy as np


class Drone:
    """Single drone: true kinematic state + fusion-maintained state estimate."""

    def __init__(self, drone_id: int, x: float, y: float = 0.0, vy: float = 2.0):
        self.id = drone_id
        self.true_pos = np.array([x, y], dtype=float)
        self.true_vel = np.array([0.0, vy], dtype=float)

        # State estimate: [x, y, vx, vy]
        self.state_est = np.array([x, y, 0.0, vy], dtype=float)
        self.cov = np.eye(4) * 0.5

        self.pos_history: list = [self.true_pos.copy()]
        self.est_history: list = [self.state_est[:2].copy()]

    @property
    def est_pos(self) -> np.ndarray:
        return self.state_est[:2]

    @property
    def est_vel(self) -> np.ndarray:
        return self.state_est[2:]

    def apply_velocity(self, vel: np.ndarray, dt: float, env) -> None:
        self.true_vel = vel.copy()
        self.true_pos = env.clamp_position(self.true_pos + vel * dt)

    def record(self) -> None:
        self.pos_history.append(self.true_pos.copy())
        self.est_history.append(self.state_est[:2].copy())
