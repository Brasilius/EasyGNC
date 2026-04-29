"""
Extended Kalman Filter for lateral position estimation.

State vector: [x, y, vx, vy]
Process model: constant-velocity (x += vx*dt, y += vy*dt).
Measurement model: wall distances from 8 optical rays.
Since distance is linear in x, the EKF Jacobian is exact (no linearisation error).
"""
import numpy as np
from drone.sensors import RAY_ANGLES

_Q = np.diag([0.01, 0.01, 0.1, 0.1])   # process noise covariance
_R_VAR = SENSOR_NOISE_STD_SQ = 0.08 ** 2  # measurement noise variance per ray


def _F(dt: float) -> np.ndarray:
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    return F


def _measurement_model(state: np.ndarray, env):
    """Return (z_hat, H) for rays that hit a wall (|cos θ| ≥ 0.1)."""
    x = state[0]
    z_hat_list, H_rows = [], []
    for angle in RAY_ANGLES:
        cos_a = np.cos(angle)
        if abs(cos_a) < 0.1:
            continue
        row = np.zeros(4)
        if cos_a < 0.0:
            z_hat = (x - env.x_left) / (-cos_a)
            row[0] = 1.0 / (-cos_a)
        else:
            z_hat = (env.x_right - x) / cos_a
            row[0] = -1.0 / cos_a
        z_hat_list.append(max(0.01, z_hat))
        H_rows.append(row)
    return np.array(z_hat_list), np.array(H_rows)


def step(
    state: np.ndarray,
    cov: np.ndarray,
    distances: np.ndarray,
    env,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """EKF predict + update. Returns (new_state, new_cov)."""
    # --- Predict ---
    F = _F(dt)
    sp = F @ state
    Pp = F @ cov @ F.T + _Q

    # --- Update ---
    z_meas = np.array(
        [distances[i] for i, a in enumerate(RAY_ANGLES) if abs(np.cos(a)) >= 0.1]
    )
    z_hat, H = _measurement_model(sp, env)

    n = len(z_hat)
    R = np.eye(n) * _R_VAR
    S = H @ Pp @ H.T + R
    K = Pp @ H.T @ np.linalg.inv(S)

    state_upd = sp + K @ (z_meas - z_hat)
    cov_upd = (np.eye(4) - K @ H) @ Pp

    return state_upd, cov_upd
