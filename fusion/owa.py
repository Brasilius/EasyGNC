"""
Ordered Weighted Average (OWA) fusion.

X estimates from each ray are sorted, then combined with Gaussian-shaped weights
centred on the median (trimming extreme outliers).
"""
import numpy as np
from drone.sensors import x_estimates_from_rays


def _gaussian_weights(n: int) -> np.ndarray:
    idx = np.arange(n, dtype=float)
    center = (n - 1) / 2.0
    w = np.exp(-0.5 * ((idx - center) / (n / 4.0)) ** 2)
    return w / w.sum()


def step(
    state: np.ndarray,
    cov: np.ndarray,
    distances: np.ndarray,
    env,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_ests = x_estimates_from_rays(distances, env)
    if len(x_ests) == 0:
        return state.copy(), cov.copy()

    sorted_ests = np.sort(x_ests)
    w = _gaussian_weights(len(sorted_ests))

    new_state = state.copy()
    new_state[0] = float(w @ sorted_ests)
    new_state[1] += state[3] * dt  # dead-reckon y from estimated vy
    return new_state, cov.copy()
