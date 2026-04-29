"""
Random Weighting fusion.

Each ray that hits a wall produces an x-position estimate.
Estimates are combined with randomly drawn normalised weights.
"""
import numpy as np
from drone.sensors import x_estimates_from_rays


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

    w = np.random.uniform(0.0, 1.0, len(x_ests))
    w /= w.sum()

    new_state = state.copy()
    new_state[0] = float(w @ x_ests)
    new_state[1] += state[3] * dt  # dead-reckon y from estimated vy
    return new_state, cov.copy()
