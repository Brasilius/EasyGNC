import numpy as np

NUM_RAYS = 8
RAY_ANGLES = np.linspace(0.0, 2.0 * np.pi, NUM_RAYS, endpoint=False)
SENSOR_NOISE_STD = 0.08
MAX_RANGE = 30.0


def cast_rays(pos: np.ndarray, env, noise_std: float = SENSOR_NOISE_STD) -> np.ndarray:
    """Cast 8 rays fanning 360° from pos. Returns noisy distances array of shape (8,)."""
    distances = np.empty(NUM_RAYS)
    for i, angle in enumerate(RAY_ANGLES):
        direction = np.array([np.cos(angle), np.sin(angle)])
        d = env.ray_intersect(pos, direction)
        distances[i] = max(0.01, d + np.random.normal(0.0, noise_std))
    return distances


def x_estimates_from_rays(distances: np.ndarray, env) -> np.ndarray:
    """
    Convert ray distances to x-position estimates.
    Each ray that hits a wall (|cos θ| ≥ 0.1) yields one x estimate.
    Rays parallel to the walls are skipped.
    """
    estimates = []
    for d, angle in zip(distances, RAY_ANGLES):
        cos_a = np.cos(angle)
        if abs(cos_a) < 0.1:
            continue
        if cos_a < 0.0:
            # Ray going left  → hits left wall → x = x_left + d * |cos_a|
            estimates.append(env.x_left + d * (-cos_a))
        else:
            # Ray going right → hits right wall → x = x_right - d * cos_a
            estimates.append(env.x_right - d * cos_a)
    return np.array(estimates)
