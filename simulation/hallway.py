import numpy as np


class Hallway:
    """Infinite hallway: two parallel walls along the Y axis at x = ±width/2."""

    def __init__(self, width: float = 12.0, length: float = 100.0):
        self.width = width
        self.length = length
        self.x_left = -width / 2.0
        self.x_right = width / 2.0

    def ray_intersect(
        self, origin: np.ndarray, direction: np.ndarray, max_range: float = 30.0
    ) -> float:
        """Return distance along ray to the nearest wall, capped at max_range."""
        px = origin[0]
        dx = direction[0]

        if abs(dx) < 1e-9:
            return max_range

        t_candidates = []
        t_left = (self.x_left - px) / dx
        if t_left > 1e-6:
            t_candidates.append(t_left)
        t_right = (self.x_right - px) / dx
        if t_right > 1e-6:
            t_candidates.append(t_right)

        return min(min(t_candidates), max_range) if t_candidates else max_range

    def clamp_position(self, pos: np.ndarray, margin: float = 0.05) -> np.ndarray:
        x = float(np.clip(pos[0], self.x_left + margin, self.x_right - margin))
        return np.array([x, pos[1]])

    def is_inside(self, pos: np.ndarray) -> bool:
        return self.x_left < pos[0] < self.x_right
