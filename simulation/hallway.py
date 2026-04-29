import numpy as np


class RectObstacle:
    """Axis-aligned rectangular obstacle."""

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.x_center = (x_min + x_max) / 2.0
        self.y_center = (y_min + y_max) / 2.0

    def ray_intersect(
        self, origin: np.ndarray, direction: np.ndarray, max_range: float = 30.0
    ) -> float:
        """AABB ray–box intersection. Returns distance to surface, capped at max_range."""
        px, py = float(origin[0]), float(origin[1])
        dx, dy = float(direction[0]), float(direction[1])
        INF = float("inf")

        if abs(dx) < 1e-9:
            t_enter_x = -INF if self.x_min <= px <= self.x_max else INF
            t_exit_x = INF if self.x_min <= px <= self.x_max else -INF
        else:
            t1, t2 = (self.x_min - px) / dx, (self.x_max - px) / dx
            t_enter_x, t_exit_x = min(t1, t2), max(t1, t2)

        if abs(dy) < 1e-9:
            t_enter_y = -INF if self.y_min <= py <= self.y_max else INF
            t_exit_y = INF if self.y_min <= py <= self.y_max else -INF
        else:
            t1, t2 = (self.y_min - py) / dy, (self.y_max - py) / dy
            t_enter_y, t_exit_y = min(t1, t2), max(t1, t2)

        t_enter = max(t_enter_x, t_enter_y)
        t_exit = min(t_exit_x, t_exit_y)

        if t_enter < t_exit and t_exit > 1e-6:
            return min(max(t_enter, 1e-6), max_range)
        return max_range

    def push_outside(self, pos: np.ndarray, margin: float = 0.1) -> np.ndarray:
        """Push pos to the nearest exterior face if it is inside the obstacle."""
        px, py = float(pos[0]), float(pos[1])
        if not (self.x_min < px < self.x_max and self.y_min < py < self.y_max):
            return pos
        d_left = px - self.x_min
        d_right = self.x_max - px
        d_bottom = py - self.y_min
        d_top = self.y_max - py
        min_d = min(d_left, d_right, d_bottom, d_top)
        if min_d == d_left:
            return np.array([self.x_min - margin, py])
        if min_d == d_right:
            return np.array([self.x_max + margin, py])
        if min_d == d_bottom:
            return np.array([px, self.y_min - margin])
        return np.array([px, self.y_max + margin])


class Hallway:
    """Infinite hallway: two parallel walls along the Y axis at x = ±width/2."""

    def __init__(
        self,
        width: float = 12.0,
        length: float = 100.0,
        obstacles: list | None = None,
    ):
        self.width = width
        self.length = length
        self.x_left = -width / 2.0
        self.x_right = width / 2.0
        self.obstacles: list[RectObstacle] = obstacles or []

    def ray_intersect(
        self, origin: np.ndarray, direction: np.ndarray, max_range: float = 30.0
    ) -> float:
        """Return distance along ray to the nearest wall or obstacle, capped at max_range."""
        px = float(origin[0])
        dx = float(direction[0])

        hit = max_range
        if abs(dx) >= 1e-9:
            t_left = (self.x_left - px) / dx
            if t_left > 1e-6:
                hit = min(hit, t_left)
            t_right = (self.x_right - px) / dx
            if t_right > 1e-6:
                hit = min(hit, t_right)

        for obs in self.obstacles:
            hit = min(hit, obs.ray_intersect(origin, direction, max_range))

        return hit

    def clamp_position(self, pos: np.ndarray, margin: float = 0.05) -> np.ndarray:
        x = float(np.clip(pos[0], self.x_left + margin, self.x_right - margin))
        p = np.array([x, pos[1]])
        for obs in self.obstacles:
            p = obs.push_outside(p, margin)
        return p

    def is_inside(self, pos: np.ndarray) -> bool:
        if not (self.x_left < pos[0] < self.x_right):
            return False
        for obs in self.obstacles:
            if obs.x_min < pos[0] < obs.x_max and obs.y_min < pos[1] < obs.y_max:
                return False
        return True
