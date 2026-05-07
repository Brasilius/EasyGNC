"""
Tests for simulation/hallway.py — RectObstacle and Hallway.
"""
import numpy as np
import pytest
from simulation.hallway import RectObstacle, Hallway


# ─── RectObstacle ────────────────────────────────────────────────────────────

class TestRectObstacle:
    def setup_method(self):
        # 2×2 square centred at (5, 5)
        self.obs = RectObstacle(x_min=4.0, x_max=6.0, y_min=4.0, y_max=6.0)

    # --- ray_intersect ---

    def test_ray_hits_front_face(self):
        origin = np.array([5.0, 0.0])
        direction = np.array([0.0, 1.0])
        d = self.obs.ray_intersect(origin, direction)
        assert abs(d - 4.0) < 1e-6, "ray should hit y_min=4 at distance 4"

    def test_ray_misses_obstacle(self):
        origin = np.array([0.0, 5.0])
        direction = np.array([0.0, 1.0])  # parallel, x=0 outside box
        d = self.obs.ray_intersect(origin, direction, max_range=30.0)
        assert d == 30.0

    def test_ray_going_away(self):
        origin = np.array([5.0, 10.0])
        direction = np.array([0.0, 1.0])  # moving away
        d = self.obs.ray_intersect(origin, direction, max_range=30.0)
        assert d == 30.0

    def test_ray_from_inside(self):
        # origin inside obstacle; nearest forward exit face should be returned
        origin = np.array([5.0, 5.0])
        direction = np.array([1.0, 0.0])
        d = self.obs.ray_intersect(origin, direction, max_range=30.0)
        # t_exit_x for right face = (6-5)/1 = 1.0, t_enter_x = (4-5)/1 = -1.0 → t_enter=-1, t_exit=1 → return max(-1,1e-6) = 1e-6
        assert d <= 1.0 + 1e-6

    def test_ray_parallel_to_face_no_hit(self):
        origin = np.array([3.0, 3.0])
        direction = np.array([0.0, 1.0])  # x=3 never enters [4,6]
        d = self.obs.ray_intersect(origin, direction, max_range=30.0)
        assert d == 30.0

    def test_ray_capped_at_max_range(self):
        origin = np.array([5.0, 0.0])
        direction = np.array([0.0, 1.0])
        d = self.obs.ray_intersect(origin, direction, max_range=2.0)
        assert d == 2.0

    def test_ray_exact_corner(self):
        # Ray aimed exactly at corner (4,4) from origin (0,0)
        origin = np.array([0.0, 0.0])
        direction = np.array([1.0, 1.0]) / np.sqrt(2)
        d = self.obs.ray_intersect(origin, direction, max_range=30.0)
        # Distance to corner = sqrt(32) ≈ 5.656
        assert 5.0 < d < 30.0

    # --- push_outside ---

    def test_push_outside_already_outside(self):
        pos = np.array([0.0, 5.0])
        out = self.obs.push_outside(pos)
        np.testing.assert_array_equal(out, pos)

    def test_push_outside_inside_to_left(self):
        # Very close to left face
        pos = np.array([4.1, 5.0])
        out = self.obs.push_outside(pos, margin=0.1)
        assert out[0] < self.obs.x_min

    def test_push_outside_inside_to_right(self):
        pos = np.array([5.9, 5.0])
        out = self.obs.push_outside(pos, margin=0.1)
        assert out[0] > self.obs.x_max

    def test_push_outside_inside_to_bottom(self):
        pos = np.array([5.0, 4.1])
        out = self.obs.push_outside(pos, margin=0.1)
        assert out[1] < self.obs.y_min

    def test_push_outside_inside_to_top(self):
        pos = np.array([5.0, 5.9])
        out = self.obs.push_outside(pos, margin=0.1)
        assert out[1] > self.obs.y_max

    def test_push_outside_on_boundary(self):
        # Exactly on the boundary — interior check is strict (<), so this returns unchanged
        pos = np.array([4.0, 5.0])
        out = self.obs.push_outside(pos)
        np.testing.assert_array_equal(out, pos)

    def test_center_attributes(self):
        assert self.obs.x_center == 5.0
        assert self.obs.y_center == 5.0


# ─── Hallway ─────────────────────────────────────────────────────────────────

class TestHallway:
    def setup_method(self):
        self.hw = Hallway(width=12.0, length=100.0)

    # --- geometry ---

    def test_wall_positions(self):
        assert self.hw.x_left == -6.0
        assert self.hw.x_right == 6.0

    # --- ray_intersect (walls only) ---

    def test_ray_center_rightward(self):
        origin = np.array([0.0, 50.0])
        direction = np.array([1.0, 0.0])
        d = self.hw.ray_intersect(origin, direction)
        assert abs(d - 6.0) < 1e-9

    def test_ray_center_leftward(self):
        origin = np.array([0.0, 50.0])
        direction = np.array([-1.0, 0.0])
        d = self.hw.ray_intersect(origin, direction)
        assert abs(d - 6.0) < 1e-9

    def test_ray_near_left_wall_rightward(self):
        origin = np.array([-5.0, 50.0])
        direction = np.array([1.0, 0.0])
        d = self.hw.ray_intersect(origin, direction)
        assert abs(d - 11.0) < 1e-9  # right wall at +6, distance = 11

    def test_ray_near_left_wall_leftward(self):
        origin = np.array([-5.0, 50.0])
        direction = np.array([-1.0, 0.0])
        d = self.hw.ray_intersect(origin, direction)
        assert abs(d - 1.0) < 1e-9  # left wall at -6, distance = 1

    def test_ray_along_y_axis_no_wall_hit(self):
        # Rays parallel to walls never hit
        origin = np.array([0.0, 50.0])
        direction = np.array([0.0, 1.0])
        d = self.hw.ray_intersect(origin, direction)
        assert d == 30.0  # default max_range

    def test_ray_diagonal_hits_wall(self):
        origin = np.array([0.0, 50.0])
        direction = np.array([1.0, 1.0]) / np.sqrt(2)
        d = self.hw.ray_intersect(origin, direction)
        # x-component: hits right wall at t = 6 / (1/√2) = 6√2 ≈ 8.485
        assert abs(d - 6.0 * np.sqrt(2)) < 1e-6

    # --- ray_intersect with obstacle ---

    def test_ray_hits_obstacle_before_wall(self):
        obs = RectObstacle(x_min=-1.5, x_max=1.5, y_min=46.0, y_max=54.0)
        hw = Hallway(width=12.0, obstacles=[obs])
        origin = np.array([0.0, 40.0])
        direction = np.array([0.0, 1.0])
        # No wall hit for vertical ray; obstacle at y=46 → distance 6
        d = hw.ray_intersect(origin, direction)
        assert abs(d - 6.0) < 1e-6

    def test_wall_closer_than_obstacle(self):
        obs = RectObstacle(x_min=4.0, x_max=8.0, y_min=0.0, y_max=10.0)  # partly outside hallway
        hw = Hallway(width=12.0, obstacles=[obs])
        origin = np.array([0.0, 5.0])
        direction = np.array([1.0, 0.0])
        # right wall at x=6, obstacle at x=4; obstacle is closer
        d = hw.ray_intersect(origin, direction)
        assert d <= 4.0 + 1e-9

    # --- clamp_position ---

    def test_clamp_inside_unchanged(self):
        pos = np.array([0.0, 50.0])
        out = self.hw.clamp_position(pos)
        np.testing.assert_array_almost_equal(out, pos)

    def test_clamp_beyond_left_wall(self):
        pos = np.array([-10.0, 50.0])
        out = self.hw.clamp_position(pos)
        assert out[0] >= self.hw.x_left

    def test_clamp_beyond_right_wall(self):
        pos = np.array([10.0, 50.0])
        out = self.hw.clamp_position(pos)
        assert out[0] <= self.hw.x_right

    def test_clamp_y_unchanged(self):
        pos = np.array([0.0, 999.0])
        out = self.hw.clamp_position(pos)
        assert out[1] == 999.0

    def test_clamp_inside_obstacle_is_pushed_out(self):
        obs = RectObstacle(x_min=-1.0, x_max=1.0, y_min=4.0, y_max=6.0)
        hw = Hallway(width=12.0, obstacles=[obs])
        pos = np.array([0.0, 5.0])  # inside obstacle
        out = hw.clamp_position(pos)
        assert not (obs.x_min < out[0] < obs.x_max and obs.y_min < out[1] < obs.y_max)

    # --- is_inside ---

    def test_is_inside_center(self):
        assert self.hw.is_inside(np.array([0.0, 50.0]))

    def test_is_inside_left_wall(self):
        assert not self.hw.is_inside(np.array([-6.0, 50.0]))

    def test_is_inside_right_wall(self):
        assert not self.hw.is_inside(np.array([6.0, 50.0]))

    def test_is_inside_obstacle(self):
        obs = RectObstacle(x_min=-1.0, x_max=1.0, y_min=4.0, y_max=6.0)
        hw = Hallway(width=12.0, obstacles=[obs])
        assert not hw.is_inside(np.array([0.0, 5.0]))

    def test_is_inside_outside_obstacle_still_in_hallway(self):
        obs = RectObstacle(x_min=-1.0, x_max=1.0, y_min=4.0, y_max=6.0)
        hw = Hallway(width=12.0, obstacles=[obs])
        assert hw.is_inside(np.array([3.0, 5.0]))

    # --- edge: narrow hallway ---

    def test_narrow_hallway(self):
        hw = Hallway(width=2.0)
        assert hw.x_left == -1.0
        assert hw.x_right == 1.0

    def test_no_obstacles_by_default(self):
        assert self.hw.obstacles == []
