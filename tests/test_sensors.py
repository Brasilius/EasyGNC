"""
Tests for drone/sensors.py — cast_rays and x_estimates_from_rays.
"""
import numpy as np
import pytest
from drone.sensors import cast_rays, x_estimates_from_rays, NUM_RAYS, RAY_ANGLES, MAX_RANGE
from simulation.hallway import Hallway, RectObstacle


class TestCastRays:
    def setup_method(self):
        self.hw = Hallway(width=12.0)

    def test_returns_eight_distances(self):
        pos = np.array([0.0, 50.0])
        d = cast_rays(pos, self.hw, noise_std=0.0)
        assert d.shape == (NUM_RAYS,)

    def test_symmetric_from_center(self):
        # With no noise, drone at center should see equal distances left/right
        pos = np.array([0.0, 50.0])
        d = cast_rays(pos, self.hw, noise_std=0.0)
        # Ray 0 (0°) hits right wall; ray 4 (180°) hits left wall — both at 6 m
        assert abs(d[0] - 6.0) < 1e-6
        assert abs(d[4] - 6.0) < 1e-6

    def test_distances_positive(self):
        np.random.seed(0)
        pos = np.array([0.0, 50.0])
        d = cast_rays(pos, self.hw)
        assert np.all(d > 0.0)

    def test_near_left_wall_short_left_ray(self):
        pos = np.array([-5.0, 50.0])
        d = cast_rays(pos, self.hw, noise_std=0.0)
        # Ray 4 (180°, leftward) should be ~1 m; ray 0 (0°, rightward) ~11 m
        assert d[4] < d[0]

    def test_noise_adds_variation(self):
        pos = np.array([0.0, 50.0])
        np.random.seed(1)
        d1 = cast_rays(pos, self.hw, noise_std=0.5)
        np.random.seed(2)
        d2 = cast_rays(pos, self.hw, noise_std=0.5)
        assert not np.allclose(d1, d2)

    def test_zero_noise_reproducible(self):
        pos = np.array([2.0, 50.0])
        d1 = cast_rays(pos, self.hw, noise_std=0.0)
        d2 = cast_rays(pos, self.hw, noise_std=0.0)
        np.testing.assert_array_equal(d1, d2)

    def test_distances_capped_at_max_range(self):
        # Along y-axis (90° and 270°) no wall is hit → returns max_range + noise
        np.random.seed(0)
        pos = np.array([0.0, 50.0])
        d = cast_rays(pos, self.hw, noise_std=0.0)
        # No ray should exceed MAX_RANGE significantly with no noise
        assert np.all(d <= MAX_RANGE + 1e-6)

    def test_with_obstacle_closer_than_wall(self):
        obs = RectObstacle(x_min=-0.5, x_max=0.5, y_min=53.0, y_max=57.0)
        hw = Hallway(width=12.0, obstacles=[obs])
        pos = np.array([0.0, 50.0])
        # Ray 2 (90°) is upward — hits obstacle at distance ~3
        d = cast_rays(pos, hw, noise_std=0.0)
        # The 90° ray: no wall hit (parallel to walls), but obstacle ahead
        assert d[2] < MAX_RANGE  # obstacle shortens that ray

    def test_minimum_distance_floor(self):
        # Even with large negative noise, distance should be ≥ 0.01
        pos = np.array([0.0, 50.0])
        np.random.seed(0)
        # Patch noise to be very large negative
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("numpy.random.normal", lambda *a, **kw: -9999.0)
            d = cast_rays(pos, self.hw, noise_std=1.0)
        assert np.all(d >= 0.01)


class TestXEstimatesFromRays:
    def setup_method(self):
        self.hw = Hallway(width=12.0)  # x_left=-6, x_right=6

    def test_returns_array(self):
        pos = np.array([0.0, 50.0])
        distances = cast_rays(pos, self.hw, noise_std=0.0)
        ests = x_estimates_from_rays(distances, self.hw)
        assert isinstance(ests, np.ndarray)

    def test_parallel_rays_excluded(self):
        # Rays at 90° (index 2) and 270° (index 6) have cos≈0 and must be excluded.
        # Total of 6 non-parallel rays for 8 rays at multiples of 45°.
        pos = np.array([0.0, 50.0])
        distances = cast_rays(pos, self.hw, noise_std=0.0)
        ests = x_estimates_from_rays(distances, self.hw)
        assert len(ests) == 6

    def test_estimates_near_true_x_no_noise(self):
        pos = np.array([2.0, 50.0])
        distances = cast_rays(pos, self.hw, noise_std=0.0)
        ests = x_estimates_from_rays(distances, self.hw)
        # All estimates should be close to true x=2
        assert np.all(np.abs(ests - 2.0) < 0.5)

    def test_estimates_within_hallway_bounds(self):
        np.random.seed(42)
        pos = np.array([3.0, 50.0])
        distances = cast_rays(pos, self.hw)
        ests = x_estimates_from_rays(distances, self.hw)
        # Despite noise, estimates should stay within or very near hallway
        assert np.all(ests > self.hw.x_left - 1.0)
        assert np.all(ests < self.hw.x_right + 1.0)

    def test_empty_if_all_distances_for_parallel_env(self):
        # If we pass all-parallel ray angles → no estimates.
        # Simulate by giving a fake env with x_left==x_right (degenerate hallway)
        # and confirming no crash — the guard `if len(x_ests) == 0` in OWA/RW handles it.
        # We verify this via the function directly: create fake distances for a
        # 2-ray-only setup where both happen to be 90°/270°.
        # In practice with RAY_ANGLES this is guarded: always ≥6 rays valid.
        # Just verify the function doesn't crash on empty result from the env side.
        class FakeEnv:
            x_left = -6.0
            x_right = 6.0
        # Pass zero-length distances array (not possible from cast_rays, but defensive)
        ests = x_estimates_from_rays(np.array([]), FakeEnv())
        assert len(ests) == 0

    def test_symmetry_at_center(self):
        pos = np.array([0.0, 50.0])
        distances = cast_rays(pos, self.hw, noise_std=0.0)
        ests = x_estimates_from_rays(distances, self.hw)
        # At center, all estimates should be ~0 with no noise
        assert np.abs(np.mean(ests)) < 0.1
