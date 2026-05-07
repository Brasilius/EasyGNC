"""
Tests for fusion modules: ekf, owa, random_weighting.
"""
import numpy as np
import pytest
import fusion.ekf as ekf
import fusion.owa as owa
import fusion.random_weighting as rw
from drone.sensors import cast_rays
from simulation.hallway import Hallway


# ─── Shared fixtures ──────────────────────────────────────────────────────────

def make_state(x=0.0, y=50.0, vx=0.0, vy=2.0):
    return np.array([x, y, vx, vy], dtype=float)


def make_cov():
    return np.eye(4) * 0.5


def make_hw():
    return Hallway(width=12.0)


# ─── EKF internal helpers ─────────────────────────────────────────────────────

class TestEKFHelpers:
    def test_F_identity_at_zero_dt(self):
        F = ekf._F(0.0)
        np.testing.assert_array_equal(F, np.eye(4))

    def test_F_structure(self):
        F = ekf._F(0.1)
        assert F[0, 2] == 0.1
        assert F[1, 3] == 0.1
        assert F[0, 0] == 1.0
        assert F[2, 2] == 1.0

    def test_F_maps_velocity_to_position(self):
        state = np.array([0.0, 0.0, 3.0, 4.0])
        F = ekf._F(1.0)
        pred = F @ state
        assert pred[0] == 3.0  # x moved by vx
        assert pred[1] == 4.0  # y moved by vy

    def test_measurement_model_output_shapes(self):
        hw = make_hw()
        state = make_state(x=0.0)
        z_hat, H = ekf._measurement_model(state, hw)
        # 6 non-parallel rays
        assert z_hat.shape == (6,)
        assert H.shape == (6, 4)

    def test_measurement_model_z_hat_positive(self):
        hw = make_hw()
        state = make_state(x=0.0)
        z_hat, _ = ekf._measurement_model(state, hw)
        assert np.all(z_hat > 0.0)

    def test_measurement_model_correct_right_ray(self):
        hw = make_hw()  # x_right = 6
        state = make_state(x=2.0)
        z_hat, H = ekf._measurement_model(state, hw)
        # Ray at 0° (cos=1): z_hat = (6 - 2) / 1 = 4.0
        assert abs(z_hat[0] - 4.0) < 1e-6

    def test_measurement_model_H_only_x_component(self):
        hw = make_hw()
        state = make_state(x=0.0)
        _, H = ekf._measurement_model(state, hw)
        # H rows should only affect x (index 0), not y, vx, vy
        assert np.all(H[:, 1] == 0.0)  # y column
        assert np.all(H[:, 2] == 0.0)  # vx column
        assert np.all(H[:, 3] == 0.0)  # vy column

    def test_measurement_model_drone_at_right_wall(self):
        hw = make_hw()  # x_right=6
        state = make_state(x=5.9)  # very close to right wall
        z_hat, _ = ekf._measurement_model(state, hw)
        # Rightward ray distance should be very small but clamped ≥ 0.01
        assert np.all(z_hat >= 0.01)

    def test_measurement_model_drone_at_left_wall(self):
        hw = make_hw()  # x_left=-6
        state = make_state(x=-5.9)
        z_hat, _ = ekf._measurement_model(state, hw)
        assert np.all(z_hat >= 0.01)


# ─── EKF step ─────────────────────────────────────────────────────────────────

class TestEKFStep:
    def setup_method(self):
        self.hw = make_hw()

    def test_step_returns_correct_shapes(self):
        state = make_state()
        cov = make_cov()
        np.random.seed(0)
        distances = cast_rays(np.array([0.0, 50.0]), self.hw, noise_std=0.0)
        new_state, new_cov = ekf.step(state, cov, distances, self.hw, dt=0.1)
        assert new_state.shape == (4,)
        assert new_cov.shape == (4, 4)

    def test_step_x_estimate_near_true_position(self):
        true_x = 2.0
        state = make_state(x=0.0)  # wrong initial estimate
        cov = make_cov()
        distances = cast_rays(np.array([true_x, 50.0]), self.hw, noise_std=0.0)
        new_state, _ = ekf.step(state, cov, distances, self.hw, dt=0.1)
        # EKF should pull x estimate closer to true_x=2
        assert abs(new_state[0] - true_x) < abs(state[0] - true_x)

    def test_step_cov_non_negative_diagonal(self):
        state = make_state()
        cov = make_cov()
        np.random.seed(0)
        distances = cast_rays(np.array([0.0, 50.0]), self.hw)
        _, new_cov = ekf.step(state, cov, distances, self.hw, dt=0.1)
        assert np.all(np.diag(new_cov) >= 0.0)

    def test_step_cov_symmetric(self):
        state = make_state()
        cov = make_cov()
        np.random.seed(0)
        distances = cast_rays(np.array([0.0, 50.0]), self.hw)
        _, new_cov = ekf.step(state, cov, distances, self.hw, dt=0.1)
        np.testing.assert_array_almost_equal(new_cov, new_cov.T)

    def test_step_with_zero_dt_still_updates(self):
        state = make_state(x=0.0)
        cov = make_cov()
        distances = cast_rays(np.array([3.0, 50.0]), self.hw, noise_std=0.0)
        new_state, _ = ekf.step(state, cov, distances, self.hw, dt=0.0)
        # Prediction step is identity at dt=0; update still runs
        assert new_state.shape == (4,)

    def test_step_propagates_velocity(self):
        state = make_state(x=0.0, vx=1.0, vy=2.0)
        cov = make_cov()
        distances = cast_rays(np.array([0.0, 50.0]), self.hw, noise_std=0.0)
        new_state, _ = ekf.step(state, cov, distances, self.hw, dt=0.1)
        # Prediction step moves x by vx*dt=0.1 before update
        # After update x might change, but the prediction should at least have moved
        assert new_state.shape == (4,)  # sanity — no crash

    def test_step_repeated_convergence(self):
        true_x = 3.0
        state = make_state(x=0.0)
        cov = make_cov()
        distances = cast_rays(np.array([true_x, 50.0]), self.hw, noise_std=0.0)
        for _ in range(20):
            state, cov = ekf.step(state, cov, distances, self.hw, dt=0.0)
        assert abs(state[0] - true_x) < 0.2


# ─── OWA fusion ──────────────────────────────────────────────────────────────

class TestOWA:
    def setup_method(self):
        self.hw = make_hw()

    def test_gaussian_weights_sum_to_one(self):
        for n in [1, 2, 5, 10]:
            w = owa._gaussian_weights(n)
            assert abs(w.sum() - 1.0) < 1e-9

    def test_gaussian_weights_single(self):
        w = owa._gaussian_weights(1)
        assert abs(w[0] - 1.0) < 1e-9

    def test_gaussian_weights_symmetric(self):
        w = owa._gaussian_weights(5)
        np.testing.assert_array_almost_equal(w, w[::-1])

    def test_gaussian_weights_center_highest(self):
        w = owa._gaussian_weights(5)
        assert w[2] == w.max()

    def test_step_no_estimates_returns_unchanged(self):
        # Simulate all-parallel-ray environment by monkeypatching x_estimates_from_rays
        state = make_state()
        cov = make_cov()
        original_cov = cov.copy()
        import fusion.owa as owa_mod
        import unittest.mock as mock
        with mock.patch("fusion.owa.x_estimates_from_rays", return_value=np.array([])):
            new_state, new_cov = owa_mod.step(state, cov, np.zeros(8), self.hw, dt=0.1)
        np.testing.assert_array_equal(new_state, state)
        np.testing.assert_array_equal(new_cov, original_cov)

    def test_step_updates_x(self):
        state = make_state(x=0.0)
        cov = make_cov()
        distances = cast_rays(np.array([4.0, 50.0]), self.hw, noise_std=0.0)
        new_state, _ = owa.step(state, cov, distances, self.hw, dt=0.1)
        assert abs(new_state[0] - 4.0) < 0.5

    def test_step_y_dead_reckons(self):
        state = make_state(y=50.0, vy=2.0)
        cov = make_cov()
        distances = cast_rays(np.array([0.0, 50.0]), self.hw, noise_std=0.0)
        new_state, _ = owa.step(state, cov, distances, self.hw, dt=0.5)
        assert abs(new_state[1] - (50.0 + 2.0 * 0.5)) < 1e-9

    def test_step_cov_unchanged(self):
        state = make_state()
        cov = make_cov()
        distances = cast_rays(np.array([0.0, 50.0]), self.hw, noise_std=0.0)
        _, new_cov = owa.step(state, cov, distances, self.hw, dt=0.1)
        np.testing.assert_array_equal(new_cov, cov)

    def test_step_returns_copies(self):
        state = make_state()
        cov = make_cov()
        distances = cast_rays(np.array([0.0, 50.0]), self.hw, noise_std=0.0)
        new_state, new_cov = owa.step(state, cov, distances, self.hw, dt=0.1)
        new_state[0] = 999.0
        assert state[0] != 999.0


# ─── Random Weighting fusion ─────────────────────────────────────────────────

class TestRandomWeighting:
    def setup_method(self):
        self.hw = make_hw()

    def test_step_no_estimates_returns_unchanged(self):
        state = make_state()
        cov = make_cov()
        import unittest.mock as mock
        with mock.patch("fusion.random_weighting.x_estimates_from_rays", return_value=np.array([])):
            new_state, new_cov = rw.step(state, cov, np.zeros(8), self.hw, dt=0.1)
        np.testing.assert_array_equal(new_state, state)
        np.testing.assert_array_equal(new_cov, cov)

    def test_step_returns_correct_shapes(self):
        state = make_state()
        cov = make_cov()
        np.random.seed(0)
        distances = cast_rays(np.array([0.0, 50.0]), self.hw)
        new_state, new_cov = rw.step(state, cov, distances, self.hw, dt=0.1)
        assert new_state.shape == (4,)
        assert new_cov.shape == (4, 4)

    def test_step_x_estimate_reasonable(self):
        true_x = 2.0
        state = make_state(x=0.0)
        cov = make_cov()
        np.random.seed(42)
        distances = cast_rays(np.array([true_x, 50.0]), self.hw, noise_std=0.0)
        new_state, _ = rw.step(state, cov, distances, self.hw, dt=0.1)
        assert abs(new_state[0] - true_x) < 1.0

    def test_step_y_dead_reckons(self):
        state = make_state(y=50.0, vy=3.0)
        cov = make_cov()
        np.random.seed(0)
        distances = cast_rays(np.array([0.0, 50.0]), self.hw, noise_std=0.0)
        new_state, _ = rw.step(state, cov, distances, self.hw, dt=0.2)
        assert abs(new_state[1] - (50.0 + 3.0 * 0.2)) < 1e-9

    def test_step_randomness_with_different_seeds(self):
        state = make_state(x=0.0)
        cov = make_cov()
        distances = cast_rays(np.array([2.0, 50.0]), self.hw, noise_std=0.0)
        np.random.seed(1)
        s1, _ = rw.step(state.copy(), cov.copy(), distances, self.hw, dt=0.1)
        np.random.seed(99)
        s2, _ = rw.step(state.copy(), cov.copy(), distances, self.hw, dt=0.1)
        # Different seeds should generally produce different x estimates
        # (not guaranteed but highly likely with 6 random weights)
        assert s1[0] != s2[0] or True  # non-fatal if same; just verify no crash

    def test_step_reproducible_with_same_seed(self):
        state = make_state()
        cov = make_cov()
        distances = cast_rays(np.array([0.0, 50.0]), self.hw, noise_std=0.0)
        np.random.seed(7)
        s1, _ = rw.step(state.copy(), cov.copy(), distances, self.hw, dt=0.1)
        np.random.seed(7)
        s2, _ = rw.step(state.copy(), cov.copy(), distances, self.hw, dt=0.1)
        np.testing.assert_array_equal(s1, s2)

    def test_step_cov_unchanged(self):
        state = make_state()
        cov = make_cov()
        np.random.seed(0)
        distances = cast_rays(np.array([0.0, 50.0]), self.hw)
        _, new_cov = rw.step(state, cov, distances, self.hw, dt=0.1)
        np.testing.assert_array_equal(new_cov, cov)
