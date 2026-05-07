"""
Tests for drone/drone.py — Drone state, properties, and movement.
"""
import numpy as np
import pytest
from drone.drone import Drone
from simulation.hallway import Hallway


class TestDrone:
    def setup_method(self):
        self.hw = Hallway(width=12.0)
        self.drone = Drone(drone_id=0, x=1.0, y=5.0, vy=2.0)

    # --- construction ---

    def test_initial_true_pos(self):
        np.testing.assert_array_equal(self.drone.true_pos, [1.0, 5.0])

    def test_initial_true_vel(self):
        np.testing.assert_array_equal(self.drone.true_vel, [0.0, 2.0])

    def test_initial_state_est(self):
        np.testing.assert_array_equal(self.drone.state_est, [1.0, 5.0, 0.0, 2.0])

    def test_initial_cov_shape(self):
        assert self.drone.cov.shape == (4, 4)

    def test_initial_history_length(self):
        assert len(self.drone.pos_history) == 1
        assert len(self.drone.est_history) == 1

    def test_default_y_is_zero(self):
        d = Drone(0, 0.0)
        np.testing.assert_array_equal(d.true_pos, [0.0, 0.0])

    def test_default_vy(self):
        d = Drone(0, 0.0)
        assert d.true_vel[1] == 2.0

    # --- properties ---

    def test_est_pos_returns_xy(self):
        np.testing.assert_array_equal(self.drone.est_pos, [1.0, 5.0])

    def test_est_vel_returns_vxvy(self):
        np.testing.assert_array_equal(self.drone.est_vel, [0.0, 2.0])

    def test_est_pos_reflects_state_est_update(self):
        self.drone.state_est[0] = 3.0
        assert self.drone.est_pos[0] == 3.0

    # --- apply_velocity ---

    def test_apply_velocity_moves_drone(self):
        vel = np.array([1.0, 2.0])
        self.drone.apply_velocity(vel, dt=0.1, env=self.hw)
        assert abs(self.drone.true_pos[0] - 1.1) < 1e-9
        assert abs(self.drone.true_pos[1] - 5.2) < 1e-9

    def test_apply_velocity_stores_velocity(self):
        vel = np.array([0.5, 3.0])
        self.drone.apply_velocity(vel, dt=0.1, env=self.hw)
        np.testing.assert_array_equal(self.drone.true_vel, vel)

    def test_apply_velocity_zero_dt(self):
        orig = self.drone.true_pos.copy()
        self.drone.apply_velocity(np.array([5.0, 5.0]), dt=0.0, env=self.hw)
        np.testing.assert_array_almost_equal(self.drone.true_pos, orig)

    def test_apply_velocity_clamps_to_wall(self):
        # Push far right — should be clamped
        self.drone.apply_velocity(np.array([100.0, 0.0]), dt=1.0, env=self.hw)
        assert self.drone.true_pos[0] <= self.hw.x_right

    def test_apply_velocity_clamps_to_left_wall(self):
        self.drone.apply_velocity(np.array([-100.0, 0.0]), dt=1.0, env=self.hw)
        assert self.drone.true_pos[0] >= self.hw.x_left

    def test_apply_velocity_copies_input(self):
        vel = np.array([1.0, 2.0])
        self.drone.apply_velocity(vel, dt=0.1, env=self.hw)
        vel[0] = 99.0  # mutate original
        assert self.drone.true_vel[0] == 1.0  # stored copy unaffected

    # --- record ---

    def test_record_appends_history(self):
        self.drone.record()
        assert len(self.drone.pos_history) == 2
        assert len(self.drone.est_history) == 2

    def test_record_stores_copy(self):
        self.drone.record()
        snap = self.drone.pos_history[-1].copy()
        self.drone.true_pos[0] = 999.0
        np.testing.assert_array_equal(self.drone.pos_history[-1], snap)

    def test_multiple_records(self):
        for _ in range(10):
            self.drone.apply_velocity(np.array([0.0, 2.0]), dt=0.1, env=self.hw)
            self.drone.record()
        assert len(self.drone.pos_history) == 11  # 1 initial + 10 records

    # --- id ---

    def test_drone_id_stored(self):
        assert self.drone.id == 0

    def test_distinct_ids(self):
        d1 = Drone(1, 0.0)
        d2 = Drone(2, 0.0)
        assert d1.id != d2.id
