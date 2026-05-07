"""
Tests for swarm/swarm.py and swarm/runner.py.
"""
import numpy as np
import pytest
from simulation.hallway import Hallway, RectObstacle
from swarm.swarm import create_swarm, step, _obstacle_avoidance, INITIAL_X, DT
from swarm.runner import run_all, _metrics, FUSION_METHODS, CONTROL_METHODS
import fusion.ekf as ekf
import control.behavior as behavior


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_hw(with_obstacle=False):
    if with_obstacle:
        obs = RectObstacle(x_min=-1.5, x_max=1.5, y_min=46.0, y_max=54.0)
        return Hallway(width=12.0, obstacles=[obs])
    return Hallway(width=12.0)


# ─── create_swarm ─────────────────────────────────────────────────────────────

class TestCreateSwarm:
    def test_correct_count(self):
        drones = create_swarm(make_hw())
        assert len(drones) == len(INITIAL_X)

    def test_initial_x_positions(self):
        drones = create_swarm(make_hw())
        for drone, expected_x in zip(drones, INITIAL_X):
            assert abs(drone.true_pos[0] - expected_x) < 1e-9

    def test_initial_y_position(self):
        drones = create_swarm(make_hw())
        for drone in drones:
            assert drone.true_pos[1] == 0.0

    def test_drone_ids_assigned(self):
        drones = create_swarm(make_hw())
        ids = [d.id for d in drones]
        assert ids == list(range(len(INITIAL_X)))

    def test_each_drone_independent(self):
        drones = create_swarm(make_hw())
        drones[0].true_pos[0] = 99.0
        assert drones[1].true_pos[0] != 99.0

    def test_narrow_hallway_drones_start_outside(self):
        # Hallway narrower than formation: drones at ±4 but hallway is ±1.
        # create_swarm doesn't clamp — just verify it doesn't raise.
        hw = Hallway(width=2.0)
        drones = create_swarm(hw)
        assert len(drones) == 5


# ─── _obstacle_avoidance ─────────────────────────────────────────────────────

class TestObstacleAvoidance:
    def test_no_obstacles_unchanged(self):
        hw = make_hw(with_obstacle=False)
        vel = np.array([1.0, 2.0])
        pos = np.array([0.0, 50.0])
        out = _obstacle_avoidance(vel, pos, hw)
        np.testing.assert_array_equal(out, vel)

    def test_obstacle_repels_laterally(self):
        obs = RectObstacle(x_min=-1.5, x_max=1.5, y_min=46.0, y_max=54.0)
        hw = Hallway(width=12.0, obstacles=[obs])
        # Drone approaching from the right side of the obstacle
        pos = np.array([3.0, 50.0])  # within _OBS_RADIUS=6 of obs surface at x=1.5
        vel = np.array([0.0, 2.0])
        out = _obstacle_avoidance(vel, pos, hw)
        # Repulsion should push drone to the right (positive vx)
        assert out[0] > 0.0

    def test_obstacle_repels_from_left(self):
        obs = RectObstacle(x_min=-1.5, x_max=1.5, y_min=46.0, y_max=54.0)
        hw = Hallway(width=12.0, obstacles=[obs])
        pos = np.array([-3.0, 50.0])  # left of obstacle
        vel = np.array([0.0, 2.0])
        out = _obstacle_avoidance(vel, pos, hw)
        assert out[0] < 0.0

    def test_decel_when_approaching_obstacle_ahead(self):
        obs = RectObstacle(x_min=-1.5, x_max=1.5, y_min=46.0, y_max=54.0)
        hw = Hallway(width=12.0, obstacles=[obs])
        # Drone directly in front (within x-span), close but not yet at obstacle
        pos = np.array([0.0, 40.0])  # dy_ahead = 46 - 40 = 6 < _OBS_Y_AHEAD=10
        vel = np.array([0.0, 2.0])
        out = _obstacle_avoidance(vel, pos, hw)
        assert out[1] < vel[1]  # vy reduced

    def test_no_decel_when_past_obstacle(self):
        obs = RectObstacle(x_min=-1.5, x_max=1.5, y_min=46.0, y_max=54.0)
        hw = Hallway(width=12.0, obstacles=[obs])
        # Drone past obstacle
        pos = np.array([0.0, 60.0])  # dy_ahead = 46 - 60 < 0
        vel = np.array([0.0, 2.0])
        out = _obstacle_avoidance(vel, pos, hw)
        assert abs(out[1] - vel[1]) < 1e-9  # vy unchanged

    def test_no_decel_when_outside_x_range(self):
        obs = RectObstacle(x_min=-1.5, x_max=1.5, y_min=46.0, y_max=54.0)
        hw = Hallway(width=12.0, obstacles=[obs])
        # Drone far to the side, not in x-range
        pos = np.array([5.0, 40.0])
        vel = np.array([0.0, 2.0])
        out = _obstacle_avoidance(vel, pos, hw)
        assert abs(out[1] - vel[1]) < 1e-9

    def test_drone_exactly_at_obstacle_x_center_no_crash(self):
        # diff[0]=0 triggers the lateral fallback branch
        obs = RectObstacle(x_min=-1.5, x_max=1.5, y_min=46.0, y_max=54.0)
        hw = Hallway(width=12.0, obstacles=[obs])
        # Drone directly above obstacle, within influence radius
        pos = np.array([0.0, 42.0])  # directly above, 4m from top face
        vel = np.array([0.0, 2.0])
        out = _obstacle_avoidance(vel, pos, hw)
        assert np.all(np.isfinite(out))

    def test_far_from_obstacle_no_repulsion(self):
        obs = RectObstacle(x_min=-1.5, x_max=1.5, y_min=46.0, y_max=54.0)
        hw = Hallway(width=12.0, obstacles=[obs])
        pos = np.array([0.0, 0.0])  # far below
        vel = np.array([1.0, 2.0])
        out = _obstacle_avoidance(vel, pos, hw)
        # Below _OBS_Y_AHEAD from obs.y_min, and outside _OBS_RADIUS from surface
        assert abs(out[0] - vel[0]) < 1e-6  # no lateral repulsion

    def test_output_is_ndarray(self):
        hw = make_hw()
        vel = np.array([0.0, 2.0])
        pos = np.array([0.0, 50.0])
        out = _obstacle_avoidance(vel, pos, hw)
        assert isinstance(out, np.ndarray)


# ─── swarm step ───────────────────────────────────────────────────────────────

class TestSwarmStep:
    def test_history_grows_by_one(self):
        hw = make_hw()
        drones = create_swarm(hw)
        np.random.seed(0)
        step(drones, hw, ekf.step, behavior.compute_velocities)
        for drone in drones:
            assert len(drone.pos_history) == 2
            assert len(drone.est_history) == 2

    def test_y_state_est_synced_to_true_pos(self):
        hw = make_hw()
        drones = create_swarm(hw)
        np.random.seed(0)
        step(drones, hw, ekf.step, behavior.compute_velocities)
        for drone in drones:
            assert drone.state_est[1] == drone.true_pos[1]

    def test_positions_stay_within_hallway(self):
        hw = make_hw()
        drones = create_swarm(hw)
        np.random.seed(0)
        for _ in range(50):
            step(drones, hw, ekf.step, behavior.compute_velocities)
        for drone in drones:
            assert hw.x_left <= drone.true_pos[0] <= hw.x_right

    def test_state_est_shape_preserved(self):
        hw = make_hw()
        drones = create_swarm(hw)
        np.random.seed(0)
        step(drones, hw, ekf.step, behavior.compute_velocities)
        for drone in drones:
            assert drone.state_est.shape == (4,)

    def test_step_with_obstacle_no_crash(self):
        hw = make_hw(with_obstacle=True)
        drones = create_swarm(hw)
        np.random.seed(0)
        for _ in range(10):
            step(drones, hw, ekf.step, behavior.compute_velocities)


# ─── _metrics ────────────────────────────────────────────────────────────────

class TestMetrics:
    def _run_n_steps(self, n=10, seed=42):
        hw = make_hw()
        drones = create_swarm(hw)
        np.random.seed(seed)
        for _ in range(n):
            step(drones, hw, ekf.step, behavior.compute_velocities)
        return drones, hw

    def test_metrics_keys(self):
        drones, hw = self._run_n_steps()
        m = _metrics(drones, hw)
        assert "pos_rmse" in m
        assert "min_wall_clearance" in m
        assert "avg_formation_spread" in m

    def test_rmse_non_negative(self):
        drones, hw = self._run_n_steps()
        m = _metrics(drones, hw)
        assert m["pos_rmse"] >= 0.0

    def test_wall_clearance_positive(self):
        drones, hw = self._run_n_steps()
        m = _metrics(drones, hw)
        assert m["min_wall_clearance"] > 0.0

    def test_spread_non_negative(self):
        drones, hw = self._run_n_steps()
        m = _metrics(drones, hw)
        assert m["avg_formation_spread"] >= 0.0

    def test_spread_zero_when_all_same_x(self):
        hw = make_hw()
        drones = create_swarm(hw)
        # Force all drones to same x in history
        for d in drones:
            d.pos_history = [np.array([0.0, 0.0])]
        m = _metrics(drones, hw)
        assert abs(m["avg_formation_spread"]) < 1e-9


# ─── run_all ──────────────────────────────────────────────────────────────────

class TestRunAll:
    def test_returns_nine_combinations(self):
        hw = make_hw()
        results = run_all(hallway=hw, seed=42)
        assert len(results) == len(FUSION_METHODS) * len(CONTROL_METHODS)

    def test_all_keys_present(self):
        hw = make_hw()
        results = run_all(hallway=hw, seed=42)
        for fusion in FUSION_METHODS:
            for ctrl in CONTROL_METHODS:
                assert (fusion, ctrl) in results

    def test_each_result_has_metrics(self):
        hw = make_hw()
        results = run_all(hallway=hw, seed=42)
        for m in results.values():
            assert "pos_rmse" in m
            assert "min_wall_clearance" in m
            assert "avg_formation_spread" in m
            assert "trajectories" in m

    def test_trajectories_length(self):
        from swarm.runner import NUM_STEPS
        hw = make_hw()
        results = run_all(hallway=hw, seed=42)
        for m in results.values():
            for traj in m["trajectories"]:
                # pos_history starts with 1 initial entry + NUM_STEPS recorded
                assert len(traj) == NUM_STEPS + 1

    def test_default_hallway_is_created(self):
        # run_all without hallway argument should not raise
        results = run_all(seed=42)
        assert len(results) == 9

    def test_deterministic_with_same_seed(self):
        hw = make_hw()
        r1 = run_all(hallway=hw, seed=0)
        r2 = run_all(hallway=hw, seed=0)
        for key in r1:
            assert abs(r1[key]["pos_rmse"] - r2[key]["pos_rmse"]) < 1e-12

    def test_different_seeds_give_different_results(self):
        hw = make_hw()
        r1 = run_all(hallway=hw, seed=0)
        r2 = run_all(hallway=hw, seed=999)
        # At least one combination should differ
        differs = any(
            abs(r1[k]["pos_rmse"] - r2[k]["pos_rmse"]) > 1e-6
            for k in r1
        )
        assert differs

    def test_custom_hallway_respected(self):
        hw = Hallway(width=20.0)
        results = run_all(hallway=hw, seed=42)
        # With wider hallway, wall clearance should generally be higher
        assert len(results) == 9
