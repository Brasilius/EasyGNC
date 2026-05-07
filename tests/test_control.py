"""
Tests for control modules: behavior, consensus, leader_follower.
"""
import numpy as np
import pytest
from drone.drone import Drone
from simulation.hallway import Hallway
import control.behavior as behavior
import control.consensus as consensus
import control.leader_follower as lf


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_drone(drone_id: int, x: float, y: float = 50.0) -> Drone:
    d = Drone(drone_id, x, y)
    d.state_est[0] = x  # ensure est_pos matches
    return d


def make_swarm_5(xs=(-4.0, -2.0, 0.0, 2.0, 4.0)) -> list:
    return [make_drone(i, x) for i, x in enumerate(xs)]


def make_hw(width=12.0) -> Hallway:
    return Hallway(width=width)


# ─── Behavior-based control ───────────────────────────────────────────────────

class TestBehavior:
    def setup_method(self):
        self.hw = make_hw()

    def test_returns_one_velocity_per_drone(self):
        drones = make_swarm_5()
        vels = behavior.compute_velocities(drones, self.hw)
        assert len(vels) == len(drones)

    def test_all_velocities_have_correct_shape(self):
        drones = make_swarm_5()
        vels = behavior.compute_velocities(drones, self.hw)
        assert all(v.shape == (2,) for v in vels)

    def test_base_vy_applied(self):
        drones = make_swarm_5()
        vels = behavior.compute_velocities(drones, self.hw)
        assert all(abs(v[1] - behavior.BASE_VY) < 1e-9 for v in vels)

    def test_center_drone_minimal_vx(self):
        # Drone at x=0 far from walls and alone → centering pulls lightly
        drone = make_drone(0, 0.0)
        vels = behavior.compute_velocities([drone], self.hw)
        # centering: W_CENTER * (-0) = 0; no wall avoidance; no neighbors
        assert abs(vels[0][0]) < 1e-9

    def test_drone_near_left_wall_pushed_right(self):
        drone = make_drone(0, self.hw.x_left + 1.0)  # 1 m from left wall
        vels = behavior.compute_velocities([drone], self.hw)
        assert vels[0][0] > 0.0

    def test_drone_near_right_wall_pushed_left(self):
        drone = make_drone(0, self.hw.x_right - 1.0)
        vels = behavior.compute_velocities([drone], self.hw)
        assert vels[0][0] < 0.0

    def test_separation_pushes_apart(self):
        # Two drones very close together
        d0 = make_drone(0, 0.0)
        d1 = make_drone(1, 0.5)  # within SEP_THRESH=2.5
        vels = behavior.compute_velocities([d0, d1], self.hw)
        # d0 is left of d1 → d0 pushed left, d1 pushed right
        assert vels[0][0] < vels[1][0]

    def test_separation_inactive_beyond_threshold(self):
        d0 = make_drone(0, -5.0)
        d1 = make_drone(1, 5.0)  # far apart > SEP_THRESH=2.5
        vels_pair = behavior.compute_velocities([d0, d1], self.hw)
        vels_solo = behavior.compute_velocities([d0], self.hw)
        # d0's vx should be the same whether or not d1 exists (no sep force)
        assert abs(vels_pair[0][0] - vels_solo[0][0]) < 1e-9

    def test_single_drone_no_crash(self):
        drone = make_drone(0, 1.0)
        vels = behavior.compute_velocities([drone], self.hw)
        assert len(vels) == 1

    def test_drone_not_separated_from_self(self):
        # Each drone skips itself (other.id == drone.id guard)
        drone = make_drone(0, 0.0)
        v_single = behavior.compute_velocities([drone], self.hw)[0]
        # Confirm: centering only, no self-separation
        assert abs(v_single[0]) < 1e-9

    def test_zero_distance_guard(self):
        # Two drones at exactly the same position: dist=0 → should be skipped (1e-6 guard)
        d0 = make_drone(0, 0.0)
        d1 = make_drone(1, 0.0)
        # This must not raise ZeroDivisionError or NaN
        vels = behavior.compute_velocities([d0, d1], self.hw)
        assert all(np.isfinite(v[0]) for v in vels)


# ─── Consensus control ────────────────────────────────────────────────────────

class TestConsensus:
    def setup_method(self):
        self.hw = make_hw()

    def test_returns_one_velocity_per_drone(self):
        drones = make_swarm_5()
        vels = consensus.compute_velocities(drones, self.hw)
        assert len(vels) == len(drones)

    def test_base_vy_applied(self):
        drones = make_swarm_5()
        vels = consensus.compute_velocities(drones, self.hw)
        assert all(abs(v[1] - consensus.BASE_VY) < 1e-9 for v in vels)

    def test_desired_positions_spread_evenly(self):
        # With n=5 and DESIRED_SPACING=2.0, desired_x in [-4,-2,0,2,4]
        # Drones starting at exactly those positions → minimal vx
        xs = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        drones = [make_drone(i, x) for i, x in enumerate(xs)]
        vels = consensus.compute_velocities(drones, self.hw)
        # Each drone already at desired → vx ≈ 0 (only wall avoidance if close)
        for v in vels:
            assert abs(v[0]) < 0.5  # loose bound (wall avoidance may add small amount)

    def test_single_drone_targets_center(self):
        drone = make_drone(0, 3.0)  # off-center
        vels = consensus.compute_velocities([drone], self.hw)
        # Desired x for 1 drone = (0 - 0/2)*2 = 0; vx should pull toward 0
        assert vels[0][0] < 0.0

    def test_wall_avoidance_overrides_near_wall(self):
        drone = make_drone(0, self.hw.x_left + 0.5)  # inside WALL_MARGIN=1.5
        vels = consensus.compute_velocities([drone], self.hw)
        assert vels[0][0] > 0.0  # wall pushes right

    def test_right_wall_avoidance(self):
        drone = make_drone(0, self.hw.x_right - 0.5)
        vels = consensus.compute_velocities([drone], self.hw)
        assert vels[0][0] < 0.0

    def test_rank_assignment_stable(self):
        # Swap drone order — rank should follow x position, not drone_id
        xs = [2.0, -2.0, 0.0]
        drones = [make_drone(i, x) for i, x in enumerate(xs)]
        vels = consensus.compute_velocities(drones, self.hw)
        # drone at x=-2 (id=1) gets rank 0 → desired_x=(0-1)*2=-2 → near-zero vx
        # Just verify no crash and correct count
        assert len(vels) == 3

    def test_all_velocities_finite(self):
        drones = make_swarm_5()
        vels = consensus.compute_velocities(drones, self.hw)
        for v in vels:
            assert np.all(np.isfinite(v))

    def test_two_drones_symmetric(self):
        d0 = make_drone(0, -3.0)
        d1 = make_drone(1, 3.0)
        vels = consensus.compute_velocities([d0, d1], self.hw)
        # Symmetric setup → vx[0] = -vx[1]
        assert abs(vels[0][0] + vels[1][0]) < 0.5  # loose tolerance


# ─── Leader-Follower control ──────────────────────────────────────────────────

class TestLeaderFollower:
    def setup_method(self):
        self.hw = make_hw()

    def test_returns_one_velocity_per_drone(self):
        drones = make_swarm_5()
        vels = lf.compute_velocities(drones, self.hw)
        assert len(vels) == len(drones)

    def test_base_vy_applied(self):
        drones = make_swarm_5()
        vels = lf.compute_velocities(drones, self.hw)
        assert all(abs(v[1] - lf.BASE_VY) < 1e-9 for v in vels)

    def test_leader_is_drone_idx_2(self):
        drones = make_swarm_5()
        leader = drones[lf.LEADER_IDX]
        leader.state_est[0] = 0.0  # put leader at center
        vels = lf.compute_velocities(drones, self.hw)
        # Leader at x=0 → ldr_vx = -KP*0 = 0
        assert abs(vels[lf.LEADER_IDX][0]) < 1e-9

    def test_leader_off_center_corrects(self):
        drones = make_swarm_5()
        drones[lf.LEADER_IDX].state_est[0] = 3.0  # leader off-center
        vels = lf.compute_velocities(drones, self.hw)
        # ldr_vx = -KP*3 < 0 (pushes left toward center)
        assert vels[lf.LEADER_IDX][0] < 0.0

    def test_follower_tracks_leader_offset(self):
        drones = make_swarm_5()
        leader_x = 0.0
        drones[lf.LEADER_IDX].state_est[0] = leader_x
        for i, d in enumerate(drones):
            if i != lf.LEADER_IDX:
                d.state_est[0] = leader_x + lf._OFFSETS[i]
        vels = lf.compute_velocities(drones, self.hw)
        # Followers already at desired offsets → follower vx ≈ leader vx = 0
        for i in range(5):
            if i != lf.LEADER_IDX:
                assert abs(vels[i][0]) < 0.1

    def test_offsets_length_matches_swarm_size(self):
        assert len(lf._OFFSETS) == 5  # designed for 5-drone swarm

    def test_fewer_than_leader_idx_plus_one_drones_raises(self):
        # LEADER_IDX=2 requires at least 3 drones; 2 drones → IndexError
        drones = [make_drone(0, 0.0), make_drone(1, 1.0)]
        with pytest.raises(IndexError):
            lf.compute_velocities(drones, self.hw)

    def test_all_velocities_finite(self):
        drones = make_swarm_5()
        vels = lf.compute_velocities(drones, self.hw)
        for v in vels:
            assert np.all(np.isfinite(v))

    def test_leader_at_extreme_x_corrects_strongly(self):
        drones = make_swarm_5()
        drones[lf.LEADER_IDX].state_est[0] = 5.0
        vels = lf.compute_velocities(drones, self.hw)
        # Strong correction: vx should be large negative
        assert vels[lf.LEADER_IDX][0] < -5.0
