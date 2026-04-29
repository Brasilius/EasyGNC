import numpy as np
from simulation.hallway import Hallway
from swarm.swarm import create_swarm, step
import fusion.ekf as ekf
import fusion.random_weighting as rw
import fusion.owa as owa
import control.leader_follower as lf
import control.consensus as consensus
import control.behavior as behavior

FUSION_METHODS = {
    "EKF": ekf.step,
    "Random Weighting": rw.step,
    "OWA": owa.step,
}

CONTROL_METHODS = {
    "Leader-Follower": lf.compute_velocities,
    "Consensus": consensus.compute_velocities,
    "Behavior-Based": behavior.compute_velocities,
}

NUM_STEPS = 500


def _metrics(drones: list, hallway: Hallway) -> dict:
    pos_rmses, min_clearances, spreads = [], [], []

    for drone in drones:
        true_xs = np.array([p[0] for p in drone.pos_history])
        est_xs = np.array([e[0] for e in drone.est_history])
        pos_rmses.append(float(np.sqrt(np.mean((true_xs - est_xs) ** 2))))

        d_left = true_xs - hallway.x_left
        d_right = hallway.x_right - true_xs
        min_clearances.append(float(np.min(np.minimum(d_left, d_right))))

    n_steps = len(drones[0].pos_history)
    for t in range(n_steps):
        xs = [d.pos_history[t][0] for d in drones]
        spreads.append(float(np.std(xs)))

    return {
        "pos_rmse": float(np.mean(pos_rmses)),
        "min_wall_clearance": float(np.min(min_clearances)),
        "avg_formation_spread": float(np.mean(spreads)),
    }


def run_all(hallway: Hallway | None = None, seed: int = 42) -> dict:
    """Run all 9 fusion × control combinations. Returns a results dict keyed by (fusion, control)."""
    if hallway is None:
        hallway = Hallway()

    results = {}
    combos = len(FUSION_METHODS) * len(CONTROL_METHODS)
    done = 0

    for fusion_name, fusion_fn in FUSION_METHODS.items():
        for ctrl_name, ctrl_fn in CONTROL_METHODS.items():
            np.random.seed(seed)
            drones = create_swarm(hallway)
            for _ in range(NUM_STEPS):
                step(drones, hallway, fusion_fn, ctrl_fn)

            m = _metrics(drones, hallway)
            m["trajectories"] = [d.pos_history for d in drones]
            results[(fusion_name, ctrl_name)] = m

            done += 1
            print(f"  [{done}/{combos}] {fusion_name} + {ctrl_name}  "
                  f"RMSE={m['pos_rmse']:.4f}m")

    return results
