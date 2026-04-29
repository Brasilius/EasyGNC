"""
Trajectory plots: 3×3 grid of (fusion row) × (control column) subplots.
Each subplot shows the X–Y path of all 5 drones through the hallway.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
_DRONE_LABELS = [f"Drone {i}" for i in range(5)]

_FUSION_ORDER = ["EKF", "Random Weighting", "OWA"]
_CONTROL_ORDER = ["Leader-Follower", "Consensus", "Behavior-Based"]


def plot_trajectories(results: dict, hallway) -> plt.Figure:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    fig.suptitle(
        "Drone Swarm Trajectories  |  rows = control method, cols = fusion method",
        fontsize=13,
        fontweight="bold",
    )

    for row, ctrl in enumerate(_CONTROL_ORDER):
        for col, fusion in enumerate(_FUSION_ORDER):
            ax = axes[row][col]
            key = (fusion, ctrl)
            data = results.get(key)

            ax.set_facecolor("#f5f5f5")
            ax.axvline(hallway.x_left, color="#555", linewidth=2)
            ax.axvline(hallway.x_right, color="#555", linewidth=2)

            if data:
                for d_idx, traj in enumerate(data["trajectories"]):
                    xs = [p[0] for p in traj]
                    ys = [p[1] for p in traj]
                    ax.plot(xs, ys, color=_COLORS[d_idx], linewidth=0.9, alpha=0.85)
                    ax.plot(xs[0], ys[0], "o", color=_COLORS[d_idx], markersize=4)
                    ax.plot(xs[-1], ys[-1], "s", color=_COLORS[d_idx], markersize=4)

            if row == 0:
                ax.set_title(fusion, fontsize=10, fontweight="bold", pad=6)
            if col == 0:
                ax.set_ylabel(ctrl, fontsize=9, labelpad=4)
            if row == 2:
                ax.set_xlabel("X (m)", fontsize=9)

            ax.set_xlim(hallway.x_left - 0.8, hallway.x_right + 0.8)
            ax.tick_params(labelsize=7)

    legend_elements = [
        Line2D([0], [0], color=_COLORS[i], linewidth=1.5, label=_DRONE_LABELS[i])
        for i in range(5)
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.005),
    )
    plt.tight_layout(rect=[0, 0.045, 1, 1])
    return fig
