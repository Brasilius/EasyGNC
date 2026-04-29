"""
Trajectory plots and live animation for the 3×3 fusion × control grid.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle as MplRect
from matplotlib.animation import FuncAnimation

_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]
_DRONE_LABELS = [f"Drone {i}" for i in range(5)]

_FUSION_ORDER = ["EKF", "Random Weighting", "OWA"]
_CONTROL_ORDER = ["Leader-Follower", "Consensus", "Behavior-Based"]

_TRAIL_LEN = 60  # steps of history shown in animation
_DT = 0.1


def _draw_static_env(ax, hallway):
    """Draw walls and obstacles on an axes (shared by static and animated views)."""
    ax.set_facecolor("#f5f5f5")
    ax.axvline(hallway.x_left, color="#555", linewidth=2)
    ax.axvline(hallway.x_right, color="#555", linewidth=2)
    for obs in hallway.obstacles:
        rect = MplRect(
            (obs.x_min, obs.y_min),
            obs.x_max - obs.x_min,
            obs.y_max - obs.y_min,
            linewidth=1.5,
            edgecolor="#222",
            facecolor="#aaaaaa",
            zorder=3,
        )
        ax.add_patch(rect)


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

            _draw_static_env(ax, hallway)

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


def animate_all(results: dict, hallway, interval: int = 25) -> FuncAnimation:
    """
    Animated 3×3 grid: all 5 drones move live through the hallway.
    Each dot is a drone; short trails show recent history.
    Returns a FuncAnimation — call plt.show() after to display.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 18))
    fig.suptitle(
        "Live Drone Swarm  |  rows = control method, cols = fusion method",
        fontsize=13,
        fontweight="bold",
    )

    art_grid: dict = {}

    for row, ctrl in enumerate(_CONTROL_ORDER):
        for col, fusion in enumerate(_FUSION_ORDER):
            ax = axes[row][col]
            key = (fusion, ctrl)
            m = results.get(key, {})
            trajs = m.get("trajectories", [])

            _draw_static_env(ax, hallway)
            ax.set_xlim(hallway.x_left - 1.0, hallway.x_right + 1.0)
            ax.set_ylim(-5.0, hallway.length + 5.0)
            ax.tick_params(labelsize=6)

            if row == 0:
                ax.set_title(fusion, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(ctrl, fontsize=8)
            if row == 2:
                ax.set_xlabel("X (m)", fontsize=8)

            # Mark start positions as faint circles
            for i, traj in enumerate(trajs):
                if traj:
                    ax.plot(traj[0][0], traj[0][1], "o", color=_COLORS[i],
                            markersize=3, alpha=0.4, zorder=2)

            # Initial scatter uses first-frame positions so colors bind correctly
            if trajs:
                init_xs = [traj[0][0] for traj in trajs]
                init_ys = [traj[0][1] for traj in trajs]
                dots = ax.scatter(
                    init_xs, init_ys,
                    s=55, zorder=5, c=_COLORS[: len(trajs)],
                    edgecolors="white", linewidths=0.5,
                )
            else:
                dots = ax.scatter([], [], s=55, zorder=5)

            trails = [
                ax.plot([], [], color=_COLORS[i], linewidth=0.9, alpha=0.75, zorder=4)[0]
                for i in range(len(trajs))
            ]
            step_txt = ax.text(
                0.03, 0.97, "t=0.0s",
                transform=ax.transAxes,
                fontsize=7, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75),
                zorder=6,
            )

            art_grid[(row, col)] = {
                "dots": dots,
                "trails": trails,
                "trajs": trajs,
                "step_txt": step_txt,
            }

    # Total frames = length of longest trajectory
    n_frames = max(
        len(m.get("trajectories", [[]])[0])
        for m in results.values()
        if m.get("trajectories")
    )

    def update(frame):
        artists_out: list = []
        for art in art_grid.values():
            trajs = art["trajs"]
            if not trajs:
                continue
            t = min(frame, len(trajs[0]) - 1)
            xs = [traj[t][0] for traj in trajs]
            ys = [traj[t][1] for traj in trajs]
            art["dots"].set_offsets(np.c_[xs, ys])

            t0 = max(0, t - _TRAIL_LEN)
            for trail, traj in zip(art["trails"], trajs):
                trail.set_data(
                    [traj[s][0] for s in range(t0, t + 1)],
                    [traj[s][1] for s in range(t0, t + 1)],
                )

            art["step_txt"].set_text(f"t={t * _DT:.1f}s")
            artists_out.append(art["dots"])
            artists_out.extend(art["trails"])

        return artists_out

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)
    return anim
