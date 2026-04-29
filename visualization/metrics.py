"""
Metrics heatmap: one subplot per metric, rows = control method, cols = fusion method.
Green = better, red = worse (colour scale direction is metric-specific).
"""
import numpy as np
import matplotlib.pyplot as plt

_FUSION_ORDER = ["EKF", "Random Weighting", "OWA"]
_CONTROL_ORDER = ["Leader-Follower", "Consensus", "Behavior-Based"]

_METRICS = [
    ("pos_rmse",             "Position RMSE (m)",          "RdYlGn_r"),  # lower better
    ("min_wall_clearance",   "Min Wall Clearance (m)",      "RdYlGn"),    # higher better
    ("avg_formation_spread", "Avg Formation Spread (m)",    "RdYlGn_r"),  # lower better
]


def plot_metrics_table(results: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        "Performance Metrics  |  rows = control method, cols = fusion method",
        fontsize=13,
        fontweight="bold",
    )

    for ax, (key, label, cmap) in zip(axes, _METRICS):
        data = np.zeros((3, 3))
        for r, ctrl in enumerate(_CONTROL_ORDER):
            for c, fusion in enumerate(_FUSION_ORDER):
                data[r, c] = results.get((fusion, ctrl), {}).get(key, np.nan)

        vmin, vmax = np.nanmin(data), np.nanmax(data)
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1e-9

        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(_FUSION_ORDER, fontsize=8, rotation=15, ha="right")
        ax.set_yticklabels(_CONTROL_ORDER, fontsize=8)
        ax.set_title(label, fontsize=10, fontweight="bold")

        for r in range(3):
            for c in range(3):
                val = data[r, c]
                norm = (val - vmin) / (vmax - vmin)
                txt_color = "white" if abs(norm - 0.5) > 0.35 else "black"
                ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=txt_color, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    return fig
