import matplotlib.pyplot as plt
from simulation.hallway import Hallway, RectObstacle
from swarm.runner import run_all
from visualization.animator import animate_all, plot_trajectories
from visualization.metrics import plot_metrics_table


def main() -> None:
    print("=" * 60)
    print("  EasyGNC  |  5-Drone Swarm Simulation")
    print("  3 fusion methods × 3 control methods = 9 combinations")
    print("=" * 60)

    # Central pillar: 3 m wide × 8 m deep, centred at (0, 50)
    obstacle = RectObstacle(x_min=-1.5, x_max=1.5, y_min=46.0, y_max=54.0)
    hallway = Hallway(width=12.0, length=100.0, obstacles=[obstacle])

    print(f"\nEnvironment: hallway  width={hallway.width}m  length={hallway.length}m")
    print(f"Obstacle:    pillar   x=[{obstacle.x_min},{obstacle.x_max}]  "
          f"y=[{obstacle.y_min},{obstacle.y_max}]")
    print(f"Running 500 steps @ dt=0.1s per combination...\n")

    results = run_all(hallway=hallway, seed=42)

    print("\n--- Summary ---")
    fmt = "  {:<20s}| {:<18s}| RMSE={:.4f}m  Clearance={:.3f}m  Spread={:.3f}m"
    for (fusion, ctrl), m in results.items():
        print(fmt.format(fusion, ctrl, m["pos_rmse"], m["min_wall_clearance"], m["avg_formation_spread"]))

    print("\nGenerating live animation…  (close window to continue to static plots)")
    anim = animate_all(results, hallway, interval=25)
    plt.show()

    print("Generating static plots…")
    fig1 = plot_trajectories(results, hallway)
    fig2 = plot_metrics_table(results)
    plt.show()


if __name__ == "__main__":
    main()
