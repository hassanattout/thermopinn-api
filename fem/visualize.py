import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from fem.heat_solver_2d import solve_steady_state_heat_2d


def generate_thermal_map(output_path="results/thermal_map.png"):
    x, y, T = solve_steady_state_heat_2d(heat_source=100000)

    plt.figure()
    contour = plt.contourf(x, y, T, levels=50)
    plt.colorbar(contour, label="Temperature (°C)")
    plt.title("2D Thermal Map")
    plt.xlabel("Length (m)")
    plt.ylabel("Width (m)")

    plt.savefig(output_path)
    plt.close()

    print(f"Thermal map saved to {output_path}")


if __name__ == "__main__":
    generate_thermal_map()
