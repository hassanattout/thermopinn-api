import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from fem.heat_solver_2d import solve_steady_state_heat_2d
from pinn.inference import load_model


def generate_pinn_map(output_path="results/pinn_thermal_map.png"):
    model = load_model()

    nx, ny = 50, 50

    x_grid, y_grid, T_solver = solve_steady_state_heat_2d(
        nx=nx,
        ny=ny,
        length=0.1,
        width=0.05,
        heat_source=100000,
        boundary_temperature=25.0,
    )

    x_norm = x_grid / 0.1
    y_norm = y_grid / 0.05

    coords = np.stack([x_norm.flatten(), y_norm.flatten()], axis=1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32)

    with torch.no_grad():
        T_pinn_norm = model(coords_tensor).numpy().reshape((ny, nx))

    T_min = T_solver.min()
    T_max = T_solver.max()
    T_pinn = T_pinn_norm * (T_max - T_min) + T_min

    plt.figure()
    contour = plt.contourf(x_grid, y_grid, T_pinn, levels=50)
    plt.colorbar(contour, label="Temperature (°C)")
    plt.title("PINN Thermal Map")
    plt.xlabel("Length (m)")
    plt.ylabel("Width (m)")

    plt.savefig(output_path)
    plt.close()

    print(f"PINN thermal map saved to {output_path}")
    print(f"Minimum PINN temperature: {T_pinn.min():.2f} °C")
    print(f"Maximum PINN temperature: {T_pinn.max():.2f} °C")


if __name__ == "__main__":
    generate_pinn_map()
