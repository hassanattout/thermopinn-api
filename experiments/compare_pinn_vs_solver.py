import numpy as np
import torch

from fem.heat_solver_2d import solve_steady_state_heat_2d
from pinn.inference import load_model


def compare_pinn_vs_solver():
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

    error = T_pinn - T_solver

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))

    print("PINN vs Solver Comparison")
    print(f"MAE: {mae:.4f} °C")
    print(f"RMSE: {rmse:.4f} °C")
    print(f"Max Error: {max_error:.4f} °C")


if __name__ == "__main__":
    compare_pinn_vs_solver()
