import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from fem.heat_solver_2d import solve_steady_state_heat_2d
from pinn.inference import load_model


def generate_benchmark_artifacts():
    model = load_model()
    nx, ny = 50, 50

    solver_start = time.time()
    x_grid, y_grid, T_solver = solve_steady_state_heat_2d(
        nx=nx,
        ny=ny,
        length=0.1,
        width=0.05,
        heat_source=100000,
        boundary_temperature=25.0,
    )
    solver_time = time.time() - solver_start

    x_norm = x_grid / 0.1
    y_norm = y_grid / 0.05

    coords = np.stack([x_norm.flatten(), y_norm.flatten()], axis=1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32)

    pinn_start = time.time()
    with torch.no_grad():
        T_pinn_norm = model(coords_tensor).numpy().reshape((ny, nx))
    pinn_time = time.time() - pinn_start

    T_min = T_solver.min()
    T_max = T_solver.max()
    T_pinn = T_pinn_norm * (T_max - T_min) + T_min

    error = T_pinn - T_solver

    vmin = min(T_solver.min(), T_pinn.min())
    vmax = max(T_solver.max(), T_pinn.max())

    abs_error = np.abs(error)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    cs0 = axs[0].contourf(x_grid, y_grid, T_solver, levels=50, vmin=vmin, vmax=vmax)
    axs[0].set_title("Numerical Solver")
    axs[0].set_xlabel("Length (m)")
    axs[0].set_ylabel("Width (m)")

    cs1 = axs[1].contourf(x_grid, y_grid, T_pinn, levels=50, vmin=vmin, vmax=vmax)
    axs[1].set_title("PINN Prediction")
    axs[1].set_xlabel("Length (m)")
    axs[1].set_ylabel("Width (m)")

    cs2 = axs[2].contourf(x_grid, y_grid, abs_error, levels=50)
    axs[2].set_title("Absolute Error")
    axs[2].set_xlabel("Length (m)")
    axs[2].set_ylabel("Width (m)")

    fig.colorbar(cs0, ax=axs[:2], label="Temperature (°C)")
    fig.colorbar(cs2, ax=axs[2], label="Error (°C)")

    plt.savefig("results/pinn_vs_solver_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))

    with open("results/benchmark_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["MAE_C", mae])
        writer.writerow(["RMSE_C", rmse])
        writer.writerow(["Max_Error_C", max_error])
        writer.writerow(["Solver_Time_s", solver_time])
        writer.writerow(["PINN_Inference_Time_s", pinn_time])

    plt.figure()
    labels = ["MAE", "RMSE", "Max Error"]
    values = [mae, rmse, max_error]
    plt.bar(labels, values)
    plt.ylabel("Error (°C)")
    plt.title("PINN vs Solver Error Metrics")
    plt.savefig("results/error_metrics.png")
    plt.close()

    print("Benchmark artifacts generated:")
    print("results/benchmark_metrics.csv")
    print("results/error_metrics.png")
    print("results/pinn_vs_solver_comparison.png")


if __name__ == "__main__":
    generate_benchmark_artifacts()
