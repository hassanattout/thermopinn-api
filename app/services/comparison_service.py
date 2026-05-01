import time

import numpy as np
from fastapi import HTTPException

from app.services.pinn_service import pinn_service
from fem.heat_solver_2d import solve_steady_state_heat_2d
from pinn.inference import predict_temperature


def compare_fem_vs_pinn(data):
    if pinn_service.model is None:
        raise HTTPException(
            status_code=500,
            detail=f"PINN model not loaded. Status: {pinn_service.status}",
        )

    fem_start = time.perf_counter()

    x, y, fem_temperature = solve_steady_state_heat_2d(
        length=data.length,
        width=data.width,
        heat_source=data.heat_power,
        boundary_temperature=data.ambient_temperature,
    )

    fem_time = time.perf_counter() - fem_start

    pinn_start = time.perf_counter()

    rows, cols = fem_temperature.shape
    pinn_temperature = np.zeros_like(fem_temperature, dtype=float)

    for i in range(rows):
        for j in range(cols):
            x_norm = j / (cols - 1)
            y_norm = i / (rows - 1)
            pinn_temperature[i, j] = predict_temperature(
                pinn_service.model,
                x_norm,
                y_norm,
            )

    pinn_time = time.perf_counter() - pinn_start

    error = pinn_temperature - fem_temperature

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    max_error = float(np.max(np.abs(error)))

    speedup = fem_time / pinn_time if pinn_time > 0 else None

    return {
        "message": "FEM vs PINN comparison completed",
        "fem": {
            "min_temperature": float(np.min(fem_temperature)),
            "max_temperature": float(np.max(fem_temperature)),
            "mean_temperature": float(np.mean(fem_temperature)),
            "time_seconds": round(fem_time, 6),
        },
        "pinn": {
            "min_temperature": float(np.min(pinn_temperature)),
            "max_temperature": float(np.max(pinn_temperature)),
            "mean_temperature": float(np.mean(pinn_temperature)),
            "time_seconds": round(pinn_time, 6),
        },
        "error_metrics": {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "max_error": round(max_error, 6),
        },
        "speedup": round(speedup, 3) if speedup else None,
    }
