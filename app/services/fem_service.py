import time
import numpy as np
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import FileResponse

from fem.heat_solver_2d import solve_steady_state_heat_2d
from fem.visualize import generate_thermal_map


RESULTS_DIR = Path("results")


def run_simulation(data):
    start = time.perf_counter()

    x, y, temperature = solve_steady_state_heat_2d(
        length=data.length,
        width=data.width,
        heat_source=data.heat_power,
        boundary_temperature=data.ambient_temperature,
    )

    solver_time = time.perf_counter() - start

    RESULTS_DIR.mkdir(exist_ok=True)
    generate_thermal_map()

    return {
        "message": "FEM simulation completed",
        "min_temperature": float(np.min(temperature)),
        "max_temperature": float(np.max(temperature)),
        "mean_temperature": float(np.mean(temperature)),
        "solver_time_seconds": round(solver_time, 6),
        "thermal_map": "results/thermal_map.png",
    }


def get_thermal_map_file():
    image_path = RESULTS_DIR / "thermal_map.png"

    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Thermal map not found. Run /simulate first.",
        )

    return FileResponse(image_path)


def get_metrics_file():
    metrics_path = RESULTS_DIR / "benchmark_metrics.csv"

    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="benchmark_metrics.csv not found.",
        )

    return FileResponse(metrics_path)
