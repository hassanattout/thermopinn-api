from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import numpy as np
import time
from pathlib import Path

from fem.heat_solver_2d import solve_steady_state_heat_2d
from fem.visualize import generate_thermal_map
from pinn.inference import load_model, predict_temperature


app = FastAPI(
    title="ThermoPINN API",
    description="Physics-informed thermal prediction API for engineering systems.",
    version="0.3.0",
)

RESULTS_DIR = Path("results")
MODEL_PATH = Path("models/pinn_model.pth")

try:
    pinn_model = load_model()
    MODEL_STATUS = "loaded"
except Exception as e:
    pinn_model = None
    MODEL_STATUS = f"error: {str(e)}"


class SimulationInput(BaseModel):
    length: float = Field(default=0.1, gt=0)
    width: float = Field(default=0.05, gt=0)
    heat_power: float = Field(default=100000, gt=0)
    ambient_temperature: float = 25.0


class PointInput(BaseModel):
    x: float = Field(..., ge=0, le=1)
    y: float = Field(..., ge=0, le=1)


class GridInput(BaseModel):
    grid_size: int = Field(default=20, ge=5, le=100)


@app.get("/")
def root():
    return {
        "message": "ThermoPINN API",
        "status": "running",
        "version": "0.3.0",
        "model_status": MODEL_STATUS,
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_status": MODEL_STATUS,
        "model_exists": MODEL_PATH.exists(),
    }


@app.post("/simulate")
def simulate(data: SimulationInput):
    start = time.perf_counter()

    x, y, T = solve_steady_state_heat_2d(
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
        "min_temperature": float(np.min(T)),
        "max_temperature": float(np.max(T)),
        "mean_temperature": float(np.mean(T)),
        "solver_time_seconds": round(solver_time, 6),
        "thermal_map": "results/thermal_map.png",
    }


@app.get("/thermal-map")
def get_thermal_map():
    image_path = RESULTS_DIR / "thermal_map.png"

    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Thermal map not found. Run /simulate first.",
        )

    return FileResponse(image_path)


@app.post("/predict-pinn")
def predict_pinn(data: PointInput):
    if pinn_model is None:
        raise HTTPException(
            status_code=500,
            detail=f"PINN model not loaded. {MODEL_STATUS}",
        )

    start = time.perf_counter()
    temp = predict_temperature(pinn_model, data.x, data.y)
    inference_time = time.perf_counter() - start

    return {
        "x": data.x,
        "y": data.y,
        "predicted_temperature": float(temp),
        "inference_time_seconds": round(inference_time, 6),
    }


@app.post("/predict-grid")
def predict_grid(data: GridInput):
    if pinn_model is None:
        raise HTTPException(
            status_code=500,
            detail=f"PINN model not loaded. {MODEL_STATUS}",
        )

    start = time.perf_counter()

    xs = np.linspace(0, 1, data.grid_size)
    ys = np.linspace(0, 1, data.grid_size)

    temperature_grid = []

    for y in ys:
        row = []
        for x in xs:
            temp = predict_temperature(pinn_model, float(x), float(y))
            row.append(float(temp))
        temperature_grid.append(row)

    inference_time = time.perf_counter() - start
    arr = np.array(temperature_grid)

    return {
        "message": "PINN grid prediction completed",
        "grid_size": data.grid_size,
        "min_temperature": float(np.min(arr)),
        "max_temperature": float(np.max(arr)),
        "mean_temperature": float(np.mean(arr)),
        "inference_time_seconds": round(inference_time, 6),
        "temperature_grid": temperature_grid,
    }


@app.post("/compare")
def compare(data: SimulationInput):
    if pinn_model is None:
        raise HTTPException(
            status_code=500,
            detail=f"PINN model not loaded. {MODEL_STATUS}",
        )

    fem_start = time.perf_counter()

    x, y, T_fem = solve_steady_state_heat_2d(
        length=data.length,
        width=data.width,
        heat_source=data.heat_power,
        boundary_temperature=data.ambient_temperature,
    )

    fem_time = time.perf_counter() - fem_start

    pinn_start = time.perf_counter()

    rows, cols = T_fem.shape
    T_pinn = np.zeros_like(T_fem, dtype=float)

    for i in range(rows):
        for j in range(cols):
            x_norm = j / (cols - 1)
            y_norm = i / (rows - 1)
            T_pinn[i, j] = predict_temperature(pinn_model, x_norm, y_norm)

    pinn_time = time.perf_counter() - pinn_start

    error = T_pinn - T_fem
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    max_error = float(np.max(np.abs(error)))

    speedup = fem_time / pinn_time if pinn_time > 0 else None

    return {
        "message": "FEM vs PINN comparison completed",
        "fem": {
            "min_temperature": float(np.min(T_fem)),
            "max_temperature": float(np.max(T_fem)),
            "mean_temperature": float(np.mean(T_fem)),
            "time_seconds": round(fem_time, 6),
        },
        "pinn": {
            "min_temperature": float(np.min(T_pinn)),
            "max_temperature": float(np.max(T_pinn)),
            "mean_temperature": float(np.mean(T_pinn)),
            "time_seconds": round(pinn_time, 6),
        },
        "error_metrics": {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "max_error": round(max_error, 6),
        },
        "speedup": round(speedup, 3) if speedup else None,
    }


@app.get("/metrics")
def get_metrics():
    metrics_path = RESULTS_DIR / "benchmark_metrics.csv"

    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="benchmark_metrics.csv not found.",
        )

    return FileResponse(metrics_path)
