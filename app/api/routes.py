from pathlib import Path

from fastapi import APIRouter

from app.schemas.thermal import (BatchPointInput, GridInput, PointInput,
                                 SimulationInput)
from app.services.comparison_service import compare_fem_vs_pinn
from app.services.fem_service import (get_metrics_file, get_thermal_map_file,
                                      run_simulation)
from app.services.pinn_service import MODEL_PATH, pinn_service

router = APIRouter()


@router.get("/", tags=["System"])
def root():
    return {
        "message": "ThermoPINN API",
        "status": "running",
        "version": "1.0.0",
        "model_status": pinn_service.status,
    }


@router.get("/health", tags=["System"])
def health_check():
    return {
        "status": "ok",
        "model_status": pinn_service.status,
        "model_exists": MODEL_PATH.exists(),
    }


@router.post("/simulate", tags=["FEM Solver"])
def simulate(data: SimulationInput):
    return run_simulation(data)


@router.get("/thermal-map", tags=["FEM Solver"])
def thermal_map():
    return get_thermal_map_file()


@router.post("/predict-pinn", tags=["PINN Inference"])
def predict_pinn(data: PointInput):
    return pinn_service.predict_point(data.x, data.y)


@router.post("/predict-batch", tags=["PINN Inference"])
def predict_batch(data: BatchPointInput):
    return pinn_service.predict_batch(data.points)


@router.post("/predict-grid", tags=["PINN Inference"])
def predict_grid(data: GridInput):
    return pinn_service.predict_grid(data.grid_size)


@router.post("/compare", tags=["Benchmarking"])
def compare(data: SimulationInput):
    return compare_fem_vs_pinn(data)


@router.get("/metrics", tags=["Benchmarking"])
def metrics():
    return get_metrics_file()
