from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from fem.heat_solver_2d import solve_steady_state_heat_2d
from fem.visualize import generate_thermal_map

from pinn.inference import load_model, predict_temperature

app = FastAPI(
    title="ThermoPINN API",
    description="Physics-informed thermal prediction API for engineering systems.",
    version="0.2.0",
)

# Load PINN once at startup
pinn_model = load_model()


class SimulationInput(BaseModel):
    length: float = 0.1
    width: float = 0.05
    heat_power: float = 100000
    ambient_temperature: float = 25.0


class PointInput(BaseModel):
    x: float
    y: float


@app.get("/")
def root():
    return {"message": "ThermoPINN API", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/simulate")
def simulate(data: SimulationInput):
    x, y, T = solve_steady_state_heat_2d(
        length=data.length,
        width=data.width,
        heat_source=data.heat_power,
        boundary_temperature=data.ambient_temperature,
    )

    max_temp = float(np.max(T))
    min_temp = float(np.min(T))

    generate_thermal_map()

    return {
        "min_temperature": min_temp,
        "max_temperature": max_temp,
        "message": "Simulation completed",
        "thermal_map": "results/thermal_map.png"
    }


@app.post("/predict-pinn")
def predict_pinn(data: PointInput):
    temp = predict_temperature(pinn_model, data.x, data.y)

    return {
        "x": data.x,
        "y": data.y,
        "predicted_temperature": temp
    }
