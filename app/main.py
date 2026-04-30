from fastapi import FastAPI
from pydantic import BaseModel
from fem.heat_solver_2d import solve_steady_state_heat_2d
from fem.visualize import generate_thermal_map
import numpy as np

app = FastAPI(
    title="ThermoPINN API",
    description="Physics-informed thermal prediction API for engineering systems.",
    version="0.1.0",
)


class SimulationInput(BaseModel):
    length: float = 0.1
    width: float = 0.05
    heat_power: float = 100000
    ambient_temperature: float = 25.0


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
