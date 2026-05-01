from fastapi import FastAPI
from app.api.routes import router


app = FastAPI(
    title="ThermoPINN API",
    description="""
Physics-informed thermal prediction API for engineering systems.

This API combines:

- FEM-based thermal simulation
- Physics-informed neural network inference
- Batch and grid prediction
- FEM vs PINN benchmarking
- Real-time engineering ML deployment

Use this project as a deployable backend for simulation acceleration, digital twins, optimization loops, and thermal monitoring systems.
""",
    version="1.0.0",
    contact={
        "name": "Hassan Attout",
    },
)

app.include_router(router)
