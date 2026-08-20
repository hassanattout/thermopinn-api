from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    description="""
Physics-informed thermal prediction API for engineering systems.

This API provides:

- FEM-based thermal simulation
- PINN surrogate inference
- Batch prediction
- Grid prediction
- FEM vs PINN benchmarking
- Generated thermal map access
- Deployment-ready engineering ML backend

This project demonstrates how physics-based simulation and machine learning can be combined into a real API system for simulation acceleration, digital twins, thermal monitoring, and optimization workflows.
""",
    version=settings.app_version,
    contact={
        "name": "Hassan Attout",
    },
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

app.include_router(router)
