from pydantic import BaseModel, Field


class SimulationInput(BaseModel):
    length: float = Field(
        default=0.1, gt=0, le=10, description="Plate length in meters"
    )
    width: float = Field(
        default=0.05, gt=0, le=10, description="Plate width in meters"
    )
    heat_power: float = Field(
        default=100000, gt=0, le=10_000_000, description="Heat-source term"
    )
    ambient_temperature: float = Field(
        default=25.0,
        ge=-100,
        le=500,
        description="Boundary temperature in Celsius",
    )


class PointInput(BaseModel):
    x: float = Field(
        ..., ge=0, le=1, description="Normalized x coordinate between 0 and 1"
    )
    y: float = Field(
        ..., ge=0, le=1, description="Normalized y coordinate between 0 and 1"
    )


class BatchPointInput(BaseModel):
    points: list[PointInput] = Field(..., min_length=1, max_length=500)


class GridInput(BaseModel):
    grid_size: int = Field(default=20, ge=5, le=100)
