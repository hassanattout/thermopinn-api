import os

from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "ThermoPINN API"
    app_version: str = "1.1.0"
    environment: str = "production"
    model_path: str = "models/pinn_model.pth"
    results_dir: str = "results"
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "THERMOPINN_CORS_ORIGINS",
                "http://localhost:8501",
            ).split(",")
            if origin.strip()
        ]
    )


settings = Settings()
