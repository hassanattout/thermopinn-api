from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "ThermoPINN API"
    app_version: str = "1.1.0"
    environment: str = "production"
    model_path: str = "models/pinn_model.pth"
    results_dir: str = "results"


settings = Settings()
