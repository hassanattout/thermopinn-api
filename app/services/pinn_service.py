import time
import numpy as np
from pathlib import Path
from fastapi import HTTPException

from pinn.inference import load_model, predict_temperature


MODEL_PATH = Path("models/pinn_model.pth")


class PINNService:
    def __init__(self):
        self.model = None
        self.status = "not_loaded"
        self.load()

    def load(self):
        try:
            self.model = load_model()
            self.status = "loaded"
        except Exception as e:
            self.model = None
            self.status = f"error: {str(e)}"

    def ensure_loaded(self):
        if self.model is None:
            raise HTTPException(
                status_code=500,
                detail=f"PINN model not loaded. Status: {self.status}",
            )

    def predict_point(self, x: float, y: float):
        self.ensure_loaded()

        start = time.perf_counter()
        temperature = predict_temperature(self.model, x, y)
        inference_time = time.perf_counter() - start

        return {
            "x": x,
            "y": y,
            "predicted_temperature": float(temperature),
            "inference_time_seconds": round(inference_time, 6),
        }

    def predict_batch(self, points):
        self.ensure_loaded()

        start = time.perf_counter()
        predictions = []

        for point in points:
            temp = predict_temperature(self.model, point.x, point.y)
            predictions.append({
                "x": point.x,
                "y": point.y,
                "predicted_temperature": float(temp),
            })

        inference_time = time.perf_counter() - start

        return {
            "message": "Batch PINN prediction completed",
            "number_of_points": len(points),
            "total_inference_time_seconds": round(inference_time, 6),
            "predictions": predictions,
        }

    def predict_grid(self, grid_size: int):
        self.ensure_loaded()

        start = time.perf_counter()

        xs = np.linspace(0, 1, grid_size)
        ys = np.linspace(0, 1, grid_size)

        temperature_grid = []

        for y in ys:
            row = []
            for x in xs:
                temp = predict_temperature(self.model, float(x), float(y))
                row.append(float(temp))
            temperature_grid.append(row)

        inference_time = time.perf_counter() - start
        arr = np.array(temperature_grid)

        return {
            "message": "PINN grid prediction completed",
            "grid_size": grid_size,
            "min_temperature": float(np.min(arr)),
            "max_temperature": float(np.max(arr)),
            "mean_temperature": float(np.mean(arr)),
            "inference_time_seconds": round(inference_time, 6),
            "temperature_grid": temperature_grid,
        }


pinn_service = PINNService()
