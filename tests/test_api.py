from fastapi.testclient import TestClient

from app.main import app
from app.services.pinn_service import pinn_service

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_and_openapi_use_the_same_version():
    root = client.get("/").json()
    schema = client.get("/openapi.json").json()
    assert root["version"] == schema["info"]["version"]


def test_predict_pinn():
    response = client.post(
        "/predict-pinn",
        json={"x": 0.5, "y": 0.5},
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_temperature" in data
    assert isinstance(data["predicted_temperature"], float)


def test_predict_batch():
    response = client.post(
        "/predict-batch",
        json={
            "points": [
                {"x": 0.1, "y": 0.1},
                {"x": 0.5, "y": 0.5},
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["number_of_points"] == 2
    assert "predictions" in data


def test_predict_grid():
    response = client.post(
        "/predict-grid",
        json={"grid_size": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["grid_size"] == 5
    assert "temperature_grid" in data


def test_rejects_out_of_range_point():
    response = client.post(
        "/predict-pinn",
        json={"x": 1.1, "y": 0.5},
    )
    assert response.status_code == 422


def test_rejects_empty_and_oversized_batches():
    assert client.post("/predict-batch", json={"points": []}).status_code == 422

    points = [{"x": 0.5, "y": 0.5} for _ in range(501)]
    assert client.post("/predict-batch", json={"points": points}).status_code == 422


def test_rejects_unbounded_simulation_parameters():
    response = client.post(
        "/simulate",
        json={
            "length": 100,
            "width": 0.05,
            "heat_power": 100000,
            "ambient_temperature": 25,
        },
    )
    assert response.status_code == 422


def test_model_failure_is_sanitized(monkeypatch):
    monkeypatch.setattr(pinn_service, "model", None)
    monkeypatch.setattr(pinn_service, "status", "error: /private/model/path")

    response = client.post(
        "/predict-pinn",
        json={"x": 0.5, "y": 0.5},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Thermal surrogate model is unavailable."
    assert "private" not in response.text
