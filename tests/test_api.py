from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


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
