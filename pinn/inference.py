import torch

from pinn.model import PINN


def load_model(model_path="models/pinn_model.pth"):
    model = PINN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def predict_temperature(model, x: float, y: float) -> float:
    with torch.no_grad():
        inputs = torch.tensor([[x, y]], dtype=torch.float32)
        prediction = model(inputs)
        return float(prediction.item())


if __name__ == "__main__":
    model = load_model()
    temperature = predict_temperature(model, 0.5, 0.5)

    print(f"Predicted normalized temperature at (0.5, 0.5): {temperature:.4f}")
