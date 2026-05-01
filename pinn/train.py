import numpy as np
import torch

from fem.heat_solver_2d import solve_steady_state_heat_2d
from pinn.model import PINN


def train_pinn(epochs=5000):
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = torch.nn.MSELoss()

    x_grid, y_grid, T_solver = solve_steady_state_heat_2d(
        nx=50,
        ny=50,
        length=0.1,
        width=0.05,
        heat_source=100000,
        boundary_temperature=25.0,
    )

    x_norm = x_grid / 0.1
    y_norm = y_grid / 0.05

    T_min = T_solver.min()
    T_max = T_solver.max()
    T_norm = (T_solver - T_min) / (T_max - T_min)

    coords = np.stack([x_norm.flatten(), y_norm.flatten()], axis=1)
    temps = T_norm.flatten().reshape(-1, 1)

    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    temps_tensor = torch.tensor(temps, dtype=torch.float32)

    for epoch in range(epochs):
        optimizer.zero_grad()

        indices = torch.randperm(coords_tensor.shape[0])[:1000]
        batch_coords = coords_tensor[indices]
        batch_temps = temps_tensor[indices]

        predictions = model(batch_coords)
        loss = mse_loss(predictions, batch_temps)

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.8f}")

    torch.save(model.state_dict(), "models/pinn_model.pth")
    print("Model saved.")


if __name__ == "__main__":
    train_pinn()
