import torch
import numpy as np
from pinn.model import PINN
from pinn.loss import physics_loss


def train_pinn(epochs=2000):
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # sample domain points
        x = torch.rand((1000, 1))
        y = torch.rand((1000, 1))

        loss = physics_loss(model, x, y)

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "models/pinn_model.pth")
    print("Model saved.")


if __name__ == "__main__":
    train_pinn()
