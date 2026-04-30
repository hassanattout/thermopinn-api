import numpy as np
import matplotlib.pyplot as plt
import torch

from pinn.inference import load_model


def generate_pinn_map(output_path="results/pinn_thermal_map.png"):
    model = load_model()

    nx, ny = 50, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)

    X, Y = np.meshgrid(x, y)

    coords = np.stack([X.flatten(), Y.flatten()], axis=1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32)

    with torch.no_grad():
        T = model(coords_tensor).numpy()

    T = T.reshape((ny, nx))

    plt.figure()
    contour = plt.contourf(X, Y, T, levels=50)
    plt.colorbar(contour, label="Temperature (°C)")
    plt.title("PINN Thermal Map")

    plt.savefig(output_path)
    plt.close()

    print(f"PINN thermal map saved to {output_path}")


if __name__ == "__main__":
    generate_pinn_map()
