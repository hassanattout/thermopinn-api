import numpy as np


def solve_steady_state_heat_2d(
    nx: int = 50,
    ny: int = 50,
    length: float = 0.1,
    width: float = 0.05,
    heat_source: float = 1000.0,
    boundary_temperature: float = 25.0,
    max_iterations: int = 5000,
    tolerance: float = 1e-5,
):
    """
    Solves a simplified 2D steady-state heat equation.

    Equation:
        ∇²T + Q = 0

    Boundary condition:
        Fixed temperature on all edges.

    Returns:
        x_grid, y_grid, temperature field
    """

    x = np.linspace(0, length, nx)
    y = np.linspace(0, width, ny)

    temperature = np.ones((ny, nx)) * boundary_temperature

    dx = length / (nx - 1)
    dy = width / (ny - 1)

    source = heat_source * dx * dy

    for _ in range(max_iterations):
        old_temperature = temperature.copy()

        temperature[1:-1, 1:-1] = 0.25 * (
            old_temperature[1:-1, 2:]
            + old_temperature[1:-1, :-2]
            + old_temperature[2:, 1:-1]
            + old_temperature[:-2, 1:-1]
            + source
        )

        error = np.max(np.abs(temperature - old_temperature))

        if error < tolerance:
            break

    x_grid, y_grid = np.meshgrid(x, y)

    return x_grid, y_grid, temperature


if __name__ == "__main__":
    x_grid, y_grid, temperature = solve_steady_state_heat_2d()

    print("2D heat solver completed.")
    print(f"Grid shape: {temperature.shape}")
    print(f"Minimum temperature: {temperature.min():.2f} °C")
    print(f"Maximum temperature: {temperature.max():.2f} °C")
