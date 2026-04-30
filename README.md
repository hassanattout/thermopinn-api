# ThermoPINN API

Physics-informed neural network for real-time thermal prediction in engineering systems.

---

## Overview

Thermal management is critical in modern systems such as:

- Data centers
- Electric vehicles
- Batteries
- Electronics

Traditional solvers (FEM) are accurate but slow.

This project builds a Physics-Informed Neural Network (PINN) that:

- Learns the heat equation
- Predicts temperature fields
- Runs faster than numerical solvers
- Is deployable via API

---

## Tech Stack

- PyTorch
- FastAPI
- NumPy
- Matplotlib

---

## Features

- 2D steady-state heat equation solver
- Thermal map visualization
- PINN training with supervised + normalized learning
- API endpoints for simulation and prediction
- Benchmark comparison (PINN vs solver)

---

## API Endpoints

### Run physics simulation

POST /simulate

{
  "length": 0.1,
  "width": 0.05,
  "heat_power": 100000,
  "ambient_temperature": 25
}

### Predict with PINN

POST /predict-pinn

{
  "x": 0.5,
  "y": 0.5
}

---

## Results

- MAE: 0.18 °C
- RMSE: 0.26 °C
- Max Error: 2.32 °C

---

## Performance

- Solver time: ~0.13 s
- PINN inference: ~0.036 s

---

## Visualizations

### Solver thermal map
![Solver Thermal Map](results/thermal_map.png)

### PINN thermal map
![PINN Thermal Map](results/pinn_thermal_map.png)

### Error metrics
![Error Metrics](results/error_metrics.png)

---

## Author

Hassan Attout  
Mechanical & Energy Engineering | AI for Engineering Systems
