import streamlit as st
import requests
import pandas as pd

API_URL = "https://thermopinn-api.onrender.com"

st.set_page_config(
    page_title="ThermoPINN Dashboard",
    page_icon="🌡️",
    layout="wide",
)

st.title("ThermoPINN Engineering Dashboard")
st.write("Physics-informed thermal prediction system powered by FastAPI and PINN inference.")

st.sidebar.header("Controls")

endpoint = st.sidebar.selectbox(
    "Choose action",
    ["Health Check", "Single Prediction", "Batch Prediction", "Grid Prediction", "FEM vs PINN Comparison"],
)

if endpoint == "Health Check":
    st.header("API Health Check")

    if st.button("Check API"):
        response = requests.get(f"{API_URL}/health")
        st.json(response.json())

elif endpoint == "Single Prediction":
    st.header("Single Point Temperature Prediction")

    col1, col2 = st.columns(2)

    with col1:
        x = st.slider("x coordinate", 0.0, 1.0, 0.5, 0.01)

    with col2:
        y = st.slider("y coordinate", 0.0, 1.0, 0.5, 0.01)

    if st.button("Predict Temperature"):
        payload = {"x": x, "y": y}
        response = requests.post(f"{API_URL}/predict-pinn", json=payload)

        if response.status_code == 200:
            data = response.json()
            st.metric("Predicted Temperature", f"{data['predicted_temperature']:.2f} °C")
            st.metric("Inference Time", f"{data['inference_time_seconds']} s")
            st.json(data)
        else:
            st.error(response.text)

elif endpoint == "Batch Prediction":
    st.header("Batch Temperature Prediction")

    points = [
        {"x": 0.1, "y": 0.1},
        {"x": 0.5, "y": 0.5},
        {"x": 0.9, "y": 0.9},
    ]

    st.write("Default batch points:")
    st.json(points)

    if st.button("Run Batch Prediction"):
        response = requests.post(f"{API_URL}/predict-batch", json={"points": points})

        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data["predictions"])
            st.dataframe(df, use_container_width=True)
            st.json(data)
        else:
            st.error(response.text)

elif endpoint == "Grid Prediction":
    st.header("Full Thermal Field Prediction")

    grid_size = st.slider("Grid size", 5, 50, 10)

    if st.button("Run Grid Prediction"):
        response = requests.post(f"{API_URL}/predict-grid", json={"grid_size": grid_size})

        if response.status_code == 200:
            data = response.json()

            col1, col2, col3 = st.columns(3)
            col1.metric("Min Temperature", f"{data['min_temperature']:.2f} °C")
            col2.metric("Max Temperature", f"{data['max_temperature']:.2f} °C")
            col3.metric("Mean Temperature", f"{data['mean_temperature']:.2f} °C")

            df = pd.DataFrame(data["temperature_grid"])
            st.write("Temperature field:")
            st.dataframe(df, use_container_width=True)
            st.line_chart(df)
        else:
            st.error(response.text)

elif endpoint == "FEM vs PINN Comparison":
    st.header("FEM Solver vs PINN Surrogate Benchmark")

    col1, col2 = st.columns(2)

    with col1:
        length = st.number_input("Length", value=0.1)
        width = st.number_input("Width", value=0.05)

    with col2:
        heat_power = st.number_input("Heat power", value=100000)
        ambient_temperature = st.number_input("Ambient temperature", value=25.0)

    if st.button("Compare FEM vs PINN"):
        payload = {
            "length": length,
            "width": width,
            "heat_power": heat_power,
            "ambient_temperature": ambient_temperature,
        }

        response = requests.post(f"{API_URL}/compare", json=payload)

        if response.status_code == 200:
            data = response.json()

            st.subheader("Benchmark Summary")

            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", data["error_metrics"]["mae"])
            col2.metric("RMSE", data["error_metrics"]["rmse"])
            col3.metric("Max Error", data["error_metrics"]["max_error"])

            st.metric("Speedup", data["speedup"])

            st.subheader("Full Response")
            st.json(data)
        else:
            st.error(response.text)
