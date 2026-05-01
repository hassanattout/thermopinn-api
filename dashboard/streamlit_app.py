import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="ThermoPINN",
    page_icon="🌡️",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero {
    padding: 1.5rem;
    border-radius: 1rem;
    background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
    color: white;
    margin-bottom: 1.5rem;
}
.hero h1 {margin-bottom: 0rem;}
.hero p {color: #d1d5db;}
</style>
""", unsafe_allow_html=True)

API_URL = st.sidebar.text_input(
    "API URL",
    "https://thermopinn-api.onrender.com"
).rstrip("/")

st.sidebar.caption("Use Render URL for production or http://127.0.0.1:8000 locally.")

st.markdown("""
<div class="hero">
<h1>ThermoPINN</h1>
<p>Physics-informed thermal simulation platform: FEM solver + PINN surrogate + real-time API.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Predict",
    "Thermal Field",
    "Benchmark"
])


def api_post(endpoint, payload, timeout=120):
    try:
        response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=timeout)
        if response.status_code != 200:
            st.error(response.text)
            return None
        return response.json()
    except Exception as e:
        st.error(str(e))
        return None


with tab1:
    st.subheader("What this system does")

    col1, col2, col3 = st.columns(3)
    col1.metric("Backend", "FastAPI")
    col2.metric("Physics", "FEM Solver")
    col3.metric("ML", "PINN Surrogate")

    st.info(
        "ThermoPINN compares a physics-based numerical solver against a learned neural surrogate. "
        "The goal is faster thermal prediction for engineering workflows such as optimization, monitoring, and digital twins."
    )

    if st.button("Check Live API"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=60)
            st.json(r.json())
        except Exception as e:
            st.error(str(e))


with tab2:
    st.subheader("Single Point Temperature Prediction")

    c1, c2 = st.columns(2)
    x = c1.slider("x coordinate", 0.0, 1.0, 0.5, 0.01)
    y = c2.slider("y coordinate", 0.0, 1.0, 0.5, 0.01)

    if st.button("Run Prediction", use_container_width=True):
        with st.spinner("Running PINN inference..."):
            data = api_post("/predict-pinn", {"x": x, "y": y})

        if data:
            c1, c2, c3 = st.columns(3)
            c1.metric("x", data["x"])
            c2.metric("y", data["y"])
            c3.metric("Temperature", f"{data['predicted_temperature']:.2f} °C")
            st.caption(f"Inference time: {data['inference_time_seconds']} s")


with tab3:
    st.subheader("Thermal Field Heatmap")

    grid_size = st.slider("Grid size", 5, 75, 25)

    if st.button("Generate Thermal Field", use_container_width=True):
        with st.spinner("Generating thermal field..."):
            data = api_post("/predict-grid", {"grid_size": grid_size})

        if data:
            grid = np.array(data["temperature_grid"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Grid", data["grid_size"])
            c2.metric("Min", f"{data['min_temperature']:.2f} °C")
            c3.metric("Max", f"{data['max_temperature']:.2f} °C")
            c4.metric("Mean", f"{data['mean_temperature']:.2f} °C")

            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(grid, origin="lower", aspect="auto")
            ax.set_title("Predicted Thermal Field")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, label="Temperature (°C)")
            st.pyplot(fig)

            with st.expander("Show raw grid"):
                st.dataframe(pd.DataFrame(grid), use_container_width=True)


with tab4:
    st.subheader("FEM Solver vs PINN Surrogate")

    c1, c2 = st.columns(2)
    length = c1.number_input("Length (m)", min_value=0.001, value=0.1)
    width = c2.number_input("Width (m)", min_value=0.001, value=0.05)

    c3, c4 = st.columns(2)
    heat_power = c3.number_input("Heat Power", min_value=1.0, value=100000.0)
    ambient_temperature = c4.number_input("Ambient Temperature", value=25.0)

    if st.button("Run Benchmark", use_container_width=True):
        with st.spinner("Running FEM vs PINN comparison..."):
            data = api_post(
                "/compare",
                {
                    "length": length,
                    "width": width,
                    "heat_power": heat_power,
                    "ambient_temperature": ambient_temperature,
                },
            )

        if data:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", data["error_metrics"]["mae"])
            c2.metric("RMSE", data["error_metrics"]["rmse"])
            c3.metric("Max Error", data["error_metrics"]["max_error"])
            c4.metric("Speedup", data["speedup"])

            df = pd.DataFrame([
                {
                    "System": "FEM Solver",
                    "Min Temp": data["fem"]["min_temperature"],
                    "Max Temp": data["fem"]["max_temperature"],
                    "Mean Temp": data["fem"]["mean_temperature"],
                    "Runtime": data["fem"]["time_seconds"],
                },
                {
                    "System": "PINN Surrogate",
                    "Min Temp": data["pinn"]["min_temperature"],
                    "Max Temp": data["pinn"]["max_temperature"],
                    "Mean Temp": data["pinn"]["mean_temperature"],
                    "Runtime": data["pinn"]["time_seconds"],
                },
            ])

            st.dataframe(df, use_container_width=True)

            st.subheader("Runtime Comparison")
            st.bar_chart(df.set_index("System")["Runtime"])

            st.info(
                "The FEM solver is the physics reference. The PINN is the learned surrogate designed for faster inference."
            )
