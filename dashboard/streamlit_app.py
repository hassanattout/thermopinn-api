import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="ThermoPINN Dashboard",
    page_icon="🌡️",
    layout="wide",
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0rem;
}

.subtitle {
    color: #666;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

.card {
    padding: 1.2rem;
    border-radius: 1rem;
    border: 1px solid #e6e6e6;
    background-color: #fafafa;
    margin-bottom: 1rem;
}

.small-note {
    color: #777;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


def call_api(method, url, payload=None, timeout=60):
    try:
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        else:
            response = requests.post(url, json=payload, timeout=timeout)

        if response.status_code >= 400:
            return None, f"{response.status_code}: {response.text}"

        return response.json(), None

    except requests.exceptions.Timeout:
        return None, "Request timed out. Render free tier may be waking up. Try again."
    except requests.exceptions.ConnectionError:
        return None, "Connection error. Check that the API URL is correct and the backend is running."
    except Exception as e:
        return None, str(e)


st.markdown('<div class="main-title">ThermoPINN Engineering Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Physics-informed thermal simulation platform • FEM solver • PINN surrogate • FastAPI backend</div>',
    unsafe_allow_html=True,
)

st.sidebar.title("ThermoPINN Controls")

API_URL = st.sidebar.text_input(
    "API URL",
    "https://thermopinn-api.onrender.com",
    help="Use the Render URL for production or http://127.0.0.1:8000 for local testing.",
).rstrip("/")

st.sidebar.caption("Render free tier may sleep. First request can take 10-30 seconds.")

endpoint = st.sidebar.radio(
    "Select module",
    [
        "System Health",
        "Single Prediction",
        "Batch Prediction",
        "Grid Prediction",
        "FEM Simulation",
        "FEM vs PINN Benchmark",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("### System Positioning")
st.sidebar.write(
    "This dashboard demonstrates a real-time engineering ML backend that compares numerical physics simulation against a learned neural surrogate."
)


if endpoint == "System Health":
    st.header("System Health")

    st.markdown('<div class="card">Check whether the FastAPI backend and PINN model are available.</div>', unsafe_allow_html=True)

    if st.button("Check API Health", use_container_width=True):
        with st.spinner("Checking backend status..."):
            data, error = call_api("GET", f"{API_URL}/health")

        if error:
            st.error(error)
        else:
            col1, col2, col3 = st.columns(3)

            col1.metric("API Status", data.get("status", "unknown"))
            col2.metric("Model Status", data.get("model_status", "unknown"))
            col3.metric("Model Exists", str(data.get("model_exists", "unknown")))

            st.subheader("Raw Response")
            st.json(data)


elif endpoint == "Single Prediction":
    st.header("Single Point PINN Prediction")

    st.write("Predict temperature at one normalized coordinate using the trained PINN surrogate.")

    col1, col2 = st.columns(2)

    with col1:
        x = st.slider("x coordinate", 0.0, 1.0, 0.5, 0.01)

    with col2:
        y = st.slider("y coordinate", 0.0, 1.0, 0.5, 0.01)

    if st.button("Predict Temperature", use_container_width=True):
        with st.spinner("Running PINN inference..."):
            payload = {"x": x, "y": y}
            data, error = call_api("POST", f"{API_URL}/predict-pinn", payload)

        if error:
            st.error(error)
        else:
            col1, col2, col3 = st.columns(3)

            col1.metric("x", data["x"])
            col2.metric("y", data["y"])
            col3.metric("Temperature", f"{data['predicted_temperature']:.2f} °C")

            st.metric("Inference Time", f"{data['inference_time_seconds']} s")

            st.subheader("Raw Response")
            st.json(data)


elif endpoint == "Batch Prediction":
    st.header("Batch PINN Prediction")

    st.write("Run multiple point predictions in one API request.")

    default_points = pd.DataFrame(
        [
            {"x": 0.1, "y": 0.1},
            {"x": 0.3, "y": 0.7},
            {"x": 0.5, "y": 0.5},
            {"x": 0.8, "y": 0.2},
            {"x": 0.9, "y": 0.9},
        ]
    )

    edited_df = st.data_editor(
        default_points,
        num_rows="dynamic",
        use_container_width=True,
    )

    if st.button("Run Batch Prediction", use_container_width=True):
        points = edited_df.to_dict(orient="records")

        with st.spinner("Running batch inference..."):
            data, error = call_api("POST", f"{API_URL}/predict-batch", {"points": points})

        if error:
            st.error(error)
        else:
            predictions = pd.DataFrame(data["predictions"])

            st.metric("Number of Points", data["number_of_points"])
            st.metric("Total Inference Time", f"{data['total_inference_time_seconds']} s")

            st.subheader("Predictions")
            st.dataframe(predictions, use_container_width=True)

            st.subheader("Temperature by Point")
            chart_df = predictions.copy()
            chart_df["point"] = chart_df.index.astype(str)
            st.bar_chart(chart_df.set_index("point")["predicted_temperature"])

            st.subheader("Raw Response")
            st.json(data)


elif endpoint == "Grid Prediction":
    st.header("Full Thermal Field Prediction")

    st.write("Generate a thermal field using PINN inference over a 2D grid.")

    grid_size = st.slider("Grid size", 5, 75, 20)

    if st.button("Run Grid Prediction", use_container_width=True):
        with st.spinner("Generating thermal field..."):
            data, error = call_api("POST", f"{API_URL}/predict-grid", {"grid_size": grid_size})

        if error:
            st.error(error)
        else:
            grid = np.array(data["temperature_grid"])
            df = pd.DataFrame(grid)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Grid Size", data["grid_size"])
            col2.metric("Min Temp", f"{data['min_temperature']:.2f} °C")
            col3.metric("Max Temp", f"{data['max_temperature']:.2f} °C")
            col4.metric("Mean Temp", f"{data['mean_temperature']:.2f} °C")

            st.metric("Inference Time", f"{data['inference_time_seconds']} s")

            st.subheader("Thermal Field Heatmap")
            st.write("Brighter zones represent higher predicted temperature.")
            st.dataframe(
                df.style.background_gradient(axis=None),
                use_container_width=True,
            )

            st.subheader("Temperature Profile")
            st.line_chart(df.mean(axis=0))

            with st.expander("Show raw thermal grid"):
                st.dataframe(df, use_container_width=True)


elif endpoint == "FEM Simulation":
    st.header("FEM Thermal Simulation")

    st.write("Run the numerical physics solver and generate a thermal map.")

    col1, col2 = st.columns(2)

    with col1:
        length = st.number_input("Length (m)", min_value=0.001, value=0.1)
        width = st.number_input("Width (m)", min_value=0.001, value=0.05)

    with col2:
        heat_power = st.number_input("Heat Power", min_value=1.0, value=100000.0)
        ambient_temperature = st.number_input("Ambient Temperature (°C)", value=25.0)

    payload = {
        "length": length,
        "width": width,
        "heat_power": heat_power,
        "ambient_temperature": ambient_temperature,
    }

    if st.button("Run FEM Simulation", use_container_width=True):
        with st.spinner("Running FEM solver..."):
            data, error = call_api("POST", f"{API_URL}/simulate", payload)

        if error:
            st.error(error)
        else:
            col1, col2, col3 = st.columns(3)

            col1.metric("Min Temp", f"{data['min_temperature']:.2f} °C")
            col2.metric("Max Temp", f"{data['max_temperature']:.2f} °C")
            col3.metric("Mean Temp", f"{data['mean_temperature']:.2f} °C")

            st.metric("Solver Time", f"{data['solver_time_seconds']} s")

            st.subheader("Generated Thermal Map")
            st.write("Open this endpoint after simulation:")
            st.code(f"{API_URL}/thermal-map")

            st.subheader("Raw Response")
            st.json(data)


elif endpoint == "FEM vs PINN Benchmark":
    st.header("FEM Solver vs PINN Surrogate Benchmark")

    st.write(
        "This is the core validation layer: compare the numerical solver against the learned PINN surrogate."
    )

    col1, col2 = st.columns(2)

    with col1:
        length = st.number_input("Length (m)", min_value=0.001, value=0.1)
        width = st.number_input("Width (m)", min_value=0.001, value=0.05)

    with col2:
        heat_power = st.number_input("Heat Power", min_value=1.0, value=100000.0)
        ambient_temperature = st.number_input("Ambient Temperature (°C)", value=25.0)

    payload = {
        "length": length,
        "width": width,
        "heat_power": heat_power,
        "ambient_temperature": ambient_temperature,
    }

    if st.button("Compare FEM vs PINN", use_container_width=True):
        with st.spinner("Running benchmark comparison..."):
            data, error = call_api("POST", f"{API_URL}/compare", payload, timeout=120)

        if error:
            st.error(error)
        else:
            st.subheader("Error Metrics")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", data["error_metrics"]["mae"])
            col2.metric("RMSE", data["error_metrics"]["rmse"])
            col3.metric("Max Error", data["error_metrics"]["max_error"])
            col4.metric("Speedup", data["speedup"])

            st.subheader("Temperature Summary")

            comparison_df = pd.DataFrame(
                [
                    {
                        "System": "FEM Solver",
                        "Min Temperature": data["fem"]["min_temperature"],
                        "Max Temperature": data["fem"]["max_temperature"],
                        "Mean Temperature": data["fem"]["mean_temperature"],
                        "Time (s)": data["fem"]["time_seconds"],
                    },
                    {
                        "System": "PINN Surrogate",
                        "Min Temperature": data["pinn"]["min_temperature"],
                        "Max Temperature": data["pinn"]["max_temperature"],
                        "Mean Temperature": data["pinn"]["mean_temperature"],
                        "Time (s)": data["pinn"]["time_seconds"],
                    },
                ]
            )

            st.dataframe(comparison_df, use_container_width=True)

            st.subheader("Runtime Comparison")
            st.bar_chart(comparison_df.set_index("System")["Time (s)"])

            st.subheader("Mean Temperature Comparison")
            st.bar_chart(comparison_df.set_index("System")["Mean Temperature"])

            st.subheader("Interpretation")
            st.info(
                "The FEM solver acts as the physics-based reference. The PINN surrogate approximates the thermal field and can be used for faster inference, optimization loops, and real-time digital twin workflows."
            )

            with st.expander("Show raw benchmark response"):
                st.json(data)
