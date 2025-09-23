import streamlit as st
import pickle
import json
import numpy as np

# --- Load model, scaler, and feature columns ---
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# --- Streamlit App UI ---
st.set_page_config(page_title="Manufacturing Output Predictor")
st.title("📦 Manufacturing Equipment Output Prediction")

with st.form("prediction_form"):
    st.subheader("Enter Machine Parameters")

    Injection_Temperature = st.number_input("Injection Temperature (°C)", 180.0, 250.0)
    Injection_Pressure = st.number_input("Injection Pressure (bar)", 80.0, 150.0)
    Cycle_Time = st.number_input("Cycle Time (s)", 15.0, 45.0)
    Cooling_Time = st.number_input("Cooling Time (s)", 8.0, 20.0)
    Material_Viscosity = st.number_input("Material Viscosity (Pa·s)", 100.0, 400.0)
    Ambient_Temperature = st.number_input("Ambient Temperature (°C)", 18.0, 28.0)
    Machine_Age = st.number_input("Machine Age (years)", 1.0, 15.0)
    Operator_Experience = st.number_input("Operator Experience (months)", 1.0, 120.0)
    Maintenance_Hours = st.number_input("Maintenance Hours Since Last Service", 0.0, 200.0)

    submitted = st.form_submit_button("Predict Output")

# --- Prediction Logic ---
if submitted:
    try:
        # Base input
        input_data = {
            "Injection_Temperature": Injection_Temperature,
            "Injection_Pressure": Injection_Pressure,
            "Cycle_Time": Cycle_Time,
            "Cooling_Time": Cooling_Time,
            "Material_Viscosity": Material_Viscosity,
            "Ambient_Temperature": Ambient_Temperature,
            "Machine_Age": Machine_Age,
            "Operator_Experience": Operator_Experience,
            "Maintenance_Hours": Maintenance_Hours
        }

        # Compute derived features
        input_data["Temperature_Pressure_Ratio"] = Injection_Temperature / Injection_Pressure
        input_data["Total_Cycle_Time"] = Cycle_Time + Cooling_Time
        input_data["Efficiency_Score"] = Operator_Experience / (Cycle_Time + 1)
        input_data["Machine_Utilization"] = 1 - (Maintenance_Hours / 200)

        # Match feature order
        input_list = [input_data[col] for col in feature_columns]

        # Scale and predict
        scaled_input = scaler.transform([input_list])
        prediction = model.predict(scaled_input)

        st.success(f"✅ Predicted Output: **{round(float(prediction[0]), 2)} parts/hour**")

    except Exception as e:
        st.error(f"🚨 Prediction Error: {e}")
