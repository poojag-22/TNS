# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Load model and scaler
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.json", "r") as f:
    import json
    feature_columns = json.load(f)

# Define request schema
class InputData(BaseModel):
    Injection_Temperature: float
    Injection_Pressure: float
    Cycle_Time: float
    Cooling_Time: float
    Material_Viscosity: float
    Ambient_Temperature: float
    Machine_Age: float
    Operator_Experience: float
    Maintenance_Hours: float
    Temperature_Pressure_Ratio:float
    Total_Cycle_Time:float
    Efficiency_Score:float
    Machine_Utilization:float
        

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Manufacturing Output Prediction API!"}

@app.post("/predict")
def predict(data: InputData):
    try:
        input_dict = data.dict()
        input_df = [input_dict[col] for col in feature_columns]
        scaled_input = scaler.transform([input_df])
        prediction = model.predict(scaled_input)
        return {"Predicted_Output_Parts_Per_Hour": round(float(prediction[0]), 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "Model is loaded and API is healthy"}
