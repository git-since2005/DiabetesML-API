from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Diabetes Prediction API")

# ---- Load model ONCE at startup ----
model = joblib.load("diabetes_predictor.pkl")

# ---- Input schema (adjust fields to your model) ----
class DiabetesInput(BaseModel):
    glucose: float
    blood_pressure: float
    bmi: float
    insulin: float
    pedigree: float
    age: int


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: DiabetesInput):
    # Convert input to model format
    features = np.array([[
        data.glucose,
        data.blood_pressure,
        data.insulin,
        data.bmi,
        data.pedigree,
        data.age
    ]])

    prediction = model.predict(features)[0]

    return {
        "prediction": int(prediction),
        "label": "Diabetic" if prediction == 1 else "Non-Diabetic"
    }
