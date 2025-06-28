from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow:5000")

# Define request and response schemas
class CustomerInput(BaseModel):
    recency: float
    frequency: float
    monetary: float

class PredictionResponse(BaseModel):
    probability: float
    label: int

# Initialize FastAPI
app = FastAPI(title="Credit Risk Prediction API")

# Load the model from MLflow registry, specifying version 3
MODEL_NAME = "credit_risk_model"
MODEL_VERSION = 3  # Specify the version directly

try:
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}")
except mlflow.exceptions.MlflowException as e:
    raise RuntimeError(f"Failed to load model '{MODEL_NAME}' version '{MODEL_VERSION}': {e}")

@app.get("/")
def root():
    return {"message": "Credit Risk Prediction API is live!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: CustomerInput):
    input_df = pd.DataFrame([input_data.dict()])
    probabilities = model.predict(input_df)

    if hasattr(probabilities, "tolist"):
        probabilities = probabilities.tolist()
    elif isinstance(probabilities, (float, int)):
        probabilities = [[probabilities]]

    # Assuming binary classification and probability output
    risk_prob = probabilities[0][1] if isinstance(probabilities[0], list) else probabilities[0]
    label = int(risk_prob >= 0.5)

    return PredictionResponse(probability=risk_prob, label=label)