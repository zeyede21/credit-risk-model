# Placeholder for pydantic_models.py
# src/api/pydantic_models.py
from pydantic import BaseModel
from typing import List

class CustomerFeatures(BaseModel):
    # Replace these example fields with your actual model features
    feature1: float
    feature2: float
    feature3: float
    # ... add all required fields from preprocess_dataset.csv

class PredictionResponse(BaseModel):
    probability: float
    risk_class: str
