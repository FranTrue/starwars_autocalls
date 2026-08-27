"""FastAPI service that serves duration predictions."""
import joblib
import pandas as pd
from fastapi import FastAPI

from starwars_autocalls.api.schemas import DurationPrediction, RFQFeatures
from starwars_autocalls.config import MODEL_DIR

MODEL_PATH = MODEL_DIR / "model.joblib"

app = FastAPI(title="Star Wars Autocalls -- Duration Predictor")

_pipeline = joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/predict", response_model=DurationPrediction)
def predict(features: RFQFeatures) -> DurationPrediction:
    """Predicts avg_duration_months for one RFQ."""
    X = pd.DataFrame([features.model_dump()])
    prediction = _pipeline.predict(X)[0]
    return DurationPrediction(predicted_avg_duration_months=round(float(prediction), 2))
