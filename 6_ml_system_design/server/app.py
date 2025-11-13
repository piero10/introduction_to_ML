from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import List


# --------------------------------------------------
# Load trained model
# --------------------------------------------------
model = joblib.load("titanic_model.pkl")

# --------------------------------------------------
# Схема
# --------------------------------------------------
class Passenger(BaseModel):
    age: float = Field(..., description="Passenger age in years")
    fare: float = Field(..., description="Ticket fare")
    sex: str = Field(..., description="Passenger sex (male/female)")
    pclass: str = Field(..., description="Passenger class (First/Second/Third)")
    embarked: str = Field(..., description="Port of embarkation (C, Q, S)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 29,
                    "fare": 35.5,
                    "sex": "female",
                    "pclass": "Second",
                    "embarked": "S"
                }
            ]
        }
    }
app = FastAPI(
    title="🚢 Titanic Survival Prediction API",
    description="Predicts whether a passenger survived the Titanic disaster using a trained ML model.",
    version="1.0.0",
)

# --------------------------------------------------
# Root endpoint
# --------------------------------------------------
@app.get("/", tags=["Root"])
def home():
    """
    Simple health check endpoint.
    """
    return {"message": "Titanic Survival Prediction API is running ✅"}

# --------------------------------------------------
# Predict endpoint
# --------------------------------------------------
@app.get("/predict", tags=["Prediction"])
def predict(passenger: Passenger):
    """
    Predict whether a passenger survived.
    """
    data = pd.DataFrame([{
        "age": passenger.age,
        "fare": passenger.fare,
        "sex": passenger.sex,
        "class": passenger.pclass,
        "embarked": passenger.embarked
    }])

    prediction = model.predict(data)[0]
    survival_prob = (
        model.predict_proba(data)[0][1]
        if hasattr(model, "predict_proba")
        else None
    )

    return {
        "survived": bool(prediction),
        "probability_of_survival": float(survival_prob) if survival_prob else None
    }

@app.get("/predict_simple")
def predict_simple(
    age: float,
    fare: float,
    sex: str,
    pclass: str,
    embarked: str
):
    data = pd.DataFrame([{
        "age": age,
        "fare": fare,
        "sex": sex,
        "class": pclass,
        "embarked": embarked
    }])
    prediction = model.predict(data)[0]
    survival_prob = model.predict_proba(data)[0][1] if hasattr(model, "predict_proba") else None
    return {
        "survived": bool(prediction),
        "probability_of_survival": float(survival_prob) if survival_prob else None
    }


# 🔹 GET для списка пассажиров
@app.get("/predict_batch", tags=["Prediction"])
def predict_batch(
        age: List[float] = Query(...),
        fare: List[float] = Query(...),
        sex: List[str] = Query(...),
        pclass: List[str] = Query(...),
        embarked: List[str] = Query(...)
):
    # Проверка что все списки одной длины
    n = len(age)
    if not all(len(lst) == n for lst in [fare, sex, pclass, embarked]):
        return {"error": "Все списки должны быть одной длины"}

    data = pd.DataFrame([{
        "age": age[i],
        "fare": fare[i],
        "sex": sex[i],
        "class": pclass[i],
        "embarked": embarked[i]
    } for i in range(n)])

    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1] if hasattr(model, "predict_proba") else [None] * len(predictions)

    results = []
    for pred, prob in zip(predictions, probabilities):
        results.append({
            "survived": bool(pred),
            "probability_of_survival": float(prob) if prob is not None else None
        })
    return results






@app.get("/predict_simple")
def predict_simple(
    age: float,
    fare: float,
    sex: str,
    pclass: str,
    embarked: str
):
    data = pd.DataFrame([{
        "age": age,
        "fare": fare,
        "sex": sex,
        "class": pclass,
        "embarked": embarked
    }])
    prediction = model.predict(data)[0]
    survival_prob = model.predict_proba(data)[0][1] if hasattr(model, "predict_proba") else None
    return {
        "survived": bool(prediction),
        "probability_of_survival": float(survival_prob) if survival_prob else None
    }