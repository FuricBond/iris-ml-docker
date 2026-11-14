"""
api.py - REST API that serves the trained Iris model with FastAPI.

The model (model.joblib) is loaded once when the server starts.
Clients send four flower measurements (in cm) to POST /predict
and get back the predicted species name.

Run:
    python -m uvicorn api:app
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# The model file sits next to this script. Building the path from __file__
# means it is found both locally and inside the Docker container (/app).
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


# ---------------------------------------------------------------
# 1. LOAD the model once, when the server starts
# ---------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code before `yield` runs once at startup.
    # If the model is missing, stop the server immediately (fail fast)
    # instead of failing later on every request.
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}. "
            "Run 'python train_model.py' first to train and save the model."
        )
    # Keep the loaded model on app.state so every request can reuse it.
    app.state.model = joblib.load(MODEL_PATH)

    yield
    # Code after `yield` runs at shutdown (nothing to clean up here).


# ---------------------------------------------------------------
# 2. CREATE the FastAPI application
# ---------------------------------------------------------------
app = FastAPI(
    title="Iris ML API",
    version="1.0",
    description="Predicts the Iris flower species from four measurements in cm.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------
# 3. REQUEST and RESPONSE formats (checked by Pydantic)
# ---------------------------------------------------------------
# Each measurement must be > 0 and <= 20 cm (same rule as app.py).
# Invalid input is rejected automatically with HTTP 422.
class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0, le=20, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, le=20, description="Sepal width in cm")
    petal_length: float = Field(gt=0, le=20, description="Petal length in cm")
    petal_width: float = Field(gt=0, le=20, description="Petal width in cm")

    # Example request shown (and pre-filled) in Swagger's "Try it out".
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        }
    )


# What POST /predict sends back, e.g. {"prediction": "setosa"}.
class PredictionResponse(BaseModel):
    prediction: str


# ---------------------------------------------------------------
# 4. ENDPOINTS
# ---------------------------------------------------------------
@app.get("/")
def root():
    # Simple check that the API is up, plus a pointer to the docs.
    return {"message": "Iris Model API is running", "docs": "/docs"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    # Put the four values in the SAME order used during training.
    # scikit-learn expects a 2-D array: 1 row (one flower) x 4 columns.
    X = np.array(
        [
            [
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width,
            ]
        ]
    )

    # Only the prediction itself is wrapped. If it fails, the client gets a
    # generic 500 error, without internal details such as the stack trace.
    try:
        prediction = app.state.model.predict(X)[0]
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")

    # predict() returns a NumPy string; str() turns it into a plain str.
    return {"prediction": str(prediction)}
