"""
app.py

Task 2: Loads the trained model (model.joblib) and predicts the Iris species
for four measurements given as command-line arguments:
    python app.py SEPAL_LENGTH SEPAL_WIDTH PETAL_LENGTH PETAL_WIDTH
With no arguments, the default sample 5.1 3.5 1.4 0.2 is used.
"""
import math
import sys
from pathlib import Path

import joblib
import numpy as np

# The model file sits next to this script. Building the path from __file__
# means it is found both locally and inside the Docker container (/app).
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

# The four input features, in the same order the model was trained on.
FEATURE_NAMES = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

# Upper limit for any measurement (real Iris values are all under 8 cm).
MAX_MEASUREMENT_CM = 20.0

# Used when no arguments are given (e.g. a plain `docker run`).
# This is the first flower in the Iris dataset (a setosa).
DEFAULT_SAMPLE = ["5.1", "3.5", "1.4", "0.2"]

USAGE = (
    "Usage:   python app.py SEPAL_LENGTH SEPAL_WIDTH PETAL_LENGTH PETAL_WIDTH\n"
    "Example: python app.py 6.7 3.0 5.2 2.3\n"
    "Run with no arguments to use the default sample (5.1 3.5 1.4 0.2)."
)


def fail(message):
    """Print an error message to stderr and stop with exit code 1."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def read_features(args):
    """Turn the command-line arguments into a list of 4 positive numbers."""
    # No arguments -> use the default sample.
    if len(args) == 0:
        args = DEFAULT_SAMPLE
    # Anything other than 0 or 4 arguments is a mistake.
    elif len(args) != 4:
        fail(f"expected 4 values but got {len(args)}.\n{USAGE}")

    features = []
    for name, text in zip(FEATURE_NAMES, args):
        # Each value must be a number...
        try:
            value = float(text)
        except ValueError:
            fail(f"{name} must be a number, but got '{text}'.")

        # ...a finite one (float() also accepts "nan" and "inf")...
        if not math.isfinite(value):
            fail(f"{name} must be a finite number, but got '{text}'.")

        # ...and a realistic flower measurement: above 0 and at most 20 cm.
        if value <= 0 or value > MAX_MEASUREMENT_CM:
            fail(
                f"{name} must be greater than 0 and at most "
                f"{MAX_MEASUREMENT_CM:g} cm, but got {value}."
            )

        features.append(value)

    return features


def main():
    # ---------------------------------------------------------------
    # 1. READ the four input features from the command line
    # ---------------------------------------------------------------
    # sys.argv[0] is the script name, so the values start at index 1.
    features = read_features(sys.argv[1:])

    # ---------------------------------------------------------------
    # 2. LOAD the trained model
    # ---------------------------------------------------------------
    if not MODEL_PATH.is_file():
        fail(
            f"model file not found: {MODEL_PATH}\n"
            "Run 'python train_model.py' first to train and save the model."
        )
    model = joblib.load(MODEL_PATH)

    # ---------------------------------------------------------------
    # 3. PREDICT the species
    # ---------------------------------------------------------------
    # scikit-learn expects a 2-D array: one row per flower and one
    # column per feature. We have one flower, so the shape is (1, 4).
    X = np.array([features])
    prediction = model.predict(X)[0]  # e.g. "setosa"

    # ---------------------------------------------------------------
    # 4. DISPLAY the input values and the predicted species
    # ---------------------------------------------------------------
    print("Iris Species Prediction")
    print("Input:")
    for name, value in zip(FEATURE_NAMES, features):
        print(f"  {name:<12} : {value}")
    print("Prediction:")
    print(f"  {prediction}")


if __name__ == "__main__":
    main()
