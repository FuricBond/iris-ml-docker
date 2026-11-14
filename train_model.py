from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Save the model next to this script, no matter which folder we run it from.
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


def main():
    # ---------------------------------------------------------------
    # 1. LOAD the dataset
    # ---------------------------------------------------------------
    # load_iris() returns NumPy arrays:
    #   X -> 150 rows x 4 features (sepal/petal length and width, in cm)
    #   y -> 150 numeric labels (0, 1, 2)
    iris = load_iris()
    X = iris.data

    # Turn the numeric labels into species names
    # (0 -> "setosa", 1 -> "versicolor", 2 -> "virginica"),
    # so the trained model predicts the species name directly.
    y = iris.target_names[iris.target]

    # ---------------------------------------------------------------
    # 2. SPLIT into training and testing sets
    # ---------------------------------------------------------------
    # 80% for training, 20% for testing.
    # random_state=42 makes the split the same on every run.
    # stratify=y keeps the same proportion of each species in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("=" * 55)
    print("IRIS MODEL TRAINING")
    print("=" * 55)
    print(f"Dataset size      : {len(X)} samples, {X.shape[1]} features")
    print(f"Training samples  : {len(X_train)}")
    print(f"Testing samples   : {len(X_test)}")

    # ---------------------------------------------------------------
    # 3. TRAIN the model
    # ---------------------------------------------------------------
    # A Random Forest is a group of 100 decision trees that vote
    # on the final answer.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ---------------------------------------------------------------
    # 4. EVALUATE on the test set (data the model has never seen)
    # ---------------------------------------------------------------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("-" * 55)
    print(f"Accuracy          : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("-" * 55)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # ---------------------------------------------------------------
    # 5. SAVE the trained model to disk
    # ---------------------------------------------------------------
    # Only the model object is saved; app code can load it later
    # with joblib.load() and call model.predict().
    joblib.dump(model, MODEL_PATH)

    print("-" * 55)
    print(f"Model saved to    : {MODEL_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    main()
