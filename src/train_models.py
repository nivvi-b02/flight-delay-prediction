import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flights_features.csv"
)

MODEL_PATH = PROJECT_ROOT / "models"

REPORTS_PATH = PROJECT_ROOT / "reports"

MODEL_PATH.mkdir(exist_ok=True)
REPORTS_PATH.mkdir(exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

flights = pd.read_csv(
    DATA_PATH,
    low_memory=False,
    dtype={
        "AIRLINE": str,
        "ORIGIN_AIRPORT": str,
        "DESTINATION_AIRPORT": str,
        "TIME_OF_DAY": str
    }
)

print(f"Dataset shape: {flights.shape}")


# ==================================================
# 3. SELECT FEATURES
# ==================================================

print("\n" + "=" * 60)
print("SELECTING FEATURES")
print("=" * 60)

# Features available before a flight departs
features = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_TIME",
    "DISTANCE",
    "SCHEDULED_ARRIVAL",
    "DEPARTURE_HOUR",
    "TIME_OF_DAY"
]

target = "IS_DELAYED"

X = flights[features]
y = flights[target]

print("\nFeatures:")
for feature in features:
    print(f" - {feature}")

print(f"\nTarget: {target}")


# ==================================================
# 4. TRAIN / TEST SPLIT
# ==================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows:  {len(X_test):,}")

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(4))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(4))


# ==================================================
# 5. DEFINE FEATURE TYPES
# ==================================================

categorical_features = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "TIME_OF_DAY"
]

numeric_features = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_TIME",
    "DISTANCE",
    "SCHEDULED_ARRIVAL",
    "DEPARTURE_HOUR"
]


# ==================================================
# 6. PREPROCESSING
# ==================================================

print("\n" + "=" * 60)
print("SETTING UP PREPROCESSING")
print("=" * 60)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_features
        )
    ]
)


# ==================================================
# 7. DEFINE MODELS
# ==================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Linear SVM": LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=42
    )
}


# ==================================================
# 8. TRAIN MODELS
# ==================================================

results = []

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    print("Training model...")

    pipeline.fit(X_train, y_train)

    print("Model training complete.")

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    y_pred = pipeline.predict(X_test)

    # --------------------------------------------------
    # Scores for ROC-AUC and PR-AUC
    # --------------------------------------------------

    if hasattr(pipeline, "predict_proba"):

        y_score = pipeline.predict_proba(X_test)[:, 1]

    else:

        y_score = pipeline.decision_function(X_test)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_score
    )

    pr_auc = average_precision_score(
        y_test,
        y_score
    )

    print("\nResults:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    safe_name = model_name.lower().replace(" ", "_")

    model_file = MODEL_PATH / f"{safe_name}.joblib"

    joblib.dump(
        pipeline,
        model_file
    )

    print(f"\nSaved model:")
    print(model_file)

    # --------------------------------------------------
    # Store results
    # --------------------------------------------------

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc
    })


# ==================================================
# 9. MODEL COMPARISON
# ==================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results)

results_df = results_df.round(4)

print("\n")
print(results_df.to_string(index=False))


# ==================================================
# 10. SAVE RESULTS
# ==================================================

results_path = REPORTS_PATH / "model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("\nModel comparison saved to:")
print(results_path)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)