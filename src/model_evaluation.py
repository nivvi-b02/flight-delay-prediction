import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
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

REPORTS_PATH.mkdir(exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

print("=" * 60)
print("LOADING DATA FOR MODEL EVALUATION")
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


# ==================================================
# 4. SAME TRAIN / TEST SPLIT
# ==================================================

print("\n" + "=" * 60)
print("CREATING TEST SET")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Testing rows: {len(X_test):,}")


# ==================================================
# 5. FEATURE TYPES
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

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
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
# 7. LOAD SAVED MODELS
# ==================================================

print("\n" + "=" * 60)
print("LOADING SAVED MODELS")
print("=" * 60)

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "Linear SVM": "linear_svm.joblib"
}

models = {}

for model_name, filename in model_files.items():

    model_path = MODEL_PATH / filename

    models[model_name] = joblib.load(model_path)

    print(f"Loaded: {model_name}")


# ==================================================
# 8. EVALUATE MODELS
# ==================================================

roc_results = {}
pr_results = {}

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"EVALUATING: {model_name}")
    print("=" * 60)

    # Predictions
    y_pred = model.predict(X_test)

    # Score for ROC and PR curves
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Delayed", "Delayed"]
    )

    display.plot()

    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    cm_path = (
        REPORTS_PATH
        / f"confusion_matrix_{safe_name}.png"
    )

    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"Saved: {cm_path}")

    # --------------------------------------------------
    # ROC Curve Data
    # --------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        y_score
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    roc_results[model_name] = (
        fpr,
        tpr,
        roc_auc
    )

    # --------------------------------------------------
    # Precision-Recall Curve Data
    # --------------------------------------------------

    precision, recall, _ = precision_recall_curve(
        y_test,
        y_score
    )

    pr_auc = average_precision_score(
        y_test,
        y_score
    )

    pr_results[model_name] = (
        precision,
        recall,
        pr_auc
    )


# ==================================================
# 9. ROC CURVE
# ==================================================

print("\n" + "=" * 60)
print("CREATING ROC CURVE")
print("=" * 60)

plt.figure(figsize=(8, 6))

for model_name, (fpr, tpr, roc_auc) in roc_results.items():

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {roc_auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()

roc_path = REPORTS_PATH / "roc_curve_comparison.png"

plt.savefig(
    roc_path,
    dpi=300
)

plt.close()

print(f"Saved: {roc_path}")


# ==================================================
# 10. PRECISION-RECALL CURVE
# ==================================================

print("\n" + "=" * 60)
print("CREATING PRECISION-RECALL CURVE")
print("=" * 60)

plt.figure(figsize=(8, 6))

for model_name, (precision, recall, pr_auc) in pr_results.items():

    plt.plot(
        recall,
        precision,
        label=f"{model_name} (AP = {pr_auc:.3f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.tight_layout()

pr_path = REPORTS_PATH / "precision_recall_curve_comparison.png"

plt.savefig(
    pr_path,
    dpi=300
)

plt.close()

print(f"Saved: {pr_path}")


# ==================================================
# 11. COMPLETE
# ==================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETE")
print("=" * 60)

print("\nEvaluation files saved in:")
print(REPORTS_PATH)