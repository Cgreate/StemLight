"""Reproduce StemLight's five classifiers in standard Python or BITS Virtual Lab.

Example: python train_models.py --output-dir experiments/reproduced
Existing outputs are never overwritten. The Streamlit app never invokes this script.
"""
import argparse
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
DATASET_URL = "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"


def build_models():
    """Match the parameters used to produce the bundled example models."""
    return {
        "Logistic Regression": ("logistic_regression", make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))),
        "Decision Tree": ("decision_tree", DecisionTreeClassifier(random_state=SEED)),
        "KNN": ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier())),
        "Gaussian Naive Bayes": ("gaussian_naive_bayes", GaussianNB()),
        "Support Vector Machine": ("support_vector_machine", make_pipeline(
            StandardScaler(), SVC(probability=True, random_state=SEED))),
    }


def train_models(output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    models = build_models()
    paths = [output_dir / "model" / f"{stem}.joblib" for stem, _ in models.values()]
    paths += [output_dir / name for name in
              ["test_data.csv", "metrics.csv", "training_info.json"]]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Choose a fresh --output-dir; existing outputs will not be overwritten: "
            + ", ".join(existing)
        )

    dataset = load_breast_cancer(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2,
        stratify=dataset.target, random_state=SEED,
    )
    assert set(X_train.index).isdisjoint(X_test.index)
    (output_dir / "model").mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (stem, model) in models.items():
        # Pipelines fit their scalers only on X_train, preventing data leakage.
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        positive_column = list(model.classes_).index(1)
        probability = model.predict_proba(X_test)[:, positive_column]
        rows.append({
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, prediction),
            "AUC": roc_auc_score(y_test, probability),
            "Precision": precision_score(y_test, prediction, zero_division=0),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "F1": f1_score(y_test, prediction, zero_division=0),
            "MCC": matthews_corrcoef(y_test, prediction),
        })
        joblib.dump(model, output_dir / "model" / f"{stem}.joblib")

    test_data = X_test.copy()
    test_data["target"] = y_test
    test_data.to_csv(output_dir / "test_data.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    metadata = {
        "dataset": "Wisconsin Diagnostic Breast Cancer",
        "source_url": DATASET_URL,
        "loader": "sklearn.datasets.load_breast_cancer(as_frame=True)",
        "instances": len(dataset.data), "features": dataset.data.shape[1],
        "class_counts": {str(k): int(v) for k, v in dataset.target.value_counts().sort_index().items()},
        "train_rows": len(X_train), "test_rows": len(X_test),
        "train_class_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "random_state": SEED, "test_fraction": 0.2, "stratified": True,
        "target_labels": {"0": "malignant", "1": "benign"},
        "positive_class": "benign (1)",
        "evaluation": "One held-out split; no cross-validation or hyperparameter tuning",
        "preprocessing": "StandardScaler inside LR/KNN/SVM pipelines, fitted on training rows only",
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
    }
    (output_dir / "training_info.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Dataset: {len(dataset.data)} rows, {dataset.data.shape[1]} features")
    print(f"Training: {len(X_train)} rows | Held-out test: {len(X_test)} rows")
    print("Positive class: benign (1)")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved five models, test_data.csv, metrics.csv and training_info.json to: {output_dir}")
    print("This output alone is not proof of BITS Virtual Lab execution.")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "experiments" / "reproduced",
                        help="Fresh output folder; bundled models are not overwritten")
    args = parser.parse_args()
    try:
        train_models(args.output_dir)
    except FileExistsError as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
