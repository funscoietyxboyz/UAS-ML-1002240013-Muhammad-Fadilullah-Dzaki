"""
src/train.py

Training pipeline untuk model estimasi harga mobil bekas (regresi).

Aturan yang dipatuhi:
- Split train/test dilakukan SEBELUM preprocessing apa pun.
- Seluruh preprocessing (feature engineering, imputasi, scaling, one-hot)
  ada DI DALAM satu sklearn Pipeline.
- Yang disimpan ke .joblib adalah Pipeline utuh (feature engineering +
  preprocessing + model), bukan model telanjang.
- Test set TIDAK disentuh di file ini sama sekali (hanya disentuh sekali,
  di src/evaluate.py).

Jalankan:
    python src/train.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CarFeatureEngineer, NUMERIC_FEATURES, CATEGORICAL_FEATURES

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "used_car_data.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("feature_engineer", CarFeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])


CANDIDATES = {
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "random_forest": RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
}


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    y = df["Price"].copy()
    X = df.drop(columns=["Price"])

    # Split SEBELUM preprocessing apa pun. random_state tetap agar reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"[train] Train: {X_train.shape[0]} baris | Test: {X_test.shape[0]} baris "
          "(test set disimpan, TIDAK dipakai sampai evaluate.py)")

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    print("\n=== Perbandingan 3 algoritma dengan 5-fold CV (di data TRAIN) ===")
    cv_summary = {}
    for name, estimator in CANDIDATES.items():
        pipe = build_pipeline(estimator)
        results = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        summary = {}
        for metric in scoring:
            scores = results[f"test_{metric}"]
            if metric != "R2":
                scores = -scores  # neg_* -> positif
            summary[metric] = {"mean": float(scores.mean()), "std": float(scores.std())}
        cv_summary[name] = summary
        print(f"\n{name}:")
        for metric, stats in summary.items():
            print(f"  {metric}: mean={stats['mean']:.3f}  std={stats['std']:.3f}")

    # Metrik utama: MAE (lihat justifikasi bisnis di README/laporan).
    best_name = min(cv_summary, key=lambda n: cv_summary[n]["MAE"]["mean"])
    print(f"\n[train] Model terbaik berdasarkan MAE rata-rata CV: {best_name}")

    final_pipeline = build_pipeline(CANDIDATES[best_name])
    final_pipeline.fit(X_train, y_train)

    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(final_pipeline, model_path)
    print(f"[train] Pipeline utuh (feature engineering + preprocessing + model) "
          f"disimpan ke {model_path}")

    metadata = {
        "chosen_model": best_name,
        "primary_metric": "MAE",
        "random_state": RANDOM_STATE,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "cv_results": cv_summary,
        "target": "Price (Lakh INR)",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "sklearn_version": __import__("sklearn").__version__,
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[train] metadata.json disimpan ke {MODELS_DIR / 'metadata.json'}")
    print("[train] Test set TIDAK dipakai/dilihat sama sekali di file ini. "
          "src/evaluate.py akan mereproduksi split identik (test_size=0.2, "
          f"random_state={RANDOM_STATE}) lalu menyentuhnya SEKALI untuk evaluasi akhir.")


if __name__ == "__main__":
    main()
