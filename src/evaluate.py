"""
src/evaluate.py

Evaluasi akhir pada test set. File ini adalah SATU-SATUNYA tempat test set
disentuh, dan hanya SEKALI, setelah model final dipilih & dilatih di
src/train.py. Split direproduksi persis (test_size=0.2, random_state=42)
sehingga baris test di sini identik dengan yang disisihkan saat training.

Jalankan:
    python src/evaluate.py
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "used_car_data.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

sns.set_theme(style="whitegrid")


def reproduce_test_split():
    df = pd.read_csv(DATA_PATH)
    y = df["Price"].copy()
    X = df.drop(columns=["Price"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_test, y_test


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = joblib.load(MODELS_DIR / "model.joblib")

    X_test, y_test = reproduce_test_split()
    print(f"[evaluate] Test set: {X_test.shape[0]} baris (disentuh SEKALI di sini)")

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    print("\n=== Hasil evaluasi akhir pada test set (disentuh sekali) ===")
    print(f"MAE  : {mae:.3f} Lakh INR")
    print(f"RMSE : {rmse:.3f} Lakh INR")
    print(f"R2   : {r2:.3f}")

    # Plot: actual vs predicted
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.35, color="#3b6fa0")
    lims = [0, max(y_test.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1, label="Prediksi sempurna")
    ax.set_xlabel("Harga aktual (Lakh INR)")
    ax.set_ylabel("Harga prediksi (Lakh INR)")
    ax.set_title(f"Actual vs Predicted (Test set, n={len(y_test)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "06_actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    # Plot: residuals
    residuals = y_test.values - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(residuals, bins=50, kde=True, ax=ax, color="#c0504d")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Distribusi Residual (Actual - Predicted)")
    ax.set_xlabel("Residual (Lakh INR)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "07_residuals.png", dpi=150)
    plt.close(fig)

    # Analisis 5 kesalahan terburuk
    error_df = X_test.copy()
    error_df["Price_actual"] = y_test.values
    error_df["Price_predicted"] = y_pred
    error_df["abs_error"] = np.abs(error_df["Price_actual"] - error_df["Price_predicted"])
    worst5 = error_df.sort_values("abs_error", ascending=False).head(5)
    cols_to_show = ["Name", "Year", "Kilometers_Driven", "Fuel_Type",
                     "Price_actual", "Price_predicted", "abs_error"]
    print("\n=== 5 Kesalahan Prediksi Terburuk ===")
    print(worst5[cols_to_show].to_string(index=False))

    worst5[cols_to_show].to_csv(REPORTS_DIR / "worst5_errors.csv", index=False)

    eval_results = {
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "n_test": int(len(y_test)),
    }
    with open(MODELS_DIR / "evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n[evaluate] Hasil evaluasi disimpan ke {MODELS_DIR / 'evaluation_results.json'}")
    print(f"[evaluate] Grafik & tabel 5 error terburuk disimpan ke {REPORTS_DIR}")


if __name__ == "__main__":
    main()
