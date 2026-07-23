"""
src/features.py

Transformer feature-engineering kustom untuk dataset harga mobil bekas.
Dipakai di dalam sklearn Pipeline (src/train.py) DAN saat serving (app/main.py)
supaya logika pembersihan/derivasi fitur identik antara training dan produksi.

Kekotoran data yang ditangani di sini (lihat laporan Tahap 2 untuk detail):
1. Mileage / Engine / Power tersimpan sebagai teks bersatuan ("998 CC", "58.16 bhp")
   -> diekstrak jadi numerik dengan regex.
2. Nilai hilang yang menyamar: string "null bhp" pada Power, dan Mileage = "0.0 kmpl"
   untuk mobil non-listrik -> dikonversi eksplisit menjadi NaN agar bisa diimputasi
   secara wajar oleh SimpleImputer di dalam Pipeline (bukan di sini, supaya tidak
   ada leakage: imputer di-fit hanya pada data train).
3. Outlier tak masuk akal: Kilometers_Driven memiliki nilai ekstrem (mis. 6.500.000 km
   untuk mobil tahun 2017) yang mustahil secara fisik -> di-cap ke persentil ke-99.5
   data TRAIN (ambang batas dipelajari saat fit, bukan dihitung ulang saat transform
   pada data baru/test, sehingga tidak ada kebocoran informasi dari test/production).
4. Kolom "Name" (nyaris unik per baris, mirip ID) diganti dengan "Brand" (kata pertama
   nama mobil, dengan penanganan khusus merek dua kata seperti "Land Rover").
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TWO_WORD_BRANDS = {"Land": "Land Rover", "Mini": "Mini"}  # "Mini" tetap 1 kata (brand sah)

NUMERIC_FEATURES = [
    "Car_Age",
    "Kilometers_Driven",
    "Mileage_num",
    "Engine_num",
    "Power_num",
    "Seats",
]
CATEGORICAL_FEATURES = ["Brand", "Location", "Fuel_Type", "Transmission", "Owner_Type"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _extract_number(series: pd.Series) -> pd.Series:
    """Ambil bagian numerik pertama dari kolom string bersatuan.
    String non-numerik seperti 'null bhp' otomatis menjadi NaN."""
    extracted = series.astype(str).str.extract(r"([\d.]+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def _extract_brand(name_series: pd.Series) -> pd.Series:
    first_word = name_series.astype(str).str.split().str[0]
    return first_word.map(lambda w: TWO_WORD_BRANDS.get(w, w))


class CarFeatureEngineer(BaseEstimator, TransformerMixin):
    """Mengubah dataframe mentah menjadi fitur bersih siap dipakai ColumnTransformer.

    Semua ambang batas (reference_year, cap kilometer) dipelajari saat fit()
    dari data TRAIN saja, lalu dipakai kembali saat transform() -- termasuk
    saat transform dipanggil untuk request tunggal dari API.
    """

    def __init__(self, km_cap_quantile: float = 0.995):
        self.km_cap_quantile = km_cap_quantile

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.reference_year_ = int(X["Year"].max()) + 1
        km_numeric = pd.to_numeric(X["Kilometers_Driven"], errors="coerce")
        self.km_cap_value_ = float(km_numeric.quantile(self.km_cap_quantile))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1) Brand dari Name
        X["Brand"] = _extract_brand(X["Name"]) if "Name" in X.columns else "Unknown"

        # 2) Umur mobil (fitur non-linear utama terhadap harga)
        X["Car_Age"] = self.reference_year_ - pd.to_numeric(X["Year"], errors="coerce")

        # 3) Bersihkan kolom numerik bertipe teks
        X["Mileage_num"] = _extract_number(X["Mileage"]) if "Mileage" in X.columns else np.nan
        X["Engine_num"] = _extract_number(X["Engine"]) if "Engine" in X.columns else np.nan
        X["Power_num"] = _extract_number(X["Power"]) if "Power" in X.columns else np.nan

        # Mileage 0.0 untuk mobil non-listrik = nilai hilang yang menyamar
        non_electric = X.get("Fuel_Type", pd.Series(["Petrol"] * len(X))) != "Electric"
        X.loc[non_electric & (X["Mileage_num"] == 0), "Mileage_num"] = np.nan

        # Seats = 0 mustahil -> jadikan NaN agar diimputasi
        if "Seats" in X.columns:
            X.loc[X["Seats"] == 0, "Seats"] = np.nan

        # 4) Cap outlier Kilometers_Driven memakai ambang dari data train
        X["Kilometers_Driven"] = pd.to_numeric(X["Kilometers_Driven"], errors="coerce")
        X["Kilometers_Driven"] = X["Kilometers_Driven"].clip(upper=self.km_cap_value_)

        for col in CATEGORICAL_FEATURES:
            if col not in X.columns:
                X[col] = "Unknown"
            X[col] = X[col].fillna("Unknown").astype(str)

        return X[ALL_FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.array(ALL_FEATURES)
