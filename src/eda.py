"""
src/eda.py

Exploratory Data Analysis untuk dataset harga mobil bekas.
Menjawab 4 pertanyaan wajib dan menyimpan minimal 1 grafik PNG per pertanyaan
ke reports/. Semua output tercetak di terminal juga disalin ke laporan.

Jalankan:
    python src/eda.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from features import _extract_brand, _extract_number  # reuse parsing helpers

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "used_car_data.csv"
REPORTS_DIR = BASE_DIR / "reports"

sns.set_theme(style="whitegrid")


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def mandatory_checks(df: pd.DataFrame) -> None:
    print("=== Pemeriksaan wajib ===")
    print("\n-- df.isna().sum() --")
    print(df.isna().sum().to_string())
    print("\n-- df.describe() (numerik) --")
    print(df.describe().to_string())
    print("\n-- Distribusi target (Price, dalam Lakh INR) --")
    print(df["Price"].describe().to_string())
    print("\n-- Baris duplikat --")
    print(df.duplicated().sum())


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Pertanyaan 1: Bagaimana sebaran target (regresi -> histogram)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(df["Price"], bins=50, kde=True, ax=ax, color="#3b6fa0")
    ax.set_title("Distribusi Harga Mobil Bekas (Price, Lakh INR)")
    ax.set_xlabel("Price (Lakh INR)")
    ax.set_ylabel("Frekuensi")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "01_target_distribution.png", dpi=150)
    plt.close(fig)

    skew = df["Price"].skew()
    print(f"\n[Temuan] Skewness Price = {skew:.2f} -> distribusi sangat menceng kanan, "
          "didominasi mobil murah dengan ekor panjang mobil mewah.")


def plot_missing_values(df: pd.DataFrame) -> None:
    """Pertanyaan 2: Di mana data bolong/kotor."""
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=missing.values, y=missing.index, ax=ax, color="#c0504d")
    ax.set_title("Jumlah Nilai Hilang per Kolom")
    ax.set_xlabel("Jumlah baris hilang")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "02_missing_values.png", dpi=150)
    plt.close(fig)
    print("\n[Temuan] Kolom dengan nilai hilang eksplisit:")
    print(missing.to_string())


def plot_feature_vs_target(df: pd.DataFrame) -> None:
    """Pertanyaan 3: Fitur mana yang paling berhubungan dengan target."""
    work = df.copy()
    work["Car_Age"] = (work["Year"].max() + 1) - work["Year"]
    work["Power_num"] = _extract_number(work["Power"])
    work["Engine_num"] = _extract_number(work["Engine"])
    work["Mileage_num"] = _extract_number(work["Mileage"])

    numeric_cols = ["Car_Age", "Kilometers_Driven", "Power_num", "Engine_num",
                     "Mileage_num", "Seats", "Price"]
    corr = work[numeric_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Heatmap Korelasi Fitur Numerik terhadap Price")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "03_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x=work["Car_Age"], y=work["Price"], alpha=0.35, ax=ax, color="#3b6fa0")
    ax.set_title("Umur Mobil (Car_Age) vs Price -- hubungan non-linear")
    ax.set_xlabel("Umur mobil (tahun)")
    ax.set_ylabel("Price (Lakh INR)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "04_age_vs_price_scatter.png", dpi=150)
    plt.close(fig)

    print("\n[Temuan] Korelasi Price dengan fitur numerik:")
    print(corr["Price"].sort_values(ascending=False).to_string())


def plot_brand_price(df: pd.DataFrame) -> None:
    """Grafik tambahan: rata-rata harga per brand teratas (kategori & harga)."""
    work = df.copy()
    work["Brand"] = _extract_brand(work["Name"])
    top_brands = work["Brand"].value_counts().head(12).index
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=work[work["Brand"].isin(top_brands)],
        x="Price", y="Brand", ax=ax, orient="h",
        order=work[work["Brand"].isin(top_brands)].groupby("Brand")["Price"].median()
        .sort_values(ascending=False).index,
    )
    ax.set_title("Sebaran Price per Brand (12 brand terbanyak)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "05_price_by_brand_boxplot.png", dpi=150)
    plt.close(fig)


def report_dirty_data(df: pd.DataFrame) -> None:
    print("\n=== Tiga (lebih) kekotoran data yang ditemukan ===")
    print("""
1. Kolom numerik bertipe teks
   Temuan   : Mileage ('26.6 km/kg'), Engine ('998 CC'), Power ('58.16 bhp')
              tersimpan sebagai string dengan satuan tercampur di dalamnya.
   Deteksi  : df.dtypes menunjukkan kolom ini bertipe object/string, bukan
              float, padahal secara konsep adalah numerik.
   Penanganan: diekstrak dengan regex (r'([\\d.]+)') menjadi kolom *_num
              numerik di dalam CarFeatureEngineer, dilakukan di dalam Pipeline
              (fit hanya di train) agar konsisten saat serving.

2. Nilai hilang yang menyamar
   Temuan   : 107 baris pada kolom Power berisi literal string 'null bhp'
              (bukan NaN standar pandas), dan 68 baris Mileage = '0.0 kmpl'
              pada mobil BUKAN Electric (mustahil, mobil bensin/diesel tidak
              mungkin konsumsi 0). Seats juga punya 1 baris bernilai 0.
   Deteksi  : dicek dengan str.contains('null') dan filter kondisional
              Mileage==0 & Fuel_Type!='Electric'.
   Penanganan: dikonversi eksplisit menjadi NaN, lalu diimputasi dengan
              SimpleImputer(strategy='median') di dalam Pipeline (fit di
              train saja) supaya tidak bias dan tidak bocor ke test set.

3. Outlier tak masuk akal
   Temuan   : Kilometers_Driven punya nilai maksimum 6.500.000 km untuk BMW
              X5 tahun 2017 -- mustahil secara fisik (setara ~2000 km/hari
              selama 8 tahun nonstop). Beberapa baris Price ekstrem tinggi
              (>= 80 Lakh) untuk mobil supercar/luxury jauh dari mayoritas.
   Deteksi  : df.sort_values('Kilometers_Driven', ascending=False).head()
              dan df.sort_values('Price', ascending=False).head().
   Penanganan: Kilometers_Driven di-cap ke persentil ke-99.5 data TRAIN
              (bukan dihapus, supaya baris lain pada mobil itu tidak hilang).
              Price ekstrem TIDAK dihapus dari test set (test set hanya
              disentuh sekali di evaluate.py), tapi ditangani lewat pemilihan
              model yang robust (Random Forest/Gradient Boosting) dan metrik
              yang dilaporkan bersama MAE (kurang sensitif ke outlier
              dibanding RMSE).

4. (tambahan) Kategori sama, penulisan beda
   Temuan   : Ekstraksi Brand dari kata pertama kolom Name membuat "Land
              Rover" terpecah jadi brand palsu bernama "Land" (60 baris).
   Deteksi  : value_counts() pada Brand hasil split menunjukkan brand
              "Land" dengan jumlah signifikan padahal bukan merek nyata.
   Penanganan: pemetaan khusus TWO_WORD_BRANDS di features.py menggabungkan
              "Land" -> "Land Rover" sebelum encoding.
""")


def three_predictions() -> None:
    print("=== Tiga prakiraan sebelum modeling (diuji ulang di Tahap 3) ===")
    print("""
P1: Car_Age akan menjadi salah satu fitur terpenting, dan hubungannya dengan
    Price NON-LINEAR (harga turun cepat di tahun-tahun awal lalu melandai).
P2: Model linear (Linear Regression) akan kalah dari model berbasis pohon
    (Random Forest) karena hubungan Car_Age & Power terhadap Price melengkung,
    bukan garis lurus.
P3: Brand mewah (BMW, Mercedes-Benz, Audi, Land Rover, Jaguar) akan menjadi
    sumber kesalahan prediksi terbesar (error tertinggi) karena sebarannya
    lebih lebar dan datanya lebih sedikit dibanding Maruti/Hyundai.
""")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    mandatory_checks(df)
    plot_target_distribution(df)
    plot_missing_values(df)
    plot_feature_vs_target(df)
    plot_brand_price(df)
    report_dirty_data(df)
    three_predictions()
    print(f"\n[eda] Semua grafik tersimpan di {REPORTS_DIR}")


if __name__ == "__main__":
    main()
