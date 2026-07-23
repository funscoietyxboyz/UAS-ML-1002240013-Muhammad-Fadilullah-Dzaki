"""
src/load_data.py

Mengunduh dataset mentah "Used Cars Price Prediction" ke data/,
lalu mencetak ringkasan jumlah baris, kolom, tipe data, dan nilai hilang.

Sumber data
-----------
Dataset asli: Kaggle - avikasliwal/used-cars-price-prediction (train-data.csv),
berasal dari kompetisi publik MachineHack "Used Cars Price Prediction Hackathon".
File diakses melalui mirror publik di GitHub (kolom & isi tidak diubah):
https://raw.githubusercontent.com/rezaputra15/Used-Car-Data-in-India-Analysis/main/used_car_data.csv

Jalankan:
    python src/load_data.py
"""

import sys
import urllib.request
from pathlib import Path

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/rezaputra15/"
    "Used-Car-Data-in-India-Analysis/main/used_car_data.csv"
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "used_car_data.csv"


def download_data(url: str = DATA_URL, dest: Path = RAW_PATH) -> Path:
    """Unduh CSV mentah jika belum ada secara lokal."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[load_data] File sudah ada di {dest}, lewati unduhan.")
        return dest

    print(f"[load_data] Mengunduh dataset dari {url} ...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:  # pragma: no cover
        print(f"[load_data] GAGAL mengunduh dataset: {exc}", file=sys.stderr)
        raise
    print(f"[load_data] Dataset tersimpan di {dest}")
    return dest


def summarize(path: Path = RAW_PATH) -> pd.DataFrame:
    """Cetak ringkasan wajib: jumlah baris, kolom, tipe kolom, nilai hilang."""
    df = pd.read_csv(path)

    print("\n=== RINGKASAN DATASET ===")
    print(f"Jumlah baris  : {df.shape[0]}")
    print(f"Jumlah kolom  : {df.shape[1]}")

    print("\nTipe tiap kolom:")
    print(df.dtypes.to_string())

    print("\nJumlah nilai hilang per kolom:")
    print(df.isna().sum().to_string())

    print("\nJumlah baris duplikat:", df.duplicated().sum())
    print("=== END RINGKASAN ===\n")
    return df


if __name__ == "__main__":
    p = download_data()
    summarize(p)
