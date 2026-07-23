# UAS Praktikum ML End-to-End — Kasus B: Estimasi Harga Kendaraan Bekas

> **Ganti nama folder ini** dari `uas-ml-nim` menjadi `uas-ml-<NIM Anda>` sebelum dikumpulkan,
> dan isi bagian **[ISI DATA MAHASISWA]** di bawah.

**Kasus yang dipilih:** Kasus B — Regresi: Estimasi Harga Kendaraan Bekas
**Nama / NIM:** [ISI DATA MAHASISWA]

---

## 1. Deskripsi Masalah

Sebuah marketplace otomotif ingin menyarankan **harga jual wajar** kepada penjual mobil bekas,
berdasarkan spesifikasi kendaraan (merek, tahun, jarak tempuh, mesin, dll). Target prediksi
adalah `Price` — harga mobil bekas dalam satuan **Lakh INR** (1 Lakh = 100.000 Rupee India),
sebuah bilangan kontinu, sehingga ini adalah masalah **regresi**.

Tantangan utama yang wajib ditangani (lihat Bagian 3 & laporan PDF untuk detail):
1. **Hubungan non-linear** antara umur kendaraan (`Car_Age`) dan harga.
2. **Outlier harga & jarak tempuh ekstrem** (mis. supercar/luxury car, dan satu baris
   `Kilometers_Driven` = 6.500.000 km yang mustahil secara fisik).

## 2. Sumber & Lisensi Data

- **Dataset asli:** [Used Cars Price Prediction](https://www.kaggle.com/avikasliwal/used-cars-price-prediction)
  oleh Aditya Kasliwal di Kaggle, berasal dari kompetisi publik **MachineHack "Used Cars Price
  Prediction Hackathon"**.
- **File yang digunakan (mirror publik, isi/kolom tidak diubah):**
  `https://raw.githubusercontent.com/rezaputra15/Used-Car-Data-in-India-Analysis/main/used_car_data.csv`
- **Jumlah data:** 6.019 baris × 12 kolom — memenuhi syarat minimal 1.000 baris, data nyata
  (bukan sintetis), bukan dataset dari Modul 2–6 praktikum.
- **Lisensi:** Kaggle tidak mencantumkan lisensi eksplisit pada halaman dataset asli (umum
  dipakai untuk keperluan edukasi/riset dan direplikasi luas oleh komunitas data science).
  **Catatan kejujuran akademik:** sebelum memakai ulang dataset ini untuk keperluan di luar
  tugas kuliah (mis. publikasi/komersial), disarankan memverifikasi langsung ke halaman
  Kaggle pemilik dataset.
- **Kolom dataset:** `Name, Location, Year, Kilometers_Driven, Fuel_Type, Transmission,
  Owner_Type, Mileage, Engine, Power, Seats, Price`.

## 3. Ringkasan Tantangan yang Ditangani

| Tantangan | Bagaimana ditangani |
|---|---|
| `Mileage`, `Engine`, `Power` tersimpan sebagai teks bersatuan | Diekstrak jadi numerik dgn regex di `src/features.py` (`CarFeatureEngineer`), **di dalam Pipeline** |
| Nilai hilang menyamar (`"null bhp"`, `Mileage=0` non-listrik, `Seats=0`) | Dikonversi eksplisit ke NaN, lalu diimputasi `SimpleImputer` (fit di train saja) |
| `Kilometers_Driven` outlier ekstrem (6,5 juta km) | Di-cap ke persentil ke-99.5 **data train** (ambang dipelajari saat fit, dipakai ulang saat transform) |
| Hubungan non-linear `Car_Age` vs `Price` | Dibandingkan model linear vs Random Forest — Random Forest menang telak (lihat Bagian training) |
| `Price` sangat skewed (outlier harga mobil mewah) | Metrik utama **MAE** (kurang sensitif ke outlier dibanding RMSE); dilaporkan bersama RMSE & R² |
| Kategori sama, tulisan beda (`"Land"` vs `"Land Rover"`) | Pemetaan khusus `TWO_WORD_BRANDS` di `features.py` |

Detail lengkap deteksi & alasan penanganan ada di output `src/eda.py` dan laporan PDF.

## 4. Struktur Proyek

```
uas-ml-<nim>/
├── src/
│   ├── load_data.py      # unduh dataset mentah + ringkasan wajib
│   ├── features.py        # CarFeatureEngineer (dipakai training & serving)
│   ├── eda.py              # 5 grafik EDA + temuan kekotoran data
│   ├── train.py            # split, CV 3 model, simpan pipeline .joblib
│   └── evaluate.py         # evaluasi SEKALI di test set + analisis error
├── app/
│   └── main.py              # FastAPI: GET /, GET /health, POST /predict-harga
├── tests/
│   └── test_api.py          # 4 test mekanis + 2 behavioral test
├── data/                    # dataset (di-gitignore, dibuat ulang oleh load_data.py)
├── models/                  # model.joblib + metadata.json (di-gitignore)
├── reports/                 # grafik EDA & evaluasi (PNG, boleh dikomit)
├── requirements.txt         # dependensi training
├── requirements-api.txt     # dependensi serving (versi di-pin persis)
└── README.md
```

## 5. Cara Menjalankan dari Nol

```bash
# 1) Clone & masuk ke folder
git clone <URL_REPO_ANDA>
cd uas-ml-<nim>

# 2) Buat & aktifkan virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3) Install dependensi training
pip install -r requirements.txt

# 4) Unduh data + lihat ringkasan wajib
python src/load_data.py

# 5) Jalankan EDA (hasil grafik masuk ke reports/)
python src/eda.py

# 6) Training (bandingkan 3 model dgn CV, simpan pipeline terbaik)
python src/train.py

# 7) Evaluasi akhir (test set disentuh SEKALI di sini)
python src/evaluate.py

# 8) Install dependensi serving (versi terkunci) & jalankan API
pip install -r requirements-api.txt
uvicorn app.main:app --reload
# buka http://127.0.0.1:8000/docs

# 9) Jalankan test otomatis
python -m pytest tests/ -v
```

### Versi yang dipakai saat pengembangan
- Python: 3.12.3
- pandas: 3.0.2 (kompatibel juga dgn pandas 2.x — lihat range di `requirements.txt`)
- scikit-learn: 1.8.0
- fastapi: 0.139.2

## 6. Kenapa `data/` dan `models/` Tidak Dikomit

`data/` (dataset mentah) dan `models/` (artefak `.joblib` hasil training) masuk ke
`.gitignore` karena keduanya adalah **artefak yang bisa diproduksi ulang**, bukan kode
sumber — mengomit file besar/biner ke git membuat repo berat dan riwayatnya sulit dikelola.
Penguji tetap bisa memproduksi ulang keduanya secara identik dengan menjalankan
`python src/load_data.py` (mengunduh ulang CSV dari sumber yang sama) lalu
`python src/train.py` (melatih ulang dgn `random_state=42` yang tetap, sehingga
hasil model & metrik akan sama/nyaris sama).

## 7. Perbandingan Model (5-fold Cross-Validation, data train)

| Model | MAE (Lakh) | RMSE (Lakh) | R² |
|---|---|---|---|
| Linear Regression | 2.900 ± 0.099 | 5.341 ± 0.555 | 0.773 ± 0.028 |
| Ridge | 2.918 ± 0.104 | 5.335 ± 0.556 | 0.773 ± 0.028 |
| **Random Forest (terpilih)** | **1.532 ± 0.096** | **3.617 ± 0.648** | **0.894 ± 0.030** |

**Metrik utama: MAE (Mean Absolute Error).** Dipilih dari sudut pandang bisnis karena (1)
satuannya sama dengan harga (Lakh INR) sehingga mudah dikomunikasikan ke tim bisnis/penjual
("rata-rata prediksi meleset ~1,5 Lakh"), dan (2) tidak terlalu didominasi oleh beberapa
outlier harga mobil supercar/mewah seperti RMSE (yang mengkuadratkan error, jadi sangat
sensitif ke mobil mahal ekstrem yang jumlahnya sedikit). R² tetap dilaporkan sebagai ukuran
kecocokan keseluruhan, dan RMSE dilaporkan sebagai pembanding sensitivitas terhadap outlier.

**Hasil evaluasi akhir di test set (disentuh sekali, `src/evaluate.py`):**

| Metrik | Nilai |
|---|---|
| MAE | 1.580 Lakh INR |
| RMSE | 3.796 Lakh INR |
| R² | 0.883 |

## 8. Menguji Tiga Prakiraan Tahap 2

1. **"Car_Age akan menjadi fitur penting & hubungannya non-linear"** → **Sebagian terbukti.**
   `Power_num` ternyata korelasinya paling kuat (0,77) dibanding `Car_Age` (-0,31), tapi
   hubungan Car_Age tetap jelas non-linear terlihat pada scatter plot `04_age_vs_price_scatter.png`.
2. **"Model linear akan kalah dari Random Forest"** → **Terbukti kuat.** MAE Random Forest
   (1,53) hampir setengah dari Linear Regression (2,90); R² naik dari 0,77 ke 0,89.
3. **"Brand mewah jadi sumber error terbesar"** → **Terbukti.** 4 dari 5 kesalahan prediksi
   terbesar di test set adalah Porsche, Jaguar, dan Land Rover (lihat `reports/worst5_errors.csv`
   dan laporan PDF Bagian 4).

## 9. Desain API

| Endpoint | Method | Deskripsi |
|---|---|---|
| `/` | GET | Info layanan |
| `/health` | GET | Status server & apakah model termuat |
| `/predict-harga` | POST | Prediksi harga mobil bekas |

Model dimuat sekali saat startup lewat FastAPI `lifespan` (bukan dimuat ulang di setiap
request). Validasi input memakai Pydantic: tipe data, rentang nilai (mis. `year` 1990–2026,
`kilometers_driven` 0–1.000.000), dan `Enum` untuk kolom kategorikal (`fuel_type`,
`transmission`, `owner_type`, `location`) — nilai di luar itu otomatis ditolak dengan **422**,
bukan 500. Setiap prediksi dicatat lewat `logging` (nama mobil, tahun, km, hasil prediksi,
latency).

### Contoh request berhasil (200)

```bash
curl -X POST http://127.0.0.1:8000/predict-harga \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Hyundai Creta 1.6 CRDi SX Option",
    "location": "Pune",
    "year": 2015,
    "kilometers_driven": 41000,
    "fuel_type": "Diesel",
    "transmission": "Manual",
    "owner_type": "First",
    "mileage": 19.67,
    "engine": 1582,
    "power": 126.2,
    "seats": 5
  }'
```

Respons:
```json
{
  "predicted_price_lakh_inr": 11.94,
  "confidence_interval_lakh_inr": [10.15, 13.73],
  "model_name": "RandomForestRegressor",
  "latency_ms": 42.91
}
```

### Contoh request tidak valid (422)

```bash
curl -X POST http://127.0.0.1:8000/predict-harga \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Car"}'
```

Respons (dipotong):
```json
{
  "detail": [
    {"type": "missing", "loc": ["body", "location"], "msg": "Field required"},
    {"type": "missing", "loc": ["body", "year"], "msg": "Field required"}
  ]
}
```

### Mengapa `requirements-api.txt` perlu versi terkunci, sementara `requirements.txt` tidak

Lingkungan **serving** harus reproducible persis di server produksi/mesin penguji — jika
versi `scikit-learn` yang memuat `model.joblib` berbeda dari versi saat training, struktur
internal estimator bisa berubah dan **gagal di-unpickle atau memberi hasil prediksi yang
diam-diam berbeda** tanpa error. Lingkungan **training** boleh lebih longgar (pakai rentang
versi) karena hanya dipakai developer secara lokal saat bereksperimen, dan hasil akhirnya
(`model.joblib`) sudah "dibekukan" — training ulang dgn versi library yang sedikit berbeda
tetap bisa diterima selama pipeline tetap valid.

## 10. Test Otomatis

```bash
python -m pytest tests/ -v
```

6 test (4 mekanis + 2 behavioral) — semua lolos:
- `test_health_returns_200`
- `test_predict_valid_input_returns_200_and_correct_schema`
- `test_predict_missing_field_returns_422`
- `test_predict_unknown_enum_value_returns_422`
- `test_older_car_predicted_cheaper_than_newer_identical_spec` (behavioral)
- `test_higher_power_engine_predicted_more_expensive` (behavioral)

**Mengapa behavioral test lebih tahan terhadap pelatihan ulang model:** test yang mengecek
angka prediksi persis (mis. `assert price == 11.94`) akan rusak setiap kali model dilatih
ulang meskipun modelnya tetap benar — karena data baru/random_state/versi library dapat
menggeser angka. Behavioral test hanya menuntut **arah hubungan yang masuk akal secara
bisnis** (mobil lebih tua → lebih murah; tenaga mesin lebih besar → lebih mahal) tetap
berlaku, sehingga bisa dipakai sebagai regression-safety-net lintas versi model tanpa perlu
diupdate terus-menerus.

## 11. Keterbatasan & Rencana Perbaikan

- Dataset hanya mencakup pasar India (2019 ke belakang); model tidak otomatis valid untuk
  pasar/negara lain tanpa retraining pada data lokal.
- Interval `confidence_interval_lakh_inr` pada API adalah proxy ±15%, **bukan** interval
  prediksi statistik formal (mis. quantile regression) — perbaikan lanjutan bisa memakai
  `GradientBoostingRegressor` dgn `loss="quantile"` untuk interval yang lebih valid.
  secara statistik.
  hasil.
- Belum ada hyperparameter tuning menyeluruh (mis. `RandomizedSearchCV`) karena batasan
  waktu; parameter Random Forest saat ini hasil eksperimen manual singkat.
- Baris `Porsche Cayenne` dengan `Price=2.02` Lakh pada test set (lihat Bagian 4 laporan)
  kemungkinan kesalahan input data asli (harga terlalu rendah utk spek mobil), namun tetap
  disertakan karena test set tidak boleh "dibersihkan" setelah split.
#   U A S - M L - 1 0 0 2 2 4 0 0 1 3 - M u h a m m a d - F a d i l u l l a h - D z a k i 
 
 
