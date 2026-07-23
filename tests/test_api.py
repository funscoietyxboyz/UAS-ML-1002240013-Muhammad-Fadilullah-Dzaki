"""
tests/test_api.py

Minimal 4 test mekanis + minimal 2 behavioral test.
Jalankan:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # TestClient dipakai sbg context manager agar lifespan (startup/shutdown,
    # termasuk pemuatan model ke ml_model) benar-benar dijalankan.
    with TestClient(app) as c:
        yield c

VALID_PAYLOAD = {
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
    "seats": 5,
}


# ---------------------------------------------------------------------------
# 1) Test mekanis (>= 4)
# ---------------------------------------------------------------------------
def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_input_returns_200_and_correct_schema(client):
    resp = client.post("/predict-harga", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert "predicted_price_lakh_inr" in body
    assert isinstance(body["predicted_price_lakh_inr"], float)
    assert body["predicted_price_lakh_inr"] > 0
    assert "confidence_interval_lakh_inr" in body
    assert len(body["confidence_interval_lakh_inr"]) == 2


def test_predict_missing_field_returns_422(client):
    payload = VALID_PAYLOAD.copy()
    del payload["year"]
    resp = client.post("/predict-harga", json=payload)
    assert resp.status_code == 422


def test_predict_unknown_enum_value_returns_422(client):
    payload = VALID_PAYLOAD.copy()
    payload["fuel_type"] = "Nuklir"  # bukan salah satu dari enum FuelType
    resp = client.post("/predict-harga", json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 2) Behavioral test (>= 2) -- menguji RELASI/arah perilaku model,
#    bukan angka prediksi persis. Tahan terhadap pelatihan ulang model
#    karena hanya menuntut konsistensi arah, bukan nilai output eksak.
# ---------------------------------------------------------------------------
def test_older_car_predicted_cheaper_than_newer_identical_spec(client):
    """Kasus B behavioral test: mobil lebih tua dgn spesifikasi lain identik
    harus diprediksi lebih murah daripada mobil yang lebih baru."""
    newer = VALID_PAYLOAD.copy()
    newer["year"] = 2019

    older = VALID_PAYLOAD.copy()
    older["year"] = 2010

    price_newer = client.post("/predict-harga", json=newer).json()["predicted_price_lakh_inr"]
    price_older = client.post("/predict-harga", json=older).json()["predicted_price_lakh_inr"]

    assert price_older < price_newer


def test_higher_power_engine_predicted_more_expensive(client):
    """Behavioral test tambahan: mobil dengan tenaga mesin (Power) jauh lebih
    besar, spesifikasi lain identik, harus diprediksi lebih mahal -- selaras
    dengan temuan EDA bahwa Power adalah fitur paling berkorelasi dgn Price."""
    low_power = VALID_PAYLOAD.copy()
    low_power["power"] = 60.0
    low_power["engine"] = 800

    high_power = VALID_PAYLOAD.copy()
    high_power["power"] = 300.0
    high_power["engine"] = 3000

    price_low = client.post("/predict-harga", json=low_power).json()["predicted_price_lakh_inr"]
    price_high = client.post("/predict-harga", json=high_power).json()["predicted_price_lakh_inr"]

    assert price_high > price_low
