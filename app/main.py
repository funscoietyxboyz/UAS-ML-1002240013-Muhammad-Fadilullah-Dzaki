"""
app/main.py

REST API FastAPI untuk melayani prediksi harga mobil bekas.
Jalankan dengan:
    uvicorn app.main:app --reload
Lalu buka http://127.0.0.1:8000/docs
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Supaya `import features` (dipakai di dalam pipeline.joblib) berhasil,
# src/ perlu ada di sys.path -- feature engineering harus identik dengan training.
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

MODEL_PATH = BASE_DIR / "models" / "model.joblib"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("used-car-api")

ml_model: dict = {}  # diisi saat lifespan startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        ml_model["pipeline"] = joblib.load(MODEL_PATH)
        logger.info("Model dimuat dari %s", MODEL_PATH)
    else:
        ml_model["pipeline"] = None
        logger.warning("Model TIDAK ditemukan di %s. Jalankan src/train.py dahulu.", MODEL_PATH)
    yield
    ml_model.clear()


app = FastAPI(
    title="Used Car Price Prediction API",
    description="Estimasi harga wajar mobil bekas (INR Lakh) berbasis Random Forest.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Enum & schema validasi (Pydantic) -- mencakup tipe, rentang, dan enum
# ---------------------------------------------------------------------------
class FuelType(str, Enum):
    cng = "CNG"
    diesel = "Diesel"
    electric = "Electric"
    lpg = "LPG"
    petrol = "Petrol"


class Transmission(str, Enum):
    automatic = "Automatic"
    manual = "Manual"


class OwnerType(str, Enum):
    first = "First"
    second = "Second"
    third = "Third"
    fourth_plus = "Fourth & Above"


class Location(str, Enum):
    ahmedabad = "Ahmedabad"
    bangalore = "Bangalore"
    chennai = "Chennai"
    coimbatore = "Coimbatore"
    delhi = "Delhi"
    hyderabad = "Hyderabad"
    jaipur = "Jaipur"
    kochi = "Kochi"
    kolkata = "Kolkata"
    mumbai = "Mumbai"
    pune = "Pune"


class CarFeaturesIn(BaseModel):
    name: str = Field(..., min_length=2, max_length=120,
                       description="Nama & model mobil, mis. 'Hyundai Creta 1.6 CRDi SX'",
                       examples=["Hyundai Creta 1.6 CRDi SX Option"])
    location: Location
    year: int = Field(..., ge=1990, le=2026, description="Tahun edisi/pembuatan mobil")
    kilometers_driven: float = Field(..., ge=0, le=1_000_000,
                                      description="Total kilometer terpakai")
    fuel_type: FuelType
    transmission: Transmission
    owner_type: OwnerType
    mileage: Optional[float] = Field(None, ge=0, le=50, description="kmpl atau km/kg")
    engine: Optional[float] = Field(None, ge=50, le=8000, description="cc")
    power: Optional[float] = Field(None, ge=10, le=800, description="bhp")
    seats: Optional[float] = Field(None, ge=2, le=10)

    model_config = {
        "json_schema_extra": {
            "examples": [{
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
            }]
        }
    }


class PredictionOut(BaseModel):
    predicted_price_lakh_inr: float
    confidence_interval_lakh_inr: list[float]
    model_name: str
    latency_ms: float


class HealthOut(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["info"])
def root():
    """Info layanan."""
    return {
        "service": "Used Car Price Prediction API",
        "version": "1.0.0",
        "endpoint_prediksi": "/predict-harga",
        "dokumentasi": "/docs",
    }


@app.get("/health", response_model=HealthOut, tags=["info"])
def health():
    """Status server + apakah model berhasil dimuat."""
    return HealthOut(status="ok", model_loaded=ml_model.get("pipeline") is not None)


@app.post("/predict-harga", response_model=PredictionOut, tags=["prediksi"])
def predict_harga(payload: CarFeaturesIn):
    pipeline = ml_model.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat di server.")

    start = time.perf_counter()

    row = pd.DataFrame([{
        "Name": payload.name,
        "Location": payload.location.value,
        "Year": payload.year,
        "Kilometers_Driven": payload.kilometers_driven,
        "Fuel_Type": payload.fuel_type.value,
        "Transmission": payload.transmission.value,
        "Owner_Type": payload.owner_type.value,
        "Mileage": f"{payload.mileage} kmpl" if payload.mileage is not None else None,
        "Engine": f"{payload.engine} CC" if payload.engine is not None else None,
        "Power": f"{payload.power} bhp" if payload.power is not None else None,
        "Seats": payload.seats,
    }])

    try:
        pred = float(pipeline.predict(row)[0])
    except Exception as exc:  # pragma: no cover
        logger.exception("Gagal melakukan prediksi")
        raise HTTPException(status_code=500, detail=f"Prediksi gagal: {exc}") from exc

    pred = max(pred, 0.1)  # harga tidak boleh negatif/nol
    latency_ms = (time.perf_counter() - start) * 1000

    # Interval kasar +-15% sebagai proxy ketidakpastian (bukan interval statistik formal).
    ci = [round(pred * 0.85, 2), round(pred * 1.15, 2)]

    logger.info("Prediksi: %s (%s, %s km) -> %.2f Lakh [%.1f ms]",
                payload.name, payload.year, payload.kilometers_driven, pred, latency_ms)

    return PredictionOut(
        predicted_price_lakh_inr=round(pred, 2),
        confidence_interval_lakh_inr=ci,
        model_name=type(pipeline.named_steps["model"]).__name__,
        latency_ms=round(latency_ms, 2),
    )
