import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

# Chargement du modèle
with open("model.pkl", "rb") as f:
    pipeline = pickle.load(f)

model       = pipeline["model"]
scaler      = pipeline["scaler"]
le_category = pipeline["label_encoder_category"]
le_job      = pipeline["label_encoder_job"]
features    = pipeline["feature_names"]

# Application FastAPI
app = FastAPI(
    title="API Détection de Fraude Bancaire",
    description="Prédit si une transaction est frauduleuse via un modèle Random Forest.",
    version="1.0.0"
)

# Schéma de la requête
class Transaction(BaseModel):
    trans_date_trans_time: str   
    category: str                
    amt: float                   
    gender: str                  
    lat: float
    long: float
    city_pop: int
    job: str                     
    dob: str                     
    merch_lat: float
    merch_long: float

    class Config:
        json_schema_extra = {
            "example": {
                "trans_date_trans_time": "2019-06-21 22:15:00",
                "category": "shopping_net",
                "amt": 349.99,
                "gender": "M",
                "lat": 36.0788,
                "long": -81.1781,
                "city_pop": 3495,
                "job": "Psychologist, counselling",
                "dob": "1988-03-09",
                "merch_lat": 36.011293,
                "merch_long": -82.048315
            }
        }

# Preprocessing identique au notebook
def preprocess(t: Transaction) -> np.ndarray:
    trans_dt = pd.to_datetime(t.trans_date_trans_time)
    dob_dt   = pd.to_datetime(t.dob)

    # Features temporelles
    trans_hour  = trans_dt.hour
    trans_day   = trans_dt.dayofweek
    trans_month = trans_dt.month

    # Age du client
    age = (trans_dt - dob_dt).days // 365

    # Distance géographique client-marchand
    geo_distance = np.sqrt(
        (t.lat - t.merch_lat) ** 2 + (t.long - t.merch_long) ** 2
    )

    # Encodage genre
    gender_enc = 1 if t.gender == "M" else 0

    # Encodage category
    try:
        category_enc = int(le_category.transform([t.category])[0])
    except ValueError:
        category_enc = 0

    # Encodage job
    try:
        job_enc = int(le_job.transform([t.job])[0])
    except ValueError:
        job_enc = 0

    # Construction du DataFrame dans le bon ordre
    row = pd.DataFrame([{
        "amt"          : t.amt,
        "lat"          : t.lat,
        "long"         : t.long,
        "city_pop"     : t.city_pop,
        "merch_lat"    : t.merch_lat,
        "merch_long"   : t.merch_long,
        "category"     : category_enc,
        "gender"       : gender_enc,
        "job"          : job_enc,
        "trans_hour"   : trans_hour,
        "trans_day"    : trans_day,
        "trans_month"  : trans_month,
        "age"          : age,
        "geo_distance" : geo_distance,
    }])

    # Réordonner exactement comme les features du modèle
    row = row.reindex(columns=features, fill_value=0)

    # Normalisation avec le scaler entraîné
    row_scaled = scaler.transform(row)

    return row_scaled


# Endpoints
@app.get("/", tags=["Accueil"])
def root():
    return {
        "message" : "API Détection de Fraude — opérationnelle",
        "docs"    : "/docs",
        "predict" : "/predict"
    }


@app.get("/health", tags=["Santé"])
def health():
    return {"status": "ok"}


@app.post("/predict", tags=["Prédiction"])
def predict(transaction: Transaction):
    try:
        X = preprocess(transaction)

        prediction  = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])

        # Niveau de risque selon la probabilité
        if probability < 0.3:
            risk = "FAIBLE"
        elif probability < 0.7:
            risk = "MOYEN"
        else:
            risk = "ÉLEVÉ"

        return {
            "is_fraud"          : prediction,
            "fraud_probability" : round(probability, 4),
            "risk_level"        : risk,
            "transaction_amount": transaction.amt,
            "category"          : transaction.category,
            "message"           : "Transaction frauduleuse détectée" if prediction == 1 else "Transaction légitime"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
