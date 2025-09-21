import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI

model = joblib.load("models/random_forest.pkl")

app = FastAPI()

@app.get("/redmoons")
def home():
    return {"msg": "API de Exoplanetas pronta 🚀"}

@app.post("/predict")
def predict(koi_prad : float, koi_period : float, koi_steff : float, koi_srad : float):

    # Exemplo: planeta com raio=1.2, período=50 dias, estrela com T=5800K e raio=1.0
    X_new = np.array([[koi_prad, koi_period, koi_steff, koi_srad]])
    pred = model.predict(X_new)
    print("Previsão:", "CONFIRMED" if pred[0] == 1 else "NÃO CONFIRMADO")
    return {"prediction": int(pred)}

