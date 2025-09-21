import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI

model = joblib.load("models/random_forest.pkl")
label_map = {
    0: "CONFIRMED",
    1: "CANDIDATE",
    2: "FALSE POSITIVE"
}


app = FastAPI()

@app.get("/redmoons")
def home():
    return {"msg": "API de Exoplanetas pronta 🚀"}

@app.post("/predict")
def predict(koi_prad : float, koi_period : float, koi_steff : float, koi_srad : float):

    # Exemplo: planeta com raio=1.2, período=50 dias, estrela com T=5800K e raio=1.0
    X_new = np.array([[koi_prad, koi_period, koi_steff, koi_srad]])
    pred = model.predict(X_new)
    class_name = label_map[pred[0]]
    print("Prediction: ", class_name)
    return {"prediction": class_name}

