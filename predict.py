# predict.py
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os

# Caminhos dos arquivos
MODEL_PATH = "models/sgd_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Inicializa FastAPI
app = FastAPI()

# Configura CORS (para permitir chamadas do site)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega modelo e scaler
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    raise FileNotFoundError("Modelo ou scaler não encontrados em 'models/'.")

# Mapeamento das classes
label_map_num2str = {0: "CONFIRMED", 1: "CANDIDATE", 2: "FALSE POSITIVE"}
label_map_str2num = {v: k for k, v in label_map_num2str.items()}

# Rota inicial
@app.get("/redmoons")
def home():
    return {"msg": "API de Exoplanetas pronta 🚀"}

# Endpoint de predição
@app.post("/predict")
def predict(
    koi_prad: float = Body(...),
    koi_period: float = Body(...),
    koi_steff: float = Body(...),
    koi_srad: float = Body(...)
):
    # Monta array de features
    X_new = np.array([[koi_prad, koi_period, koi_steff, koi_srad]])

    # Normaliza usando o scaler treinado
    X_new_scaled = scaler.transform(X_new)

    # Faz a predição
    pred = model.predict(X_new_scaled)
    class_name = label_map_num2str[pred[0]]

    return {"prediction": class_name}

# Endpoint de feedback (treinamento incremental)
@app.post("/feedback")
def feedback(
    koi_prad: float = Body(...),
    koi_period: float = Body(...),
    koi_steff: float = Body(...),
    koi_srad: float = Body(...),
    real_label: str = Body(...)
):
    X_new = np.array([[koi_prad, koi_period, koi_steff, koi_srad]])
    X_new_scaled = scaler.transform(X_new)
    y_new = np.array([label_map_str2num[real_label.upper()]])

    # Treinamento incremental
    model.partial_fit(X_new_scaled, y_new, classes=np.array([0,1,2]))

    # Salvar modelo atualizado
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return {"msg": f"Feedback incorporado ao modelo! Classe: {real_label.upper()}"}
