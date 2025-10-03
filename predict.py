import joblib
import numpy as np
from fastapi import FastAPI, Body
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import os
from fastapi.middleware.cors import CORSMiddleware

# Caminho do modelo
MODEL_PATH = "models/sgd_model.pkl"

# Mapeamento das classes
label_map_num2str = {0: "CONFIRMED", 1: "CANDIDATE", 2: "FALSE POSITIVE"}
label_map_str2num = {v:k for k,v in label_map_num2str.items()}

# Inicializa o app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos HTTP
    allow_headers=["*"],  # Permite todos os cabeçalhos
)

# Carrega ou cria modelo
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    # Modelo SGDClassifier inicial (treinar posteriormente com dados existentes)
    model = SGDClassifier()
    # Treinamento inicial mínimo para que partial_fit funcione
    import numpy as np
    model.partial_fit(np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3]]),
                      np.array([0,1,2]),
                      classes=np.array([0,1,2]))

# Escalador (usar mesmo para previsões e feedback)
scaler = StandardScaler()

@app.get("/redmoons")
def home():
    return {"msg": "API de Exoplanetas pronta 🚀"}

@app.post("/predict")
def predict(
    koi_prad: float = Body(...),
    koi_period: float = Body(...),
    koi_steff: float = Body(...),
    koi_srad: float = Body(...)
):
    X_new = np.array([[koi_prad, koi_period, koi_steff, koi_srad]])
    X_new_scaled = scaler.transform(X_new)  # ideal: usar scaler treinado com dados reais
    pred = model.predict(X_new_scaled)
    class_name = label_map_num2str[pred[0]]
    print("Prediction:", class_name)
    return {"prediction": class_name}

@app.post("/feedback")
def feedback(
    koi_prad: float = Body(...),
    koi_period: float = Body(...),
    koi_steff: float = Body(...),
    koi_srad: float = Body(...),
    real_label: str = Body(...)
):
    X_new = np.array([[koi_prad, koi_period, koi_steff, koi_srad]])
    X_new_scaled = scaler.fit_transform(X_new)  # ideal: usar scaler treinado com dados reais
    y_new = np.array([label_map_str2num[real_label.upper()]])

    # Treinamento incremental
    model.partial_fit(np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3]]), np.array([0,1,2]), classes=np.array([0,1,2]))

    # Salvar modelo atualizado
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return {"msg": f"Feedback incorporado ao modelo! Classe: {real_label.upper()}"}
