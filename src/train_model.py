from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import os
import numpy as np

# Caminho de saída
save_path_model = "models/sgd_model.pkl"
save_path_scaler = "models/scaler.pkl"

# Criar a pasta se não existir
os.makedirs(os.path.dirname(save_path_model), exist_ok=True)

# 1️⃣ Carregar os dados
df = pd.read_csv("data/processed/kepler_processed.csv")

# 2️⃣ Definir features (X) e rótulo (y)
X = df[["koi_prad", "koi_period", "koi_steff", "koi_srad"]]
y = df["label"]

# 3️⃣ Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Criar e treinar o scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # ⬅️ agora sim

# 5️⃣ Criar e treinar o modelo
model = SGDClassifier()
model.partial_fit(X_train_scaled, y_train, classes=np.array([0,1,2]))

# 6️⃣ Salvar modelo e scaler
joblib.dump(model, save_path_model)
joblib.dump(scaler, save_path_scaler)

print("Modelo e scaler salvos com sucesso!")
