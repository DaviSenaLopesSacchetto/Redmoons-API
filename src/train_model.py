from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import os

# Caminho de saída
save_path = "models/random_forest.pkl"

# Criar a pasta se não existir
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Salvar modelo


# 1️⃣ Carregar os dados
df = pd.read_csv("data/processed/kepler_processed.csv")

# 2️⃣ Definir features (X) e rótulo (y)
X = df[["koi_prad", "koi_period", "koi_steff", "koi_srad"]]  # colunas de entrada
y = df["label"]  # coluna de saída

# 3️⃣ Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
joblib.dump(model, save_path)
