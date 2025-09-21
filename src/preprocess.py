import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path: str):
    """Carrega o dataset original a partir de um CSV."""
    data = pd.read_csv(path)
    return data

def clean_data(df):
    """Seleciona features relevantes e trata valores ausentes."""
    features = ["koi_prad", "koi_period", "koi_steff", "koi_srad"]
    df = df[features + ["koi_disposition"]]

    # Preencher NaNs apenas nas colunas numéricas
    df[features] = df[features].fillna(df[features].mean())

    return df

def encode_labels(df):
    """Transforma a coluna koi_disposition em label binário (1 = CONFIRMED, 0 = outro)."""
    df["label"] = df["koi_disposition"].apply(lambda x: 1 if x == "CONFIRMED" else 0)
    X = df[["koi_prad", "koi_period", "koi_steff", "koi_srad"]]
    y = df["label"]
    return X, y

def scale_features(X):
    """Padroniza os dados para média = 0 e desvio padrão = 1."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

if __name__ == "__main__":
    # Caminho do dataset original (ajuste se precisar)
    raw_path = "G:/Meu Drive/Red Moons.AI/data/raw/cumulative.csv"
    save_path = "data/processed/kepler_processed.csv"

    # Carregar e processar
    df = load_data(raw_path)
    df = clean_data(df)
    X, y = encode_labels(df)
    X_scaled = scale_features(X)

    # Garantir que a pasta de destino existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Salvar arquivo processado
    processed = pd.DataFrame(X_scaled, columns=X.columns)
    processed["label"] = y
    processed.to_csv(save_path, index=False)

    print(f"✅ Pré-processamento concluído! Arquivo salvo em: {save_path}")
