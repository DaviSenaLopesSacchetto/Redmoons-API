import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance(model, feature_names, save_path="../outputs/feature_importance.png"):
    """
    Plota e salva um gráfico da importância das features do modelo.
    
    model: modelo treinado (ex: RandomForestClassifier)
    feature_names: lista de nomes das colunas de entrada
    save_path: caminho para salvar a imagem
    """
    importances = model.feature_importances_
    sns.barplot(x=importances, y=feature_names)
    plt.title("Importância das Features")
    plt.xlabel("Importância")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico de importância das features salvo em {save_path}")

def save_dataframe(df, file_path):
    """
    Salva um DataFrame em CSV.
    
    df: DataFrame do pandas
    file_path: caminho do arquivo CSV
    """
    df.to_csv(file_path, index=False)
    print(f"DataFrame salvo em {file_path}")

def load_dataframe(file_path):
    """
    Carrega um CSV para DataFrame.
    
    file_path: caminho do CSV
    """
    df = pd.read_csv(file_path)
    return df

def plot_confusion_matrix(cm, labels, save_path="../outputs/confusion_matrix.png"):
    """
    Plota e salva a matriz de confusão.
    
    cm: matriz de confusão (array 2D)
    labels: lista de nomes das classes
    save_path: caminho para salvar a imagem
    """
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Matriz de confusão salva em {save_path}")
