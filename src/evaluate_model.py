import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("data/processed/kepler_processed.csv")


X = data[["koi_prad", "koi_period", "koi_steff", "koi_srad"]]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = joblib.load("models/random_forest.pkl")


y_pred = model.predict(X_test)


report = classification_report(y_test, y_pred)
print("=== Classification Report ===")
print(report)


with open("outputs/classification_report.txt", "w") as f:
    f.write(report)


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

print("Avaliação concluída! Relatório e matriz de confusão salvos em outputs/")
