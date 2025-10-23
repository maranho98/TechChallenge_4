import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from preprocess import load_data, basic_cleaning, aplicar_mapeamentos
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Caminhos e diretórios
DATA_PATH = os.path.join('data', 'Obesity.csv')
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Carregamento e pré-processamento
df = load_data(DATA_PATH)
df = basic_cleaning(df)
df = aplicar_mapeamentos(df)

# Padroniza os nomes das colunas (compatível com o app)
df.columns = df.columns.str.lower().str.strip()

# Detecta a coluna de alvo (Obesity ou obesity)
target_col = next((c for c in df.columns if c.lower() == 'obesity'), None)
if target_col is None:
    raise KeyError("Coluna alvo 'Obesity' não encontrada no dataset!")

# Separa features e target
X = df.drop(columns=[target_col])
y = df[target_col]


# Split de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# Modelo base - XGBoost
xgb = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)


# Busca de hiperparâmetros
param_dist = {
    'n_estimators': [100, 200, 400],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.9, 1.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=10,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)


# Treinamento
print("Treinando modelo XGBoost...")
search.fit(X_train, y_train)

# Resultados do CV
print(f"\nMelhor score CV: {search.best_score_:.2f}")
print(f"Melhores parâmetros: {search.best_params_}")

# Melhor modelo
best_model = search.best_estimator_


# Avaliação no conjunto de teste

y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAcurácia no conjunto de teste: {acc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Validação cruzada final

scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f"\nCross-val mean: {scores.mean():.2f} | std: {scores.std():.2f}")


# Curva ROC Multiclasse
classes = np.unique(y)
y_test_bin = label_binarize(y_test, classes=classes)
y_score = best_model.predict_proba(X_test)

plt.figure(figsize=(8, 6))
for i, class_name in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Classe {class_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Falsos Positivos (FPR)')
plt.ylabel('Verdadeiros Positivos (TPR)')
plt.title('Curva ROC - Multi-Classe (XGBoost)')
plt.legend()
plt.tight_layout()

roc_path = os.path.join(MODEL_DIR, 'roc_curve_xgb.png')
plt.savefig(roc_path)
print(f"\nCurva ROC salva em: {roc_path}")


model_path = os.path.join(MODEL_DIR, 'model_pipeline.joblib')
joblib.dump({'model': best_model, 'columns': list(X.columns)}, model_path)
print(f"\nModelo XGBoost salvo em: {model_path}")