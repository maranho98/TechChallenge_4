import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Dicionário de mapeamentos
mapeamentos = {
    # Binárias (Yes/No → 1/0)
    'yes_no': {'yes': 1, 'no': 0},

    # Frequência de consumo (CAEC, CALC)
    'frequencia': {'no': 0, 'sometimes': 1, 'frequently': 2, 'always': 3},

    # Meio de transporte
    'mtrans': {
        'automobile': 0,
        'motorbike': 1,
        'bike': 2,
        'public_transportation': 3,
        'walking': 4
    },

    # Classes alvo (Obesity)
    'obesity': {
        'insufficient_weight': 0,
        'normal_weight': 1,
        'overweight_level_i': 2,
        'overweight_level_ii': 3,
        'obesity_type_i': 4,
        'obesity_type_ii': 5,
        'obesity_type_iii': 6
    },

    # Gênero
    'gender': {'female': 0, 'male': 1}
}


def load_data(path):
    df = pd.read_csv(path)

    # 🔹 Normaliza os nomes das colunas para minúsculas
    df.columns = df.columns.str.strip().str.lower()

    # 🔹 Cria BMI se não existir
    if 'bmi' not in df.columns and {'weight', 'height'}.issubset(df.columns):
        df['bmi'] = df['weight'] / (df['height'] ** 2)

    return df


def basic_cleaning(df):
    df = df.drop_duplicates().copy()

    # Preenche numéricos com mediana
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(df[c].median())

    # Preenche categóricos com moda
    for c in df.select_dtypes(include=['object', 'category']).columns:
        df[c] = df[c].fillna(df[c].mode().iloc[0])

    return df


def aplicar_mapeamentos(df):
    df = df.copy()

    # Normaliza tudo em minúsculas
    df = df.applymap(lambda x: x.lower().strip() if isinstance(x, str) else x)

    # Mapeia colunas binárias
    for col in ['family_history', 'favc', 'scc', 'smoke']:
        if col in df.columns:
            df[col] = df[col].map(mapeamentos['yes_no'])

    # Mapeia frequência
    for col in ['caec', 'calc']:
        if col in df.columns:
            df[col] = df[col].map(mapeamentos['frequencia'])

    # Mapeia gênero
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map(mapeamentos['gender'])

    # Mapeia meio de transporte
    if 'mtrans' in df.columns:
        df['mtrans'] = df['mtrans'].map(mapeamentos['mtrans'])

    # Mapeia obesidade (target)
    if 'obesity' in df.columns:
        df['obesity'] = df['obesity'].map(mapeamentos['obesity'])

    return df
