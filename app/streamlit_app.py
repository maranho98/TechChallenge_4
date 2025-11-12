import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import aplicar_mapeamentos
import warnings
warnings.filterwarnings('ignore')

# Cria as abas
aba1, aba2 = st.tabs(["🏋️‍♂️ Previsor de Obesidade", "📊 Painel Analítico"])

# ===================== ABA 1 - MODELO PREDITIVO =====================
with aba1:
    # Caminhos e cache
    MODEL_PATH = os.path.join('models', 'model_pipeline.joblib')

    @st.cache_resource
    def load_model():
        loaded = joblib.load(MODEL_PATH)
        if isinstance(loaded, dict):
            return loaded.get('model', loaded), loaded.get('columns', None)
        return loaded, None

    # Configurações de página
    st.set_page_config(page_title="Previsor de Obesidade", page_icon="🏋️‍♂️", layout="centered")

    st.title("🏋️‍♂️ Previsor de Obesidade - Tech Challenge 4")
    st.write("""
    Este aplicativo utiliza um modelo de **Machine Learning (XGBoost)** para prever o nível de obesidade 
    com base em características individuais e hábitos alimentares.
    """)

    # Entradas do usuário
    with st.expander("Inserir dados do indivíduo", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            Gender = st.selectbox('Gender', ['male', 'female'])
            Age = st.number_input('Age', 1, 120, 30)
            Height = st.number_input('Height (m)', 1.0, 2.5, 1.70)
            Weight = st.number_input('Weight (kg)', 20.0, 300.0, 70.0)
            family_history = st.selectbox('Family History (histórico familiar de obesidade)', ['yes', 'no'])
            FAVC = st.selectbox('FAVC (consumo de fast-food)', ['yes', 'no'])
            FCVC = st.number_input('FCVC (freq. consumo de vegetais)', 1, 5, 3)
            NCP = st.number_input('NCP (número de refeições por dia)', 1, 5, 3)

        with col2:
            CH2O = st.number_input('CH2O (litros de água/dia)', 0.0, 10.0, 2.0)
            SCC = st.selectbox('SCC (consome refrigerante?)', ['yes', 'no'])
            FAF = st.number_input('FAF (freq. atividade física)', 0, 5, 2)
            TUE = st.number_input('TUE (tempo em telas - horas/dia)', 0, 24, 2)
            SMOKE = st.selectbox('SMOKE (fuma?)', ['yes', 'no'])
            CAEC = st.selectbox('CAEC (álcool)', ['no', 'sometimes', 'frequently', 'always'])
            CALC = st.selectbox('CALC (consumo calórico extra)', ['no', 'sometimes', 'frequently', 'always'])
            MTRANS = st.selectbox('MTRANS (meio de transporte)', ['automobile', 'motorbike', 'bike', 'public_transportation', 'walking'])

        # Montagem do DataFrame
        row = pd.DataFrame([{
            'Gender': Gender,
            'Age': Age,
            'Height': Height,
            'Weight': Weight,
            'family_history': family_history,
            'FAVC': FAVC,
            'FCVC': FCVC,
            'NCP': NCP,
            'CH2O': CH2O,
            'SCC': SCC,
            'FAF': FAF,
            'TUE': TUE,
            'SMOKE': SMOKE,
            'CAEC': CAEC,
            'CALC': CALC,
            'MTRANS': MTRANS
        }])

        # Calcula BMI
        row['BMI'] = row['Weight'] / (row['Height'] ** 2)
        st.metric(label="Índice de Massa Corporal (BMI)", value=f"{row['BMI'].iloc[0]:.2f}")

        # Aplica o mesmo mapeamento do treino
        row_mapped = aplicar_mapeamentos(row)

        # Converte para numérico e preenche nulos
        for c in row_mapped.columns:
            if row_mapped[c].dtype == 'object':
                row_mapped[c] = pd.to_numeric(row_mapped[c], errors='coerce')
        row_mapped = row_mapped.fillna(0)

        # Botão de previsão
        if not os.path.exists(MODEL_PATH):
            st.error("Modelo não encontrado. Rode o script `src/train.py` para gerar `model_pipeline.joblib`.")
        else:
            model, model_columns = load_model()

            if st.button("Gerar Previsão"):
                try:
                    # Normaliza nomes e ordem das colunas
                    row_mapped.columns = row_mapped.columns.str.lower().str.strip()
                    expected_cols = model_columns if model_columns else model.get_booster().feature_names
                    row_mapped = row_mapped.reindex(columns=expected_cols, fill_value=0)

                    # Predição
                    pred_num = model.predict(row_mapped)[0]
                    probs = model.predict_proba(row_mapped)[0]

                    # Classes
                    classes = [
                        'Insufficient_Weight',
                        'Normal_Weight',
                        'Overweight_Level_I',
                        'Overweight_Level_II',
                        'Obesity_Type_I',
                        'Obesity_Type_II',
                        'Obesity_Type_III'
                    ]
                    pred_label = classes[int(pred_num)]

                    # Probabilidades
                    prob_df = pd.DataFrame({
                        'class': classes,
                        'probabilidade': probs
                    }).sort_values('probabilidade', ascending=False)

                    # Exibição
                    st.success(f"Classificação prevista: **{pred_label.replace('_', ' ')}**")
                    st.bar_chart(prob_df.set_index('class'))
                    st.dataframe(prob_df)

                    st.markdown(f"""
                    ### Interpretação:
                    O modelo previu que o indivíduo está classificado como **{pred_label.replace('_', ' ')}**, 
                    com base nas informações fornecidas sobre hábitos e perfil físico.
                    """)

                except Exception as e:
                    st.error(f"Erro ao gerar previsão: {e}")
                    st.info("Verifique se as colunas e tipos de dados estão compatíveis com o modelo treinado.")
            else:
                st.info("Preencha os dados acima e clique em **Gerar Previsão** para ver o resultado.")
            pass


# ===================== ABA 2 - PAINEL ANALÍTICO =====================
with aba2:
    st.title("📊 Painel Analítico - Insights sobre Obesidade")
    st.write("""
    Este painel apresenta uma análise exploratória dos dados de obesidade, 
    destacando os principais padrões e fatores associados ao ganho de peso.
    """)
    
    # Carrega o dataset
    df = pd.read_csv("data/Obesity.csv")
    df = aplicar_mapeamentos(df)

    # Cria o BMI se não existir
    if 'BMI' not in df.columns and 'Weight' in df.columns and 'Height' in df.columns:
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    # Renomeia coluna alvo, caso necessário
    if 'obesity' not in df.columns:
        target_col = next((c for c in df.columns if c.lower() == 'obesity'), None)
        if target_col:
            df.rename(columns={target_col: 'obesity'}, inplace=True)

    st.subheader("1️⃣ Distribuição dos Níveis de Obesidade")
    fig, ax = plt.subplots()
    sns.countplot(x='obesity', data=df, palette='coolwarm', ax=ax)
    ax.set_xlabel("Nível de Obesidade")
    ax.set_ylabel("Quantidade de Pessoas")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("2️⃣ Relação entre Idade e IMC")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Age', y='BMI', hue='obesity', data=df, palette='Spectral', ax=ax)
    ax.set_title("Dispersão de Idade x BMI por Classe de Obesidade")
    st.pyplot(fig)

    st.subheader("3️⃣ Influência da Atividade Física (FAF) na Obesidade")
    fig, ax = plt.subplots()
    sns.boxplot(x='obesity', y='FAF', data=df, palette='coolwarm', ax=ax)
    ax.set_title("Distribuição de Atividade Física por Classe de Obesidade")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("4️⃣ Correlação entre Hidratação (CH2O) e IMC")
    fig, ax = plt.subplots()
    sns.regplot(x='CH2O', y='BMI', data=df, scatter_kws={'alpha':0.4}, line_kws={'color':'red'}, ax=ax)
    ax.set_title("Correlação entre Consumo de Água e Índice de Massa Corporal (BMI)")
    st.pyplot(fig)

    st.subheader("5️⃣ Hábitos Alimentares e Obesidade (FAVC)")
    fig, ax = plt.subplots()
    sns.countplot(x='FAVC', hue='obesity', data=df, palette='Set2', ax=ax)
    ax.set_xlabel("Consumo Frequente de Fast-Food (FAVC)")
    ax.set_ylabel("Contagem de Pessoas")
    plt.legend(title="Nível de Obesidade", bbox_to_anchor=(1,1))
    st.pyplot(fig)

    st.markdown("""
    ---
    ### 💡 Principais Insights:
    - O **IMC (BMI)** aumenta progressivamente com a idade, especialmente após os 30 anos.
    - Pessoas com **baixo nível de atividade física (FAF)** concentram-se nas classes de obesidade tipo I e II.
    - Um **maior consumo de água (CH2O)** tende a estar associado a um IMC mais equilibrado.
    - O consumo frequente de **fast-food (FAVC = yes)** apresenta forte correlação com sobrepeso e obesidade tipo I.
    - Há um claro padrão entre **histórico familiar de obesidade** e risco aumentado nas classes mais elevadas.
    """)
