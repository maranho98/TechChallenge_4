streamlit run app/streamlit_app.py

python -m streamlit run app/streamlit_app.py


🧩 1. Estruturação do Projeto

Por que foi feito:
Organizamos o projeto em três módulos principais para seguir uma arquitetura de pipeline de Machine Learning clara, modular e escalável. Essa divisão facilita manutenção, reuso e futuras integrações (como APIs ou dashboards).

Como foi feito:
Criamos três scripts com responsabilidades bem definidas:

preprocess.py → trata, limpa e padroniza os dados.

train.py → realiza o treino, otimização e avaliação do modelo.

streamlit_app.py → carrega o modelo final e gera previsões via interface web interativa.

Essa separação reflete boas práticas de MLOps e engenharia de dados, aproximando o projeto de um ambiente de produção real.

🧼 2. Pré-processamento (preprocess.py)

Por que foi feito:
Os modelos de Machine Learning só interpretam números, e o dataset original possuía variáveis textuais (ex: “yes/no”, “male/female”, “frequently”, “sometimes”).
Também havia o risco de dados ausentes ou inconsistentes. O pré-processamento foi essencial para garantir integridade e padronização.

Como foi feito:

load_data() – lê o dataset e calcula automaticamente o IMC (BMI) caso não exista:
BMI = Weight / (Height²)

basic_cleaning() – remove duplicatas e trata valores nulos:

Numéricos → mediana

Categóricos → moda

aplicar_mapeamentos() – converte textos em números:

“yes/no” → 1/0

“frequently/sometimes/always” → escala 0–3

“male/female” → 1/0

“automobile/walking” → 0–4

Classes de obesidade → 0–6

Essas transformações garantem que o modelo receba entradas numéricas e padronizadas tanto no treino quanto na predição.

🧠 3. Treinamento e Avaliação (train.py)

Por que foi feito:
Era necessário um modelo robusto e eficiente para lidar com sete categorias de obesidade. O XGBoostClassifier foi escolhido por seu ótimo desempenho em tarefas multiclasse e alta capacidade de generalização.

Como foi feito:

Dividimos os dados em 80% treino / 20% teste com estratificação.

Realizamos uma otimização de hiperparâmetros via RandomizedSearchCV com 5 folds de validação cruzada.

Utilizamos métricas de desempenho e validação cruzada para garantir estabilidade.

Melhores parâmetros encontrados:

{
 'subsample': 0.7,
 'n_estimators': 400,
 'max_depth': 6,
 'learning_rate': 0.2,
 'colsample_bytree': 0.9
}


Resultados obtidos:

Melhor Score CV: 0.98

Acurácia no conjunto de teste: 0.99

Cross-val mean: 0.99 ± 0.01

Relatório de Classificação (resumo):
O modelo atingiu equilíbrio quase perfeito entre precision, recall e f1-score em todas as classes, com destaque para as categorias de obesidade severa, que alcançaram 100% de acerto.

Além disso, foi gerada a Curva ROC Multiclasse (AUC > 0.98 para todas as classes) e salva em models/roc_curve_xgb.png.
O modelo final foi armazenado em models/model_pipeline.joblib.

🌐 4. Aplicação Web Interativa (streamlit_app.py)

Por que foi feito:
A intenção era permitir que qualquer usuário, mesmo sem conhecimento técnico, pudesse simular suas informações e obter uma previsão instantânea sobre seu nível de obesidade.

Como foi feito:

Criamos uma interface com Streamlit, contendo campos interativos (ex: idade, peso, altura, hábitos alimentares).

O BMI é calculado automaticamente.

Ao clicar em “Gerar Previsão”, o app:

Aplica novamente os mapeamentos numéricos;

Carrega o modelo treinado;

Executa predict() e predict_proba();

Exibe a classe prevista e um gráfico de barras com as probabilidades.

Essa interação controlada via botão evita que o modelo execute previsões automáticas a cada alteração de campo, otimizando desempenho e usabilidade.

🧱 5. Integração e Padronização

Por que foi feito:
Durante o desenvolvimento, ocorreram erros de incompatibilidade entre colunas (ex: “Height” vs “height”).
O XGBoost exige que os nomes de features no treino sejam idênticos aos da predição.

Como foi corrigido:

Padronizamos todos os nomes de colunas para minúsculas.

Aplicamos o mesmo mapeamento no treino e na predição (garantindo consistência).

Forçamos o tipo numérico das variáveis após a transformação.

📊 6. Resultados Finais
Métrica	Resultado
Melhor Score CV	0.98
Acurácia no Teste	0.99
Cross-val Mean ± Std	0.99 ± 0.01
AUC Médio	> 0.98

Variáveis mais importantes:

BMI

FAF (atividade física)

CH2O (consumo de água)

CALC (consumo calórico extra)

Histórico familiar de obesidade

Saídas geradas:

Modelo salvo: models/model_pipeline.joblib

Gráfico ROC: models/roc_curve_xgb.png

Dashboard: streamlit run app/streamlit_app.py

🚀 7. Conclusão

O projeto foi desenvolvido de ponta a ponta, simulando um pipeline de Machine Learning profissional:
da limpeza de dados até o deploy de um modelo preditivo interativo.

Com acurácia de 99% e curva ROC próxima da perfeição, o sistema demonstra alto potencial de aplicação prática em análise de saúde, nutrição e prevenção de obesidade.

A arquitetura modular, os mapeamentos consistentes e a validação rigorosa tornam este projeto replicável, escalável e pronto para produção real.

LINK PARA O MODELO EM PRODUÇÃO VIA STREAMLIT: https://tc4grupo31.streamlit.app/
LINK PARA O REPOSITÓRIO NO GITHUB: https://github.com/maranho98/TechChallenge_4
