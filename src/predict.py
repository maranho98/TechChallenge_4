import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join('models','model_pipeline.joblib')

def load_model(path=MODEL_PATH):
    return joblib.load(path)

def predict_df(df, model=None):
    if model is None:
        model = load_model()
        preds = model.predict(df)
        probs = model.predict_proba(df)
    return preds, probs

if __name__ == '__main__':
 # exemplo rápido
    sample = pd.read_csv(os.path.join('data','Obesity.csv')).head(5).drop(columns=['Obesity'])
    m = load_model()
    p, pr = predict_df(sample, m)
    print(p)