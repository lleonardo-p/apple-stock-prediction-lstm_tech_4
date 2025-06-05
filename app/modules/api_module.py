from fastapi import FastAPI
import torch
import numpy as np
import pandas as pd
import os
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

# Variáveis globais para o modelo e scaler
model: torch.nn.Module = None
scaler: MinMaxScaler = None

WINDOW_SIZE = 20

def get_last_sequence(window_size: int = 20) -> np.ndarray:
    """Busca os últimos valores reais da tabela para predição."""
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL)

    query = f"""
        SELECT valor
        FROM apple_stonks
        WHERE is_predict = false
        ORDER BY date DESC
        LIMIT {window_size}
    """

    df = pd.read_sql_query(query, engine)
    if len(df) != window_size:
        raise ValueError(f"Não há {window_size} valores suficientes para a predição.")

    return df['valor'].values[::-1]  # Inverter para ordem cronológica

@app.get("/predict")
def predict():
    try:
        sequence = get_last_sequence(WINDOW_SIZE)
        data = np.array(sequence).reshape(-1, 1)
        data_scaled = scaler.transform(data)
        input_tensor = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            prediction = model(input_tensor).item()

        prediction_inverse = scaler.inverse_transform([[prediction]])[0][0]
        return {"valor_previsto": prediction_inverse}
    except Exception as e:
        return {"error": str(e)}

def run_api(m, s):
    global model, scaler
    model = m
    scaler = s
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
