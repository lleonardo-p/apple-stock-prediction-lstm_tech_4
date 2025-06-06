from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import pandas as pd
import os
from datetime import date
from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from starlette.responses import Response


app = FastAPI()

# Libera todas as origens (bypass total)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Aceita requisições de qualquer domínio
    allow_credentials=True,
    allow_methods=["*"],  # Aceita todos os métodos (GET, POST, etc)
    allow_headers=["*"],  # Aceita todos os headers
)

# Variáveis globais para o modelo e scaler
model: torch.nn.Module = None
scaler: MinMaxScaler = None
WINDOW_SIZE = 20

# Métricas Prometheus
REQUEST_COUNT = Counter("predict_requests_total", "Total de requisições de predição")
REQUEST_EXCEPTIONS = Counter("predict_request_exceptions", "Exceções em requisições de predição")
REQUEST_LATENCY = Histogram("predict_request_duration_seconds", "Duração das requisições de predição")
LAST_PREDICTION = Gauge("last_prediction_value", "Último valor previsto pelo modelo")

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def get_last_sequence(window_size: int = 20) -> np.ndarray:
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

    return df['valor'].values[::-1]  # Ordem cronológica

def insert_prediction(predicted_value: float):
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL)

    next_business_day = pd.date_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=1, freq="B")[0].date()

    update_query = text("""
        UPDATE apple_stonks
        SET valor_previsto = :valor
        WHERE date = :date AND is_predict = true
    """)

    insert_query = text("""
        INSERT INTO apple_stonks (date, valor_previsto, is_predict)
        VALUES (:date, :valor, true)
    """)

    with engine.connect() as conn:
        result = conn.execute(update_query, {"date": next_business_day, "valor": predicted_value})
        if result.rowcount == 0:
            conn.execute(insert_query, {"date": next_business_day, "valor": predicted_value})
        conn.commit()

@app.get("/predict")
@REQUEST_LATENCY.time()
def predict():
    REQUEST_COUNT.inc()
    try:
        sequence = get_last_sequence(WINDOW_SIZE)
        data = np.array(sequence).reshape(-1, 1)
        data_scaled = scaler.transform(data)
        input_tensor = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            prediction = model(input_tensor).item()

        prediction_inverse = float(scaler.inverse_transform([[prediction]])[0][0])

        # Atualiza métrica Prometheus com o valor previsto
        LAST_PREDICTION.set(prediction_inverse)

        insert_prediction(prediction_inverse)

        return {"valor_previsto": prediction_inverse}
    except Exception as e:
        REQUEST_EXCEPTIONS.inc()
        return {"error": str(e)}

def run_api(m, s):
    global model, scaler
    model = m
    scaler = s
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
