import os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = FastAPI()

# Métricas Prometheus
mae_gauge = Gauge("model_mae", "Mean Absolute Error")
rmse_gauge = Gauge("model_rmse", "Root Mean Square Error")
mape_gauge = Gauge("model_mape", "Mean Absolute Percentage Error")

def fetch_predictions():
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL)

    query = """
        SELECT valor, valor_previsto
        FROM apple_stonks
        WHERE is_predict = true AND valor IS NOT NULL AND valor_previsto IS NOT NULL
    """
    return pd.read_sql(query, engine)

@app.get("/metrics")
def metrics():
    df = fetch_predictions()

    if df.empty:
        return Response(status_code=204)

    y_true = df["valor"].values
    y_pred = df["valor_previsto"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))

    # Atualiza métricas
    mae_gauge.set(mae)
    rmse_gauge.set(rmse)
    mape_gauge.set(mape)

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def run_metrics_server():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9100)

if __name__ == "__main__":
    run_metrics_server()
