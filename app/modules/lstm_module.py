# lstm_module.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import mlflow.pytorch
import optuna
import joblib
from typing import Tuple
from sqlalchemy import create_engine

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def load_data() -> pd.DataFrame:
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL)

    query = """
        SELECT date AS data, valor
        FROM apple_stonks
        WHERE is_predict = false
        ORDER BY date ASC
    """

    df = pd.read_sql_query(query, con=engine)
    print(f"DF Shape: {df.shape}")
    print(f"DF Head: {df.head(1)}")
    return df

def preprocess_data(df: pd.DataFrame, window_size: int = 20,
                    test_size: float = 0.2, val_size: float = 0.1
                    ) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, MinMaxScaler]:
    series = df['valor'].values
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series.reshape(-1, 1))

    X, y = [], []
    for i in range(window_size, len(scaled_series)):
        X.append(scaled_series[i - window_size:i])
        y.append(scaled_series[i])

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]
    val_size_int = int(len(train_idx) * val_size)
    val_idx = train_idx[-val_size_int:]
    train_idx = train_idx[:-val_size_int]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

    return train_loader, val_loader, X_test, y_test, scaler

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          epochs: int, lr: float, patience: int) -> None:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = np.inf
    counter = 0

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                output = model(x_val)
                val_loss += criterion(output, y_val).item()

        val_loss /= len(val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

def evaluate(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor) -> Tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze()
        y_true = y_test.squeeze()
        mae = torch.mean(torch.abs(y_pred - y_true)).item()
        rmse = torch.sqrt(torch.mean((y_pred - y_true)**2)).item()
        mape = torch.mean(torch.abs((y_pred - y_true) / (y_true + 1e-8))).item()
    return mae, rmse, mape

def objective(trial: optuna.Trial, train_loader: DataLoader, val_loader: DataLoader,
              X_test: torch.Tensor, y_test: torch.Tensor) -> float:
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    epochs = trial.suggest_int("epochs", 10, 100)
    patience = trial.suggest_int("patience", 3, 10)

    model = LSTMModel(1, hidden_size, num_layers, dropout)

    with mlflow.start_run(nested=True):
        train(model, train_loader, val_loader, epochs, lr, patience)
        mae, rmse, mape = evaluate(model, X_test, y_test)

        mlflow.log_params({
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "lr": lr,
            "dropout": dropout,
            "epochs": epochs,
            "patience": patience
        })
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "mape": mape})
        return mae

def run_optuna_study(train_loader: DataLoader, val_loader: DataLoader,
                      X_test: torch.Tensor, y_test: torch.Tensor, n_trials: int = 30) -> optuna.trial.FrozenTrial:
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, X_test, y_test), n_trials=n_trials)
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  MAE: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    return best_trial

def run_model(window_size: int = 20, test_size: float = 0.2,
              val_size: float = 0.1, n_trials: int = 30) -> Tuple[nn.Module, MinMaxScaler]:
    mlflow.set_experiment("LSTM_Stock_Prediction")
    with mlflow.start_run():
        df = load_data()
        train_loader, val_loader, X_test, y_test, scaler = preprocess_data(df, window_size, test_size, val_size)
        best_trial = run_optuna_study(train_loader, val_loader, X_test, y_test, n_trials=n_trials)

        best_model = LSTMModel(
            input_size=1,
            hidden_size=best_trial.params["hidden_size"],
            num_layers=best_trial.params["num_layers"],
            dropout=best_trial.params["dropout"]
        )

        train(
            best_model, train_loader, val_loader,
            best_trial.params["epochs"],
            best_trial.params["lr"],
            best_trial.params["patience"]
        )

        # Salva modelo e scaler em disco
        torch.save(best_model.state_dict(), "lstm_model.pt")
        joblib.dump(scaler, "scaler.save")

        mlflow.pytorch.log_model(best_model, "model")
        return best_model, scaler
