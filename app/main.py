# main.py
from modules.ingestion_module import run_ingestion
from modules.lstm_module import run_model
from modules.api_module import run_api

def main():
    print("[1] Iniciando coleta de dados...")
    run_ingestion()

    print("[2] Treinando modelo...")
    model, scaler = run_model()

    print("[3] Iniciando API...")
    run_api(model, scaler)

if __name__ == "__main__":
    main()