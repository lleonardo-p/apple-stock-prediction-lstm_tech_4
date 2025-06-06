FROM python:3.10-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos do host para o container
COPY main.py .
COPY modules/ /app/modules/

# Instala as dependências necessárias
RUN pip install --no-cache-dir \
    yfinance \
    SQLAlchemy \
    psycopg2-binary \
    optuna \
    mlflow \
    scikit-learn \
    numpy \
    torch \
    prometheus_client \
    starlette \
    pandas

# Comando para rodar o script principal
CMD ["python", "main.py"]
