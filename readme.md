# README.md

## 📌 Descrição do Projeto

Este projeto aplica conhecimentos de deep learning para construir uma pipeline de previsão de séries temporais utilizando um modelo LSTM. Após o treinamento, uma API é disponibilizada para realizar predições em tempo real do preço das ações de uma empresa específica — neste caso, a Apple Inc.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.10**
- **PyTorch** – framework de deep learning
- **Optuna** – otimização de hiperparâmetros
- **MLflow** – rastreamento de experimentos e versionamento de modelos
- **FastAPI** – criação da API
- **PostgreSQL** – banco de dados relacional
- **SQLAlchemy** – ORM para acesso ao banco
- **Docker & Docker Compose** – containerização e orquestração
- **yFinance** – extração de dados financeiros

---

## 🚀 Como Utilizar o Projeto

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/nomedorepo.git
cd nomedorepo
```

### 2. Configure variáveis de ambiente (Docker cuida disso no `docker-compose.yml`)

### 3. Inicie os serviços com Docker Compose
```bash
docker-compose up --build
```

Isso irá:
- Subir o banco de dados PostgreSQL
- Rodar o pipeline de ingestão de dados e treinamento
- Expor a API em `http://localhost:8000`

### 4. Acessar a API

Abra no navegador ou via `curl`:
```bash
http://localhost:8000/predict
```

---

## 📂 Estrutura Esperada
```
project-root/
├── app/
│   ├── main.py
│   └── modules/
│       ├── ingestion_module.py
│       ├── lstm_module.py
│       └── api_module.py
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## 📊 Métricas
Após o treinamento, as métricas do melhor modelo (MAE, RMSE, MAPE e hiperparâmetros) são salvas em:
```bash
metrics/best_model_metrics.json
```

Essas métricas permitem reavaliar a performance do modelo e decidir se um novo treinamento é necessário.

---

Para dúvidas ou sugestões, sinta-se à vontade para abrir uma issue ou contribuir! 🚀
