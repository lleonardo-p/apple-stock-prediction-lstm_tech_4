# ingestion_module.py
import os
import yfinance as yf
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

def run_ingestion():
    # Configs do banco
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Setup SQLAlchemy
    Base = declarative_base()

    class AppleStonk(Base):
        __tablename__ = "apple_stonks"

        id = Column(Integer, primary_key=True, autoincrement=True)
        date = Column(Date, nullable=False)
        valor = Column(Float, nullable=False)
        valor_previsto = Column(Float)
        modelo_version = Column(String)
        is_predict = Column(Boolean, default=False)
        mae = Column(Float)
        rmse = Column(Float)
        mape = Column(Float)

    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    # Obter os dados do yfinance
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="max")  # últimos 5 dias

    for index, row in hist.iterrows():
        date = index.date()
        valor = float(row["Close"])

        stonk = AppleStonk(
            date=date,
            valor=valor,
            is_predict=False
        )

        existing = session.query(AppleStonk).filter_by(date=date, is_predict=False).first()
        if not existing:
            session.add(stonk)

    session.commit()
    print("Dados da Apple salvos com sucesso.")
