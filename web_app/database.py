from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv

load_dotenv()
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
password = quote_plus(POSTGRES_PASSWORD)

DATABASE_URL = f"postgresql://postgres:{password}@localhost:5432/InvestAppData"
# TODO: БД на серваке
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
