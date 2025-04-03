from sqlalchemy import Column, Integer, String, Float, DateTime
from web_app.database import Base
from datetime import datetime
from pydantic import BaseModel

class RequestHistory(Base):
    __tablename__ = "request_history"

    request_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    ticker = Column(String, index=True)
    days_back = Column(Integer, index=True)
    request_type = Column(String, index=True)  # CSV, HTML, Predict
    figi = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.now())

# Добавим модель User
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    b_day = Column(DateTime, default=datetime.now())
    country = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.now())

class UserCreate(BaseModel):
    email: str
    password: str
    birth_date: str
    country: str