from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from web_app.database import Base
from datetime import datetime
from pydantic import BaseModel


# Добавим модель User
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    b_day = Column(DateTime)
    country = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.now())


class RequestHistory(Base):
    __tablename__ = "request_history"

    request_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    ticker = Column(String, index=True)
    days_back = Column(Integer, index=True)
    request_type = Column(String, index=True)  # CSV, HTML, Predict
    figi = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.now())


# Модель для создания пользователя
class UserCreate(BaseModel):
    email: str
    password: str
    b_day: str
    country: str

# Модель для ответа (без пароля!)
class UserResponse(BaseModel):
    id: int
    email: str
    b_day: str
    country: str
    created_at: str

    class Config:
        orm_mode = True  # нужно для SQLAlchemy