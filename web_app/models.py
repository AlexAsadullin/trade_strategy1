# web_app/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime
from web_app.database import Base
from datetime import datetime

class RequestHistory(Base):
    __tablename__ = "request_history"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    request_type = Column(String)  # CSV, HTML, Predict
    created_at = Column(DateTime, default=datetime.utcnow)
