# web_app/routers/history.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from web_app.database import get_db
from web_app.crud import get_history

router = APIRouter(prefix="/history", tags=["History"])

@router.get("/")
def read_history(user_id: str, db: Session = Depends(get_db), skip: int = 0, limit:int = 10):
    history = get_history(db, user_id=user_id, skip=skip, limit=limit)
    return {"history": history}
