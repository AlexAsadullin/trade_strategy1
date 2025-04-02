from sqlalchemy.orm import Session
from web_app import web_models as models
from web_app.database import SessionLocal
from passlib.context import CryptContext

# Хеширование паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Функция для получения сессии БД (вызов через `Depends(get_db)`)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Создание пользователя
def create_user(db: Session, email: str, password: str, b_day: str, country: str):
    hashed_password = pwd_context.hash(password)  # Хешируем пароль
    db_user = models.User(email=email, hashed_password=hashed_password, b_day=b_day, country=country)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# Получение пользователя по email
def get_user(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_history(db: Session, user_id: str, skip: int, limit: int):
    return db.query(models.RequestHistory).filter(models.RequestHistory.user_id == user_id)