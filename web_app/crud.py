# web_app/crud.py
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from web_app.models import User
from web_app.database import get_db

# Инициализация для хеширования паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Функция для получения пользователя по имени
def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()


# Функция для создания нового пользователя
def create_user(db: Session, username: str, password: str):
    # Хешируем пароль
    hashed_password = pwd_context.hash(password)
    new_user = User(username=username, hashed_password=hashed_password)

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user
