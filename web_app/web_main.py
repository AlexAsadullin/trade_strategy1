# libraries
import pandas as pd
from fastapi import FastAPI, Request, Query, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer  # OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy.sql.annotation import Annotated
from datetime import datetime
from pathlib import Path
import sys
import os
from pydantic import BaseModel

#current_directory = Path(__file__).resolve()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# project dependences
from API.api_main import app as api_app, download_csv, download_html, predict_price  # Импортируем API
# from web_app.routers import history, users
# from web_app.crud import get_user, create_user
from web_app.database import engine, SessionLocal, get_db
import web_app.web_models as models

app = FastAPI(title="Trading Web App")
models.Base.metadata.create_all(bind=engine)
db_dependency = Annotated[Session, Depends(get_db)]

# Включаем маршруты API в web-приложение
app.mount("/api", api_app)  # API/api_main доступно по host/api/ ...

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
app.mount("/static", StaticFiles(directory=frontend_path, html=True), name="static")

templates = Jinja2Templates(directory=os.path.join(Path(__file__).resolve().parent, "templates"))
# Авторизация
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")
# Хеширование паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserCreate(BaseModel):
    # id: int
    email: str
    password: str
    b_day: str
    country: str
    # created_at: str


class UserLogin(BaseModel):
    email: str
    password: str


def create_user(email: str, password: str, b_day: str, country: str, db: Session = Depends(get_db)) -> models.User:
    hashed_password = pwd_context.hash(password)
    try:
        parsed_b_day = datetime.strptime(b_day, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid birth date format. Use YYYY-MM-DD.")
    db_user = models.User(email=email, hashed_password=hashed_password, b_day=b_day, country=country)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_user(email: str, db: Session = Depends(get_db)) -> models.User | None:
    return db.query(models.User).filter(models.User.email == email).first()


def get_history(user_id: str, limit: int, db: Session = Depends(get_db)):
    return db.query(models.RequestHistory).filter(models.RequestHistory.user_id == user_id)[:limit]


@app.get("/", response_class=HTMLResponse)
async def serve_react_app():
    return FileResponse(os.path.join(frontend_path, "index.html"))


@app.post("/register")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    user_in_db = get_user(db=db, email=user.email)
    if user_in_db:
        raise HTTPException(status_code=400, detail="User already exists")
    create_user(db=db, email=user.email, password=user.password, b_day=user.b_day, country=user.country)
    return {'message': 'created successfully'}


@app.post("/login")
async def login(user: UserLogin, db: Session = Depends(get_db)):
    user_in_db = get_user(db=db, email=user.email)
    if not user_in_db or not pwd_context.verify(user.password, user_in_db.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    return {'message': f'{user.email} logged in successfully'}


@app.get("/logo", response_class=FileResponse)
async def get_logo():
    logo_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "vite.svg")
    return FileResponse(logo_path)


@app.get("/test_web")
def read_root():
    return {"message": "Trading Web App is working"}


"""
# TODO: НЕСРОЧНО ВАЖНО - фронтенд добавить и подружить с бэкэндом
@app.get("/", response_class=HTMLResponse)
async def get_welcome_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/main", response_class=HTMLResponse)
async def get_main_page(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})


@app.get("/download", response_class=HTMLResponse)
async def get_download_page(request: Request):
    # TODO: добавить сохранение в БД с историей запросов
    return templates.TemplateResponse("download.html", {"request": request})


@app.get("/plot", response_class=HTMLResponse)
async def get_plot_page(request: Request):
    # TODO: добавить сохранение в БД с историей запросов
    return templates.TemplateResponse("download.html", {"request": request})


@app.get("/ai", response_class=HTMLResponse)
async def get_ai_page(request: Request):
    # TODO: добавить сохранение в БД с историей запросов
    return templates.TemplateResponse("download.html", {"request": request})
"""

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app.web_main:app", host="0.0.0.0", port=8000, reload=True)

"""
# Основной функционал
API_BASE_URL = "http://localhost:8000"

@app.get("/get_table")
def get_table(
        tinkoff_days_back: int = Query(..., description="Количество дней истории"),
        tinkoff_figi: str = Query(..., description="FIGI бумаги"),
        curr_interval: str = Query(..., description="Интервал (строка)")
):
    return download_csv(tinkoff_days_back, tinkoff_figi, curr_interval)

@app.get("/get_chart")
def get_chart(
        tinkoff_days_back: int = Query(..., description="Количество дней истории"),
        tinkoff_figi: str = Query(..., description="FIGI бумаги"),
        curr_interval: str = Query(..., description="Интервал (строка)")
):
    return download_html(tinkoff_days_back, tinkoff_figi, curr_interval)

@app.get("/ai")
def ai(
        tinkoff_days_back: int = Query(..., description="Количество дней истории"),
        tinkoff_figi: str = Query(..., description="FIGI бумаги"),
        curr_interval: str = Query(..., description="Интервал (строка)"),
):
    return predict_price(tinkoff_days_back, tinkoff_figi, curr_interval)"""
