# libraries
from fastapi import FastAPI, Request, Query, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pathlib import Path
from sqlalchemy.orm import Session
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# project dependences
from API.api_main import app as api_app, download_csv, download_html, predict_price # Импортируем API
from web_app.routers import history, users
from web_app.crud import get_db, get_user, create_user
from web_app.web_models import UserCreate

app = FastAPI(title="Trading Web App")
# Включаем маршруты API в web-приложение
app.mount("/api", api_app)  # Теперь API доступно по /api
app.mount("/static",
          StaticFiles(directory=os.path.join(Path(__file__).resolve().parent, "static")),
          name="static")
templates = Jinja2Templates(directory=os.path.join(Path(__file__).resolve().parent, "templates"))
# Подключаем роутеры
app.include_router(users.router)
app.include_router(history.router)
# Авторизация
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    user_in_db = get_user(db, user.email)
    if user_in_db:
        raise HTTPException(status_code=400, detail="User already exists")
    create_user(db, user.email, user.password, user.birth_date, user.country)
    return {"message": "User registered successfully"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user(db, form_data.username)
    if not user or user.hashed_password != form_data.password:  # Добавить хеширование
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return {"access_token": user.username, "token_type": "bearer"}

@app.get("/test_web")
def read_root():
    return {"message": "Welcome to the Trading Web App"}

@app.get("/", response_class=HTMLResponse)
async def get_welcome_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def get_login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def get_register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/main", response_class=HTMLResponse)
async def get_main_page(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/download", response_class=HTMLResponse)
async def get_download_page(request: Request):
    return templates.TemplateResponse("download.html", {"request": request})

@app.get("/plot", response_class=HTMLResponse)
async def get_plot_page(request: Request):
    return templates.TemplateResponse("download.html", {"request": request})

@app.get("/ai", response_class=HTMLResponse)
async def get_ai_page(request: Request):
    return templates.TemplateResponse("download.html", {"request": request})


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