
from fastapi import FastAPI, Request, Query, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from passlib.context import CryptContext
from pathlib import Path
from sqlalchemy import desc
from sqlalchemy.orm import Session
from sqlalchemy.sql.annotation import Annotated
from datetime import datetime
from pathlib import Path
import sys
import os

# current_directory = Path(__file__).resolve()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# project dependences
from API.api_main import app as api_app, download_csv, download_html, predict_price, get_figi_and_ticker  # Импортируем API
from API.api_models import CANDLE_INTERVAL_MAP
from web_app.database import engine, get_db
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


def record_history(
    user_id: int, ticker: str, figi: str, days_back: int, request_type: str,
    db: Session = Depends(get_db),
):
    history = models.RequestHistory(
        user_id=user_id,
        ticker=ticker,
        figi=figi,
        days_back=days_back,
        request_type=request_type,
        created_at=datetime.now()
    )
    db.add(history)
    db.commit()
    db.refresh(history)
    return history


@app.get("/", response_class=HTMLResponse)
async def serve_react_app():
    return FileResponse(os.path.join(frontend_path, "index.html"))


@app.post("/register")
async def register(user: models.UserCreate, db: Session = Depends(get_db)):
    user_in_db = get_user(db=db, email=user.email)
    if user_in_db:
        raise HTTPException(status_code=400, detail="User already exists")
    create_user(db=db, email=user.email, password=user.password, b_day=user.b_day, country=user.country)
    return {'message': 'created successfully'}


@app.post("/login")
async def login(user: models.UserLogin, db: Session = Depends(get_db)):
    user_in_db = get_user(db=db, email=user.email)
    if not user_in_db or not pwd_context.verify(user.password, user_in_db.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    return {'message': f'{user.email} logged in successfully'}


@app.post("/userhistory")
async def login(payload: models.EmailRequest,
                db: Session = Depends(get_db)):
    user = get_user(email=payload.email, db=db)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    history_records = db.query(models.RequestHistory).filter(models.RequestHistory.user_id == user.id).order_by(desc(models.RequestHistory.created_at)).all()

    return [
        {
            "ticker": r.ticker,
            "figi": r.figi,
            "days_back": r.days_back,
            "request_type": r.request_type,
            "created_at": r.created_at.isoformat()
        }
        for r in history_records
    ]


@app.get("/download/csv")
def redirect_download_csv(
        tinkoff_days_back: int = Query(..., description="Количество дней истории"),
        tinkoff_figi_or_ticker: str = Query(..., description="FIGI бумаги"),
        curr_interval: str = Query(..., description="Интервал (строка)"),
        user_email: str = Query(..., description="Должен быть зарегистрирован (строка)"),
        db: Session = Depends(get_db),
):
    user = get_user(email=user_email, db=db)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        ticker, figi = get_figi_and_ticker(tinkoff_figi_or_ticker)
    except ValueError as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))

    record_history(db=db, user_id=user.id, ticker=ticker, figi=figi,
        days_back=tinkoff_days_back, request_type="CSV")

    return download_csv(tinkoff_days_back=tinkoff_days_back,
                        tinkoff_figi_or_ticker=tinkoff_figi_or_ticker,
                        curr_interval=curr_interval)


@app.get("/download/html")
def redirect_download_html(
        tinkoff_days_back: int = Query(..., description="Количество дней истории"),
        tinkoff_figi_or_ticker: str = Query(..., description="FIGI бумаги"),
        curr_interval: str = Query(..., description="Интервал (строка)"),
        user_email: str = Query(..., description="Должен быть зарегистрирован (строка)"),
        db: Session = Depends(get_db),
):
    user = get_user(email=user_email, db=db)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        ticker, figi = get_figi_and_ticker(tinkoff_figi_or_ticker)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    record_history(db=db, user_id=user.id, ticker=ticker, figi=figi,
                   days_back=tinkoff_days_back, request_type="HTML")

    return download_html(tinkoff_days_back=tinkoff_days_back,
                         tinkoff_figi_or_ticker=tinkoff_figi_or_ticker,
                         curr_interval=curr_interval)


@app.get("/predict")
def redirect_predict_price(
        tinkoff_days_back: int = Query(..., description="Количество дней истории"),
        tinkoff_figi_or_ticker: str = Query(..., description="FIGI бумаги"),
        curr_interval: str = Query(..., description="Интервал (строка)"),
        user_email: str = Query(..., description="Должен быть зарегистрирован (строка)"),
        db: Session = Depends(get_db),
):
    user = get_user(email=user_email, db=db)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        ticker, figi = get_figi_and_ticker(tinkoff_figi_or_ticker)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    record_history(db=db, user_id=user.id, ticker=ticker, figi=figi,
                   days_back=tinkoff_days_back, request_type="Predict")

    return predict_price(tinkoff_days_back=tinkoff_days_back,
                         tinkoff_figi_or_ticker=tinkoff_figi_or_ticker,
                         curr_interval=curr_interval)


@app.get("/logo", response_class=FileResponse)
async def get_logo():
    logo_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "vite.svg")
    return FileResponse(logo_path)


@app.get("/test_web")
def read_root():
    return {"message": "Trading Web App is working"}


def create_models_dirs():
    proj_root = os.path.join(Path(__file__).resolve().parents[1])
    intervals = CANDLE_INTERVAL_MAP.keys()
    directories = []
    for i in intervals:
        directories.append(os.path.join(proj_root, "ML", "ansamble", i, "LSTM"))
        directories.append(os.path.join(proj_root, "ML", "ansamble", i, "HMM"))
        directories.append(os.path.join(proj_root, "ML", "ansamble", i, "Transformer"))
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)

    print("Директории успешно созданы (если не существовали ранее).")

if __name__ == "__main__":
    import uvicorn
    create_models_dirs()
    uvicorn.run("web_app.web_main:app", host="127.0.0.1", port=8000, reload=True)