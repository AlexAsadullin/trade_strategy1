# libraries
from fastapi import FastAPI
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# project dependences
from web_app.routers import history
from API.api_main import app as api_app  # Импортируем API

app = FastAPI(title="Trading Web App")

# Включаем маршруты API в web-приложение
app.mount("/api", api_app)  # Теперь API доступно по /api

# Подключаем роутеры веб-приложения
app.include_router(history.router)

# Главная страница
@app.get("/")
def read_root():
    return {"message": "Welcome to the Trading Web App"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_app.main:app", host="0.0.0.0", port=8000, reload=True)
