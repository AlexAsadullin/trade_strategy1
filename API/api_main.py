# libraries
from fileinput import filename

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from tinkoff.invest import CandleInterval
import plotly.graph_objects as go
from pathlib import Path
from typing import Union
import os
import sys

app = FastAPI()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# project dependences
from models import CANDLE_INTERVAL_MAP
from data_collecting.collect_tinkoff_data import get_by_timeframe_figi


def load_tinkoff(figi: str, interval: CandleInterval, days_back_begin: int, days_back_end: int=0):
    df = get_by_timeframe_figi(figi=figi, days_back_begin=days_back_begin,
                               days_back_end=days_back_end, interval=interval,
                                save_table=False)
    return df

@app.get("/download/csv")
def download_csv(
    tinkoff_days_back: int = Query(..., description="Количество дней истории"),
    tinkoff_figi: str = Query(..., description="FIGI тикера"),
    tinkoff_interval: str = Query(..., description="Интервал (строка или Enum)")
):
    # Преобразуем строку в настоящий CandleInterval Tinkoff
    if isinstance(tinkoff_interval, str):
        if tinkoff_interval in CANDLE_INTERVAL_MAP:
            candle_interval = CANDLE_INTERVAL_MAP[tinkoff_interval]
            print(candle_interval)
        else:
            raise HTTPException(status_code=400, detail="Некорректный размер таймфрейма")

    df = load_tinkoff(figi=tinkoff_figi, days_back_begin=tinkoff_days_back, interval=candle_interval)
    df_filename = f'days_{tinkoff_days_back}_figi_{tinkoff_figi}.csv'
    df_path = os.path.join("temp", df_filename)
    df.to_csv(df_path)
    return FileResponse("temp", filename=df_filename, media_type="text/csv")

@app.get("/download/html")
def download_html(tinkoff_days_back: int = Query(..., description="Количество дней истории"),
    tinkoff_figi: str = Query(..., description="FIGI тикера"),
    tinkoff_interval: str = Query(..., description="Интервал (строка или Enum)")
):
    # Преобразуем строку в настоящий CandleInterval Tinkoff
    if isinstance(tinkoff_interval, str):
        if tinkoff_interval in CANDLE_INTERVAL_MAP:
            candle_interval = CANDLE_INTERVAL_MAP[tinkoff_interval]
        else:
            raise HTTPException(status_code=400, detail="Некорректный размер таймфрейма")

    df = load_tinkoff(figi=tinkoff_figi, days_back_begin=tinkoff_days_back, interval=candle_interval)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode='lines',
        line=dict(color="blue"), name='Цена закрытия'))
    fig.update_layout(title=f"FIGI: {tinkoff_figi}, CANDLE: close, INTERVAL: {tinkoff_interval}",
                      xaxis_title=f"Дата {tinkoff_days_back} дней назад)", yaxis_title="Цена",
                      legend_title="Обозначения")
    fig_filename = f'days_{tinkoff_days_back}_figi_{tinkoff_figi}.html'
    fig_path = os.path.join("temp", fig_filename)
    fig.write_html(fig_path)
    return FileResponse("temp", filename=fig_filename, media_type="text/html")

@app.get("/predict")
def predict_price(symbol: str = Query(..., description="Тикер акции")):
    # Здесь должен быть вызов ML-модели (заглушка)
    prediction = {"symbol": symbol, "predicted_price": 123.45}  # TODO: заменить на реальный вызов модели
    return JSONResponse(content=prediction)
