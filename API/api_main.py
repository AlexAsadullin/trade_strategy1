# libraries
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from tinkoff.invest import CandleInterval
import plotly.graph_objects as go
from pathlib import Path
from typing import Union
import os
import sys
from fileinput import filename

app = FastAPI()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# project dependences
from models import CANDLE_INTERVAL_MAP
from data_collecting.collect_tinkoff_data import get_by_timeframe_figi
from ML.ansamble.ai import ensemble_predict

def load_tinkoff(figi: str, interval: CandleInterval, days_back_begin: int, days_back_end: int=0):
    df = get_by_timeframe_figi(figi=figi, days_back_begin=days_back_begin,
                               days_back_end=days_back_end, interval=interval,
                                save_table=False)
    print('df loaded successfully')
    return df

@app.get("/test")
def test():
    print("Test endpoint called!")
    return {"message": "API работает!"}

@app.get("/download/csv")
def download_csv(
    tinkoff_days_back: int = Query(..., description="Количество дней истории"),
    tinkoff_figi: str = Query(..., description="FIGI бумаги"),
    curr_interval: str = Query(..., description="Интервал (строка)")
):
    # Преобразуем строку в настоящий CandleInterval Tinkoff
    if isinstance(curr_interval, str):
        if curr_interval in CANDLE_INTERVAL_MAP:
            tinkoff_interval = CANDLE_INTERVAL_MAP[curr_interval]
            print(tinkoff_interval)
        else:
            raise HTTPException(status_code=400, detail="Некорректный размер таймфрейма")

    df = load_tinkoff(figi=tinkoff_figi, days_back_begin=tinkoff_days_back, interval=tinkoff_interval)

    current_directory = Path(__file__).resolve().parent
    df_filename = f'days_{tinkoff_days_back}_figi_{tinkoff_figi}.csv'
    print('data saved')
    df_path = os.path.join(current_directory, "temp", df_filename)
    df.to_csv(df_path)
    return FileResponse(df_path, filename=df_filename, media_type="text/csv")

@app.get("/download/html")
def download_html(tinkoff_days_back: int = Query(..., description="Количество дней истории"),
                  tinkoff_figi: str = Query(..., description="FIGI бумаги"),
                  curr_interval: str = Query(..., description="Интервал (строка)")
                  ):
    # Преобразуем строку в настоящий CandleInterval Tinkoff
    if isinstance(curr_interval, str):
        if curr_interval in CANDLE_INTERVAL_MAP:
            tinkoff_interval = CANDLE_INTERVAL_MAP[curr_interval]
        else:
            raise HTTPException(status_code=400, detail="Некорректный размер таймфрейма")

    df = load_tinkoff(figi=tinkoff_figi, days_back_begin=tinkoff_days_back, interval=tinkoff_interval)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode='lines',
        line=dict(color="blue"), name='Цена закрытия'))
    fig.update_layout(title=f"FIGI: {tinkoff_figi}, CANDLE: close, INTERVAL: {curr_interval}",
                      xaxis_title=f"Дата {tinkoff_days_back} дней назад)", yaxis_title="Цена",
                      legend_title="Обозначения")

    current_directory = Path(__file__).resolve().parent
    fig_filename = f'days_{tinkoff_days_back}_figi_{tinkoff_figi}.html'
    fig_path = os.path.join(current_directory, "temp", fig_filename)
    fig.write_html(fig_path)
    return FileResponse(fig_path, filename=fig_filename, media_type="text/html")

@app.get("/predict")
def predict_price(tinkoff_days_back: int = Query(..., description="Количество дней истории"),
                  tinkoff_figi: str = Query(..., description="FIGI бумаги"),
                  curr_interval: str = Query(..., description="Интервал (строка)"),
                  ):
    # Преобразуем строку в настоящий CandleInterval Tinkoff
    if isinstance(curr_interval, str):
        if curr_interval in CANDLE_INTERVAL_MAP:
            tinkoff_interval = CANDLE_INTERVAL_MAP[curr_interval]
        else:
            raise HTTPException(status_code=400, detail="Некорректный размер таймфрейма")

    df = load_tinkoff(figi=tinkoff_figi, days_back_begin=tinkoff_days_back, interval=tinkoff_interval)
    project_dir = Path(__file__).resolve().parents[1]
    current_timeframe = curr_interval
    models_dir = os.path.join(project_dir, 'ML', 'ansamble', current_timeframe)
    predictions, final_decision = ensemble_predict(
        df=df,
        models_dir_path=models_dir,
    )
    return {'predictions': predictions, 'final_decision': final_decision}
