# libraries
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from tinkoff.invest import CandleInterval
import plotly.graph_objects as go
from pathlib import Path
from typing import Union
import os
import sys
import pandas as pd
from fileinput import filename

app = FastAPI()
current_directory = Path(__file__).resolve()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# project dependences
from api_models import CANDLE_INTERVAL_MAP
from data_collecting.collect_tinkoff_data import get_by_timeframe_figi
from ML.ansamble.ai import ensemble_predict


def get_figi_and_ticker(label: str):
    """
    input: FIGI или Тикер финансовго инструмента
    output: tuple(Ticker, FIGI)
    FIGI: 12 символов: латинские буквы и десятичные цифры - уникальный индентификатор ценной бумаги
    Остальное расценивается как тикер (от 3-х до
    """
    project_root = current_directory.parents[1]
    all_figi_tickers_df = pd.read_csv(os.path.join(project_root, "all_figi_categ.csv"), index_col=0)
    all_figi_tickers_df['Ticker'] = all_figi_tickers_df['Ticker'].astype(str)
    all_figi_tickers_df['Figi'] = all_figi_tickers_df['Figi'].astype(str)

    # Проверяем, является ли label тикером
    ticker_match = all_figi_tickers_df[all_figi_tickers_df['Ticker'] == label]
    if not ticker_match.empty:
        row = ticker_match.iloc[0]
        return row['Ticker'], row['Figi']

    # Проверяем, является ли label FIGI
    figi_match = all_figi_tickers_df[all_figi_tickers_df['Figi'] == label]
    if not figi_match.empty:
        row = figi_match.iloc[0]
        return row['Ticker'], row['Figi']

    raise ValueError(f"Label '{label}' не найден ни в колонке Ticker, ни в Figi.")


def load_tinkoff(
        figi: str, interval: CandleInterval, days_back_begin: int,
        days_back_end: int = 0
):
    df = get_by_timeframe_figi(figi=figi, days_back_begin=days_back_begin,
                               days_back_end=days_back_end, interval=interval,
                               save_table=False)
    print('df loaded successfully')
    return df


@app.get("/download/csv")
def download_csv(
        tinkoff_days_back: int = Query(..., description="Количество дней истории"),
        tinkoff_figi_or_ticker: str = Query(..., description="FIGI бумаги"),
        curr_interval: str = Query(..., description="Интервал (строка)")
):
    # Преобразуем строку в настоящий CandleInterval Tinkoff
    if isinstance(curr_interval, str):
        if curr_interval in CANDLE_INTERVAL_MAP:
            tinkoff_interval = CANDLE_INTERVAL_MAP[curr_interval]
        else:
            raise HTTPException(status_code=400, detail="Некорректный размер таймфрейма")

    ticker, figi = get_figi_and_ticker(tinkoff_figi_or_ticker.upper())
    df = load_tinkoff(figi=figi, days_back_begin=tinkoff_days_back, interval=tinkoff_interval)

    current_directory = Path(__file__).resolve().parent
    df_filename = f'days_{tinkoff_days_back}_figi_{ticker}.csv'
    df_path = os.path.join(current_directory, "temp", df_filename)
    df.to_csv(df_path)
    return FileResponse(df_path, filename=df_filename, media_type="text/csv")


@app.get("/download/html")
def download_html(
        tinkoff_days_back: int = Query(..., description="Количество дней истории"),
        tinkoff_figi_or_ticker: str = Query(..., description="FIGI бумаги"),
        curr_interval: str = Query(..., description="Интервал (строка)")
):
    # Преобразуем строку в настоящий CandleInterval Tinkoff
    if isinstance(curr_interval, str):
        if curr_interval in CANDLE_INTERVAL_MAP:
            tinkoff_interval = CANDLE_INTERVAL_MAP[curr_interval]
        else:
            raise HTTPException(status_code=400, detail="Некорректный размер таймфрейма")
    ticker, figi = get_figi_and_ticker(tinkoff_figi_or_ticker.upper())
    df = load_tinkoff(figi=figi, days_back_begin=tinkoff_days_back, interval=tinkoff_interval)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode='lines',
        line=dict(color="blue"), name='Цена закрытия'))
    fig.update_layout(title=f"FIGI: {tinkoff_figi_or_ticker}, CANDLE: close, INTERVAL: {curr_interval}",
                      xaxis_title=f"Дата {tinkoff_days_back} дней назад)", yaxis_title="Цена",
                      legend_title="Обозначения")

    current_directory = Path(__file__).resolve().parent
    fig_filename = f'days_{tinkoff_days_back}_figi_{ticker}.html'
    fig_path = os.path.join(current_directory, "temp", fig_filename)
    fig.write_html(fig_path)
    return FileResponse(fig_path, filename=fig_filename, media_type="text/html")


@app.get("/predict")
def predict_price(
        tinkoff_days_back: int = Query(..., description="Количество дней истории"),
        tinkoff_figi_or_ticker: str = Query(..., description="FIGI бумаги"),
        curr_interval: str = Query(..., description="Интервал (строка)"),
):
    # Преобразуем строку в настоящий CandleInterval Tinkoff
    if isinstance(curr_interval, str):
        if curr_interval in CANDLE_INTERVAL_MAP:
            tinkoff_interval = CANDLE_INTERVAL_MAP[curr_interval]
        else:
            raise HTTPException(status_code=400, detail="Некорректный размер таймфрейма")

    ticker, figi = get_figi_and_ticker(tinkoff_figi_or_ticker.upper())
    df = load_tinkoff(figi=figi, days_back_begin=tinkoff_days_back, interval=tinkoff_interval)

    project_root = Path(__file__).resolve().parents[1]
    current_timeframe = curr_interval
    models_dir = os.path.join(project_root, 'ML', 'ansamble', current_timeframe)
    predictions, final_decision = ensemble_predict(
        df=df,
        models_dir_path=models_dir,
    )
    return {'predictions': predictions, 'final_decision': final_decision}


@app.get("/test_api")
def test():
    return {"message": "Meteora Capital API работает!"}
