import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tinkoff.invest import CandleInterval
import sys
import os
from hmmlearn.hmm import GaussianHMM
from torch.utils.data import DataLoader
import plotly

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# import is used for 2 levels above
from ML.data_manipulations import prepare_data_ratio, split_data
from data_collecting.collect_tinkoff_data import get_by_timeframe_figi
from ML.custom_datasets import TimeSeriesDataset
from ML.lstm_train import LSTM
from ML.transformer import TransformerModel
from ML.custom_metrics import das_metric_multi, wms_metric_multi

#indicators calsulator
from strategies_testing_n_analytycs.indicators_calculator.momentum import main as momentum
from strategies_testing_n_analytycs.indicators_calculator.overlap import main as overlap
from strategies_testing_n_analytycs.indicators_calculator.trend import main as trend
from strategies_testing_n_analytycs.indicators_calculator.volatility import main as volatility


def load_bybit():
    pass

def load_tinkoff(figi: str, days_back_begin: int, interval: CandleInterval, days_back_end: int=0):
    df = get_by_timeframe_figi(figi=figi, days_back_begin=days_back_begin,
                               days_back_end=days_back_end, interval=interval,
                                save_table=False)
    try: 
        df = df.drop(['Date'], axis='columns')
    except Exception as e: 
        print(e)
    return df

def process_data(df: pd.DataFrame, train_part: float, scaler):
    df = prepare_data_ratio(df=df, n_prev_ratio=5, n_next_ratio=5, window_size=40)
    df=df.dropna()
    print('are there any nan?', df.isna().any().any())
    print('all nans count:', df.isna().sum().sum())
    frames = dict()
    frames['momentum'] = split_data(df=momentum(df), train_part=train_part, scaler=scaler)
    frames['overlap'] = split_data(df=overlap(df), train_part=train_part, scaler=scaler)
    frames['trend'] = split_data(df=trend(df), train_part=train_part, scaler=scaler)
    frames['volatility'] = split_data(df=volatility(df), train_part=train_part, scaler=scaler)
    frames['pure'] = split_data(df=df.dropna(axis='index'), train_part=train_part, scaler=scaler)
    return frames

import numpy as np

def align_shapes(y_pred, y_test):
    """ Приводим y_pred и y_test к одинаковой форме """
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)

    # Если y_pred одномерный, превращаем его в [samples, 1]
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Если y_test одномерный, превращаем его в [samples, 1]
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    # Обрезаем до минимального размера (чтобы избежать рассинхрона)
    min_len = min(y_pred.shape[0], y_test.shape[0])
    y_pred = y_pred[:min_len]
    y_test = y_test[:min_len]

    return y_pred, y_test

def train_all_hmm(tinkoff_days_back: int, tinkoff_figi: str, tinkoff_interval: CandleInterval, 
                   model_n_components: int=3, model_covariance_type:str="full", model_n_iter:int=50,
                   model_random_state:int=42,
                   ):
    df = load_tinkoff(days_back_begin=tinkoff_days_back, figi=tinkoff_figi, interval=tinkoff_interval)
    trained_models = dict()
    ready_dataset = process_data(df=df, train_part=0.7, scaler=StandardScaler())

    for indicator in ready_dataset.keys():
        X_train, X_test, y_train, y_test = ready_dataset[indicator]

        if X_train.shape[0] < model_n_components:
            print(f"Skipping '{indicator}' - Not enough data for {model_n_components} components")
            continue
        if np.isnan(X_train).any() or np.isnan(X_test).any() or np.isnan(y_train).any() or np.isnan(y_test).any():
            print(f"NaN values detected in dataset '{indicator}'!")
            continue  # Skip this iteration to avoid training on NaN values

        model = GaussianHMM(n_components=model_n_components, covariance_type=model_covariance_type,
                            n_iter=model_n_iter, random_state=model_random_state, tol=1e-4, verbose=True)

        model.fit(X_train)
        # Копируем X_test, так как будем модифицировать его в процессе предсказаний
        X_test_modified = X_test.copy()
        y_pred_matrix = np.zeros_like(y_test)

        for i in range(y_train.shape[1]):  # Последовательно предсказываем y[i]
            y_pred_proba = model.predict_proba(X_test_modified)
            y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
            
            y_pred_matrix[:, i] = y_pred  # Записываем предсказание в матрицу

            # Добавляем предсказанный y[i] в X_test для следующих итераций
            X_test_modified = np.hstack((X_test_modified, y_pred.reshape(-1, 1)))

        # Вычисляем средние метрики
        mean_das = das_metric_multi(actuals=y_test, predictions=y_pred_matrix)
        mean_wms = wms_metric_multi(actuals=y_test, predictions=y_pred_matrix)

        print(f"HMM {indicator} - Mean DAS: {mean_das}")
        print(f"HMM {indicator} - Mean WMS: {mean_wms}")

        """y_pred_proba = model.predict_proba(X_test)[:, 1]
        #y_pred = (y_pred_proba > 0.5).astype(int)

        y_pred = (y_pred_proba[:, 1] > 0.5).astype(int) if y_train.shape[1] == 1 else (y_pred_proba > 0.5).astype(int)
        print(f'HMM {indicator} finished\nDAS: {directional_accuracy_score(actuals=y_test, predictions=y_pred)}\nWMS: {wise_match_score(actuals=y_test, predictions=y_pred)}')
        """
        trained_models[indicator] = model
        joblib.dump(model, rf'/home/alex/BitcoinScalper/ML/ansamble/trained_models/HMM/{indicator}.pkl')
    return trained_models

def train_all_lstm(tinkoff_days_back: int, tinkoff_figi: str, tinkoff_interval: CandleInterval, 
                   model_hidden_size: int=64, model_num_stacked_layers:int=5, model_batch_size:int=32, model_loss_function=nn.L1Loss(),
                   training_num_epochs:int=3):
    
    df = load_tinkoff(days_back_begin=tinkoff_days_back, figi=tinkoff_figi, interval=tinkoff_interval)
    trained_models = dict()
    ready_dataset = process_data(df=df, train_part=0.7, scaler=StandardScaler())

    for indicator in ready_dataset.keys():
        X_train, X_test, y_train, y_test = ready_dataset[indicator]

        if np.isnan(X_train).any() or np.isnan(X_test).any() or np.isnan(y_train).any() or np.isnan(y_test).any():
            print(f"NaN values detected in dataset '{indicator}'!")
            continue

        train_loader = torch.utils.data.DataLoader(
            TimeSeriesDataset(X_train, y_train, window_size=30), batch_size=model_batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            TimeSeriesDataset(X_test, y_test, window_size=30), batch_size=model_batch_size, shuffle=False
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = LSTM(input_size=X_train.shape[1], hidden_size=model_hidden_size,
                    num_stacked_layers=model_num_stacked_layers,
                    device=device, loss_function=model_loss_function).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Обучение модели
        for epoch in range(training_num_epochs):
            avg_loss = model.train_validate_one_epoch(train_loader, optimizer, epoch)

        # Инференс с последовательным добавлением предсказанных значений в X_test
        model.eval()
        actuals, predictions = [], []
        X_test_modified = torch.tensor(X_test.copy(), dtype=torch.float32).to(device)

        with torch.no_grad():
            for i in range(y_train.shape[1]):  # Итерируемся по колонкам y
                y_pred = model(X_test_modified).cpu().numpy()
                predictions.append(y_pred[:, i])  # Записываем предсказания для y[i]
                actuals.append(y_test[:, i])  # Сохраняем реальные значения y[i]

                # Добавляем предсказанный y[i] в X_test
                y_pred_col = torch.tensor(y_pred[:, i], dtype=torch.float32).view(-1, 1).to(device)
                X_test_modified = torch.cat((X_test_modified, y_pred_col), dim=1)

        # Преобразуем списки в numpy
        actuals = np.array(actuals).T  # Транспонируем, чтобы вернуть исходный размер [samples, targets]
        predictions = np.array(predictions).T

        # Вычисляем средние метрики
        mean_das = das_metric_multi(actuals=actuals, predictions=predictions)
        mean_wms = wms_metric_multi(actuals=actuals, predictions=predictions)

        print(f"Transformer {indicator} - Mean DAS: {mean_das}")
        print(f"Transformer {indicator} - Mean WMS: {mean_wms}")

        trained_models[indicator] = model

        torch.save(model, rf'/home/alex/BitcoinScalper/ML/ansamble/trained_models/LSTM/{indicator}.pth')
    
    # visual
    return trained_models

def train_all_transformer(tinkoff_days_back: int, tinkoff_figi: str, tinkoff_interval: CandleInterval,
                    model_loss_function=nn.L1Loss(), model_batch_size=32,
                    training_num_epochs:int=4,):
    df = load_tinkoff(days_back_begin=tinkoff_days_back, figi=tinkoff_figi, interval=tinkoff_interval)
    trained_models = dict()
    ready_dataset = process_data(df=df, train_part=0.7, scaler=StandardScaler())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for indicator in ready_dataset.keys():
        X_train, X_test, y_train, y_test = ready_dataset[indicator]

        if np.isnan(X_train).any() or np.isnan(X_test).any() or np.isnan(y_train).any() or np.isnan(y_test).any():
            print(f"NaN values detected in dataset '{indicator}'!")
            continue

        train_loader = DataLoader(TimeSeriesDataset(X_train, y_train, window_size=30), batch_size=model_batch_size, shuffle=True)
        test_loader = DataLoader(TimeSeriesDataset(X_test, y_test, window_size=30), batch_size=model_batch_size, shuffle=False)

        input_dim = X_train.shape[1]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = TransformerModel(input_dim, device=device, loss_function=model_loss_function).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Обучение модели
        for epoch in range(training_num_epochs):
            avg_loss = model.train_validate_one_epoch(train_loader, optimizer, epoch)

        model.eval()
        actuals, predictions = [], []
        X_test_modified = torch.tensor(X_test.copy(), dtype=torch.float32).to(device)

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch).cpu().numpy()
                predictions.extend(y_pred.flatten())
                actuals.extend(y_batch.numpy().flatten())

        mean_das = das_metric_multi(actuals=actuals, predictions=predictions)
        mean_wms = wms_metric_multi(actuals=actuals, predictions=predictions)

        print(f"Transformer {indicator} - Mean DAS: {mean_das}")
        print(f"Transformer {indicator} - Mean WMS: {mean_wms}")

        trained_models[indicator] = model  # TODO: add settings dictionary

        torch.save(model, rf'/home/alex/BitcoinScalper/ML/ansamble/trained_models/Transformer/{indicator}.pth')
    return trained_models

if __name__ == '__main__':
    train_all_transformer(
        tinkoff_days_back=1000, tinkoff_figi='BBG004731032', tinkoff_interval=CandleInterval.CANDLE_INTERVAL_2_HOUR, 
        training_num_epochs=5
        )
    train_all_lstm(
        tinkoff_days_back=1000, tinkoff_figi='BBG004731032', tinkoff_interval=CandleInterval.CANDLE_INTERVAL_2_HOUR,
        training_num_epochs=25
    )
    train_all_hmm(
        tinkoff_days_back=1000, tinkoff_figi='BBG004731032', tinkoff_interval=CandleInterval.CANDLE_INTERVAL_2_HOUR,
    )