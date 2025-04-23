import pickle
import joblib
import json
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tinkoff.invest import CandleInterval
import torch

import pickletools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# 1 dir upper
from ML.custom_datasets import TimeSeriesDataset
from ML.data_manipulations import prepare_data_ratio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# 2 dirs upper
from data_collecting.collect_tinkoff_data import get_by_timeframe_figi
from strategies_testing_n_analytycs.indicators_calculator.momentum import main as momentum
from strategies_testing_n_analytycs.indicators_calculator.overlap import main as overlap
from strategies_testing_n_analytycs.indicators_calculator.trend import main as trend
from strategies_testing_n_analytycs.indicators_calculator.volatility import main as volatility
from ML.ansamble.pipeline import process_data

def split_prod_data(df:pd.DataFrame, scaler=None):
    X = df.drop("next_ratio", axis=1)
    y = df["next_ratio"]
    if scaler:
        X_scaled = scaler.fit_transform(X.to_numpy(dtype=np.float32))
        y_reshaped = y.to_numpy(dtype=np.float32).reshape(-1, 1)

        y_scaled = scaler.fit_transform(y_reshaped).ravel()

        return X_scaled, y_scaled
    else:
        return X.to_numpy(), y.to_numpy()
    """
    X = df.drop('next_ratio', axis=1)  
    y = df['next_ratio']

    if scaler:
        return (
            scaler.fit_transform(X.to_numpy(dtype=np.float32)),
            scaler.fit_transform(y.to_numpy(dtype=np.float32).reshape(-1, 1))
            )
    else:
        return X.to_numpy(), y.to_numpy()
    """



def process_prod_data(df: pd.DataFrame, scaler):
    df = prepare_data_ratio(df=df, n_prev_ratio=8, n_next_ratio=1, window_size=40) # must match with ML.ansamble.pipeline.process_data
    df=df.dropna()
    print('are there any nan?', df.isna().any().any())
    print('all nans count:', df.isna().sum().sum())
    frames = dict()
    frames['momentum'] = split_prod_data(df=momentum(df), scaler=scaler)
    frames['overlap'] = split_prod_data(df=overlap(df), scaler=scaler)
    frames['trend'] = split_prod_data(df=trend(df), scaler=scaler)
    frames['volatility'] = split_prod_data(df=volatility(df), scaler=scaler)
    frames['pure'] = split_prod_data(df=df.dropna(axis='index'), scaler=scaler)
    return frames

def ensemble_predict(df: pd.DataFrame, models_dir_path: str):
    device = 'cpu'
    scaler = StandardScaler()
    settings = {
        'LSTM': ['trend.pth', 'pure.pth'],
        #'HMM': ['overlap.pkl'],
        'Transformer': ['momentum.pth', 'volatility.pth']
    }

    models = defaultdict(list)
    for learn_algorythm, models_pathes in settings.items():
        for model_path in models_pathes:
            # different types od 
            with open(os.path.join(models_dir_path, learn_algorythm, model_path), "rb") as file:
                if model_path.endswith('.pkl'):
                    indicator_type = model_path.replace('.pkl', '')
                    models[(learn_algorythm, indicator_type)] = (joblib.load(file)) # pickle (joblib) load doesn't read torch models
                elif model_path.endswith('.pth'):
                    indicator_type = model_path.replace('.pth', '')
                    models[(learn_algorythm, indicator_type)] = (torch.load(file, weights_only=False)) # torch load doesn't read pickle models
                else:
                    print('wrong file in models directory:', model_path)

    print('models loaded successfully')

    all_data = process_prod_data(df=df, scaler=scaler)
    print('data arrays prepared')

    ansamble_predicitons = []
    for key, value in models.items():
        learn_algorythm, indicator_type = key
        model = value

        print(learn_algorythm, indicator_type)
        X_only, y_only = all_data[indicator_type]

        ts_loader = DataLoader(TimeSeriesDataset(X_only, y_only, window_size=30), batch_size=32, shuffle=False)
        if learn_algorythm == 'LSTM':
            model.eval()
            actual_data, predicted_data = [], []
            with torch.no_grad():
                for x_batch, y_batch in ts_loader:
                    x_batch = x_batch.to(device)
                    y_pred = model(x_batch).cpu().numpy()
                    predicted_data.extend(y_pred)
                    actual_data.extend(y_batch.numpy())
            ansamble_predicitons.append(float(list(predicted_data)[-1]))
        
        elif learn_algorythm == 'HMM':
            y_pred_proba = model.predict_proba(X_only)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            ansamble_predicitons.append(float(list(y_pred)[-1]))

        elif learn_algorythm == 'Transformer':
            model.eval()
            actual_data, predicted_data = [], []
            with torch.no_grad():
                for x_batch, y_batch in ts_loader:
                    x_batch = x_batch.to(device)
                    y_pred = model(x_batch).cpu().numpy()
                    predicted_data.extend(np.atleast_1d(y_pred))
                    actual_data.extend(y_batch.numpy())
            ansamble_predicitons.append(float(list(predicted_data)[-1]))
        else: 
            print('wrong model type, please check settings.json')

    # обратный scaler
    ansamble_predicitons = np.array(ansamble_predicitons, dtype=np.float32).reshape(-1, 1)
    ansamble_predicitons = scaler.inverse_transform(ansamble_predicitons)
    
    final_votes = np.mean(np.array(ansamble_predicitons), axis=0) > 1
    print(final_votes)
    final_decision = np.where(final_votes, "Stock will grow", "Stock will fall")
    print(final_decision)
    return ansamble_predicitons, final_decision

if __name__ == '__main__':
    ensemble_predict(df=pd.read_csv(r"C:\Users\asadu\PycharmProjects\trade_strategy1\ML\ansamble\train.csv", index_col=0),
                     models_dir_path=r"C:\Users\asadu\PycharmProjects\trade_strategy1\ML\ansamble\1H")
    """
    ensemble_predict(df=get_by_timeframe_figi(figi='BBG004731032', days_back_begin=1000, interval=CandleInterval.CANDLE_INTERVAL_2_HOUR, ticker='LKOH'),
                     models_dir_path='/home/alex/BitcoinScalper/ML/ansamble/trained_models')"""