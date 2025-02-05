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
from custom_datasets import TimeSeriesDataset 
from data_manipulations import prepare_data_ratio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# 2 dirs upper
from data_collecting.collect_tinkoff_data import get_by_timeframe_figi
from strategies_testing_n_analytycs.indicators_calculator.momentum import main as momentum
from strategies_testing_n_analytycs.indicators_calculator.overlap import main as overlap
from strategies_testing_n_analytycs.indicators_calculator.trend import main as trend
from strategies_testing_n_analytycs.indicators_calculator.volatility import main as volatility

def split_prod_data(df:pd.DataFrame, scaler=None):
    X = df.drop('next_ratio', axis=1)  
    y = df['next_ratio']

    if scaler:
        return (
            scaler.fit_transform(X.to_numpy(dtype=np.float32)),
            scaler.fit_transform(y.to_numpy(dtype=np.float32).reshape(-1, 1))
            )
    else:
        return X.to_numpy(), y.to_numpy()

def process_prod_data(df: pd.DataFrame, scaler):
    df = prepare_data_ratio(df=df, n_ratio=5, window_size=40)
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

def ensemble_predict(df: pd.DataFrame, models_dir_path='/home/alex/BitcoinScalper/ML/ansamble/trained_models'):
    device = 'cpu'

    settings = {
        'LSTM': ['trend.pth', 'pure.pth'],
        'HMM': ['overlap.pkl'],
        'Transformer': ['momentum.pth', 'volatility.pth']
    }

    models = defaultdict(list)
    for learn_algorythm, models_pathes in settings.items():
        for model_path in models_pathes:
            # different types od 
            with open(f'{models_dir_path}/{learn_algorythm}/{model_path}', "rb") as file:
                print(f'{models_dir_path}/{learn_algorythm}/{model_path}')
                if model_path.endswith('.pkl'):
                    indicator_type = model_path.replace('.pkl', '')
                    models[(learn_algorythm, indicator_type)] = (joblib.load(file)) # pickle load doesn't read torch models
                elif model_path.endswith('.pth'):
                    indicator_type = model_path.replace('.pth', '')
                    models[(learn_algorythm, indicator_type)] = (torch.load(file)) # torch load doesn't read pickle models
                else:
                    print('wrong file in models directory')

    print('models loaded successfully')

    all_data = process_prod_data(df=df, scaler=StandardScaler())
    print('data arrays prepared')

    predicitons = []
    for key, value in models.items():
        learn_algorythm, indicator_type = key
        model = value

        print(learn_algorythm, indicator_type)
        X_only, y_only = all_data[indicator_type]

        ts_loader = DataLoader(TimeSeriesDataset(X_only, y_only, window_size=30), batch_size=32, shuffle=False)
        if learn_algorythm == 'LSTM':
            model.eval()
            
            actuals, predictions = [], []
            with torch.no_grad():
                for x_batch, y_batch in ts_loader:
                    x_batch = x_batch.to(device)
                    y_pred = model(x_batch).cpu().numpy()
                    predictions.extend(y_pred)
                    actuals.extend(y_batch.numpy())
        
        elif learn_algorythm == 'HMM':
            y_pred_proba = model.predict_proba(X_only)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)

        elif learn_algorythm == 'Transformer':
            model.eval()
            
            actuals, predictions = [], []
            with torch.no_grad():
                for x_batch, y_batch in ts_loader:
                    x_batch = x_batch.to(device)
                    y_pred = model(x_batch).cpu().numpy()
                    predictions.extend(y_pred)
                    actuals.extend(y_batch.numpy())
        else: 
            print('wrong model type, please check settings.json')
        
        predictions.append(y_pred)


    final_votes = np.mean(predictions, axis=0) > 1
    final_decision = np.where(final_votes, "Stock will grow", "Stock will fall")

if __name__ == '__main__':
    ensemble_predict(df=get_by_timeframe_figi(figi='BBG004731032', days_back_begin=1000, interval=CandleInterval.CANDLE_INTERVAL_2_HOUR, ticker='LKOH'),
                     models_dir_path='/home/alex/BitcoinScalper/ML/ansamble/trained_models')