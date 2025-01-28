import joblib
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tinkoff.invest import CandleInterval
import sys
import os
from hmmlearn.hmm import GaussianHMM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# import is used for 2 levels above
from data_manipulations import prepare_data_ratio, split_data
from data_collecting.collect_tinkoff_data import get_by_timeframe_figi
from custom_datasets import TimeSeriesDataset
from lstm_train import LSTM
from custom_metrics import directional_accuracy_score

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
    df = df.drop(['Date'], axis='columns')
    return df

def process_data(df: pd.DataFrame, train_part: float, scaler):
    df = prepare_data_ratio(df=df, n_ratio=5, window_size=40)
    frames = dict()
    frames['momentum'] = split_data(df=momentum(df), train_part=train_part, scaler=scaler)
    frames['overlap'] = split_data(df=overlap(df), train_part=train_part, scaler=scaler)
    frames['trend'] = split_data(df=trend(df), train_part=train_part, scaler=scaler)
    frames['volatility'] = split_data(df=volatility(df), train_part=train_part, scaler=scaler)
    frames['pure'] = split_data(df=df.dropna(), train_part=train_part, scaler=scaler)
    return frames

def train_all_lstm(tinkoff_days_back: int, tinkoff_figi: str, tinkoff_interval: CandleInterval, 
                   model_hidden_size: int=64, model_num_stacked_layers:int=5, model_batch_size:int=16, model_loss_function=nn.L1Loss(),
                   training_num_epochs:int=30):
    
    df = load_tinkoff(days_back_begin=tinkoff_days_back, figi=tinkoff_figi, interval=tinkoff_interval)
    trained_models = dict()
    ready_dataset = process_data(df=df, train_part=0.7, scaler=StandardScaler())

    for indicator in ready_dataset.keys():
        X_train, X_test, y_train, y_test = ready_dataset[indicator]
        
        train_loader = torch.utils.data.DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=model_batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=model_batch_size, shuffle=False)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = LSTM(input_size=X_train.shape[1], hidden_size=model_hidden_size,
                     num_stacked_layers=model_num_stacked_layers,
                     device=device, loss_function=model_loss_function).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(training_num_epochs):
            avg_loss = model.train_validate_one_epoch(train_loader, optimizer, epoch)
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            predicted = model(X_test_tensor).detach().numpy()

        print(indicator, directional_accuracy_score(y_test=y_test, y_pred=predicted))
        trained_models[indicator] = model

        torch.save(model.state_dict(), rf'/home/alex/BitcoinScalper/ML/ansamble/trained_models/LSTM/{indicator}.pkl')
    return trained_models
    
def train_all_hmm(tinkoff_days_back: int, tinkoff_figi: str, tinkoff_interval: CandleInterval, 
                   model_n_components: int=4, model_covariance_type:str="full", model_n_iter:int=500,
                   model_random_state:int=42,
                   ):
    
    df = load_tinkoff(days_back_begin=tinkoff_days_back, figi=tinkoff_figi, interval=tinkoff_interval)
    trained_models = dict()
    ready_dataset = process_data(df=df, train_part=0.7, scaler=StandardScaler())

    for indicator in ready_dataset.keys():
        X_train, X_test, y_train, y_test = ready_dataset[indicator]
        model = GaussianHMM(n_components=model_n_components, covariance_type=model_covariance_type,
                            n_iter=model_n_iter, random_state=model_random_state)
        model.fit(X_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        print(directional_accuracy_score(y_test=y_test, y_pred=y_pred))
        trained_models[indicator] = model

        joblib.dump(model, rf'/home/alex/BitcoinScalper/ML/ansamble/trained_models/HMM/{indicator}.pkl')
    return trained_models


if __name__ == '__main__':
    train_all_hmm(
        tinkoff_days_back=1000, tinkoff_figi='BBG004731032', tinkoff_interval=CandleInterval.CANDLE_INTERVAL_2_HOUR, 
        )