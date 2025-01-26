import pandas as pd

from ML.data_manipulations import prepare_data_ratio, split_data
from data_collecting.collect_tinkoff_data import get_by_timeframe_figi

#indicators calsulator
from strategies_testing_n_analytycs.indicators_calculator.momentum import main as momentum
from strategies_testing_n_analytycs.indicators_calculator.overlap import main as overlap
from strategies_testing_n_analytycs.indicators_calculator.trend import main as trend
from strategies_testing_n_analytycs.indicators_calculator.volatility import main as volatility


def load_bybit():
    pass

def load_tinkoff(figi: str, days_back_begin: int, days_back_end: int, data_write_path: str):
    df = get_by_timeframe_figi(figi=figi, days_back_begin=days_back_begin, days_back_end=days_back_end, save_table=False)
    df = df.drop(['Date'], axis=1)
    df.to_csv(data_write_path)
    return df

def process_data(df: pd.DataFrame, train_part: float, ):
    df = prepare_data_ratio(df=df, n_ratio=5, window_size=40)
    frames = dict()
    frames['momentum'] = split_data(df=momentum(df), train_part=train_part)
    frames['overlap'] = split_data(df=overlap(df), train_part=train_part)
    frames['trend'] = split_data(df=trend(df), train_part=train_part)
    frames['volatility'] = split_data(df=volatility(df), train_part=train_part)
    frames['pure'] = split_data(df=df, train_part=train_part)
    return frames

"""
{momentum: momentum path,
overlap: overlap path}
"""

def train_lstm():
    
def train_hmm():
