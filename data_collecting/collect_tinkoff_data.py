# for python
import pandas as pd
import os
from collections import defaultdict
from datetime import timedelta
from dotenv import load_dotenv
# for tinkoff
from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.utils import now
import time
def quote_to_float(data):
    return float(data.units + data.nano / (10 ** 9))

def get_by_timeframe_figi(figi: str, days_back_begin: int, interval: CandleInterval, ticker: str='', 
                     days_back_end: int = 0, save_table: bool = True) -> pd.DataFrame:
    load_dotenv()
    TOKEN = os.getenv("TINKOFF_TOKEN_REAL")

    from_timedelta = now() - timedelta(days=days_back_begin)
    to_timedelta = now() - timedelta(days=days_back_end)
    print('begin data loading')
    with Client(TOKEN) as client:
        data_for_df = defaultdict(list)
        for candle in client.get_all_candles(
                figi=figi,
                from_=from_timedelta,
                to=to_timedelta,
                interval=interval):
            open_price = quote_to_float(candle.open)
            close_price = quote_to_float(candle.close)
            high_price = quote_to_float(candle.high)
            low_price = quote_to_float(candle.low)
            volume = candle.volume
            is_growing = open_price < close_price

            data_for_df['Open'].append(open_price)
            data_for_df['Close'].append(close_price)
            data_for_df['High'].append(high_price)
            data_for_df['Low'].append(low_price)
            data_for_df['Volume'].append(volume)
            data_for_df['Date'].append(candle.time.strftime('%Y-%m-%d %H:%M'))
            data_for_df['IsGrowing'].append(is_growing)
            data_for_df['AvgOpenClose'].append(abs(open_price - close_price))
            data_for_df['DiffOpenClose'].append(abs(open_price - close_price))
            data_for_df['DiffHighLow'].append(abs(high_price - low_price))
        df = pd.DataFrame(data_for_df)
        if save_table:
            interval_name = interval.name.replace("CANDLE_INTERVAL_", "")
            filename = f'''prices_{ticker}_{interval_name}_{str(from_timedelta)[:10]}.csv'''
            df.to_csv(os.path.join('tinkoff_data', filename))
        return df


def get_all_figi(save_path: str):
    from tinkoff.invest import Client
    from tinkoff.invest.services import InstrumentsService, MarketDataService
    load_dotenv()
    with Client(os.getenv('TINKOFF_TOKEN_REAL')) as client:
        instruments: InstrumentsService = client.instruments
        market_data: MarketDataService = client.market_data
        l = []
        for method in ['shares', 'bonds', 'etfs']:
            for item in getattr(instruments, method)().instruments:
                l.append({
                    'Ticker': item.ticker,
                    'Figi': item.figi,
                    'Type': method,
                    'Name': item.name,
                })
        df = pd.DataFrame(l)
        df.to_csv(save_path)
        print('data is saved')
        return df


def get_massive_by_timeframe_figi(step_back_days: int,
                                  figi: str, ticker: str, interval: CandleInterval,
                                  end_iterval_back: int, start_interval_back: int = 0,
                                  save_table: bool = True, ):
    concat_list = []
    for i in range(start_interval_back, end_iterval_back, step_back_days):
        try:
            concat_list.append(get_by_timeframe_figi(figi=figi,
                                                days_back_begin=i + step_back_days, days_back_end=i,
                                                ticker=ticker, interval=interval, save_table=False))
            #time.sleep(1)
            print(f'saved {i + step_back_days}')
        except Exception as e:
            print(e)
            break
    df = pd.concat(concat_list, axis=0)
    if save_table:
        interval_name = interval.name.replace('CANDLE_INTERVAL_', '')
        filename = f'prices_massive_{ticker}_{interval_name}_{now().strftime('%Y-%m-%d')}.csv'
        df.to_csv(rf'/home/alex/BitcoinScalper/data_collecting/tinkoff_data/{filename}')
    return df

# a = ask (спрос), b = bid (предложение)
def historical_data_analyze(data: list, ):
    i = 0
    res = defaultdict(list)
    for glass in data:
        ask_offers = glass['result']['a']
        bid_offers = glass['result']['b']
        ask_offers_number = len(ask_offers)
        bid_offers_number = len(bid_offers)
        res['Num'].append(i)
        res['Ask'].append(ask_offers)
        res['Bid'].append(bid_offers)
        res['LenAsk'].append(ask_offers_number)
        res['LenBid'].append(bid_offers_number)
        i += 1
    return res

"""if __name__ == '__main__':
    load_dotenv()
    history = []
    time_range = 60 * 24 # n hours
    # time_range = 60 # 1 min
    for i in range(time_range):
        history.append(get_bybit_glass(symbol='BTCUSDT',
                                      limit=10,
                                      category='linear'))
        print(dt.now())
        time.sleep(60)
        with open('BTCUSDT16.json', 'w') as f:
            f.write(json.dumps(history))
    print('data saved')"""

if __name__ == '__main__':
    """load_dotenv()
    interval = CandleInterval.CANDLE_INTERVAL_4_HOUR
    df = get_massive_by_timeframe_figi(step_back_days=10, figi='BBG004731032', ticker='LKOH', interval=interval,
                                       start_interval_back=3680, end_iterval_back=17000, save_table=True)"""
    get_all_figi(r"C:\trade_strategy1\all_figi.csv")