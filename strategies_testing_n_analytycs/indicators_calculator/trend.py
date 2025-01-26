import pandas_ta as ta
import pandas as pd
from plotly._subplots import make_subplots


class TrendIndicatorsCalculator:
    def __init__(self, df: pd.DataFrame, fig=None):
        self.df = df
        self.fig = fig

    def add_adx(self, length: int = 14, scalar: float = 100, drift: int = 1, offset: int = 0):
        self.df = pd.concat([self.df, ta.adx(
            high=self.df["High"],
            low=self.df["Low"],
            close=self.df["Close"],
            length=length,
            scalar=scalar,
            drift=drift,
            offset=offset)], axis='columns')

    def add_aroon(self, length: int = 25, offset: int = 0):
        self.df = pd.concat([self.df, ta.aroon(
            high=self.df["High"],
            low=self.df["Low"],
            length=length,
            offset=offset)], axis='columns')

    def add_chop(self, length: int = 14, scalar: float = 100, offset: int = 0):
        self.df = pd.concat([self.df,ta.chop(
            high=self.df["High"],
            low=self.df["Low"],
            close=self.df["Close"],
            length=length,
            scalar=scalar,
            offset=offset)], axis='columns')

    def add_cksp(self, kc_mult: float = 1.0, atr_length: int = 10, offset: int = 0):
        self.df = pd.concat([self.df, ta.cksp(
            high=self.df["High"],
            low=self.df["Low"],
            close=self.df["Close"],
            kc_mult=kc_mult,
            atr_length=atr_length,
            offset=offset)], axis='columns')

    def add_dpo(self, length: int = 20, centered: bool = True, offset: int = 0):
        self.df = pd.concat([self.df, ta.dpo(
            close=self.df["Close"],
            length=length,
            centered=centered,
            offset=offset)], axis='columns')

    def add_psar(self, af: float = 0.02, max_af: float = 0.2, offset: int = 0):
        self.df = pd.concat([self.df, ta.psar(
            high=self.df["High"],
            low=self.df["Low"],
            close=self.df["Close"],
            af=af,
            max_af=max_af,
            offset=offset)], axis='columns')

    def add_ttm_trend(self, length: int = 6, offset: int = 0):
        self.df = pd.concat([self.df, ta.ttm_trend(
            close=self.df["Close"],
            high=self.df['High'],
            low=self.df['Low'],
            length=length,
            offset=offset)], axis='columns')

    def add_vhf(self, length: int = 28, offset: int = 0):
        self.df = pd.concat([self.df, ta.vhf(
            high=self.df["High"],
            low=self.df["Low"],
            close=self.df["Close"],
            length=length,
            offset=offset)], axis='columns')

    def add_vortex(self, length: int = 14, offset: int = 0):
        self.df = pd.concat([self.df, ta.vortex(
            high=self.df["High"],
            low=self.df["Low"],
            close=self.df["Close"],
            length=length,
            offset=offset)], axis='columns')

    def add_tsignals(self, trend_length: int = 14, signal_length: int = 5, offset: int = 0):
        self.df = pd.concat([self.df, ta.tsignals(
            close=self.df["Close"],
            trend_length=trend_length,
            signal_length=signal_length,
            offset=offset)], axis='columns')

def main(df:pd.DataFrame, data_write_path: str=''):
    calculator = TrendIndicatorsCalculator(df=df, fig=make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02))
    
    calculator.add_adx(length=14, scalar=100, drift=1, offset=0)
    calculator.add_aroon(length=25, offset=0)
    calculator.add_chop(length=14, scalar=100, offset=0)
    calculator.add_cksp(kc_mult=1.0, atr_length=10, offset=0)
    calculator.add_dpo(length=20, centered=True, offset=0)
    calculator.add_psar(af=0.02, max_af=0.2, offset=0)
    calculator.add_ttm_trend(length=6, offset=0)
    calculator.add_vhf(length=28, offset=0)
    calculator.add_vortex(length=14, offset=0)
    #calculator.add_tsignals(trend_length=14, signal_length=5, offset=0)
    if data_write_path != '':
        calculator.df.to_csv(data_write_path)

    return calculator.df
    
