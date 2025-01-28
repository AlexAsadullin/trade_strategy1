import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PerformanceIndicatorsCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def add_drawdown(self, method: str = "percent", offset: int = 0):
        self.df = pd.concat([self.df, ta.drawdown(
            close=self.df["Close"],
            method=method,
            offset=offset)], axis='columns')

    def add_log_return(self, length: int = 1, cumulative: bool = False, offset: int = 0):
        self.df["log_return"] = ta.log_return(
            close=self.df["Close"],
            length=length,
            cumulative=cumulative,
            offset=offset
        )

    def add_percent_return(self, length: int = 1, cumulative: bool = False, offset: int = 0):
        self.df["percent_return"] = ta.percent_return(
            close=self.df["Close"],
            length=length,
            cumulative=cumulative,
            offset=offset
        )


def main(df:pd.DataFrame, data_write_path: str=''):
    calculator = PerformanceIndicatorsCalculator(df=df)

    calculator.add_drawdown(method="percent", offset=0)
    calculator.add_log_return(length=1, cumulative=False, offset=0)
    calculator.add_percent_return(length=1, cumulative=False, offset=0)

    if data_write_path != '':
        calculator.df.to_csv(data_write_path)
