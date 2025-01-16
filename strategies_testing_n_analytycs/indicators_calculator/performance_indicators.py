import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PerformanceIndicatorsCalculator:
    def __init__(self, df: pd.DataFrame, fig=None):
        self.df = df
        self.fig = fig

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


def main(data_read_path: str, data_write_path: str, chart_path: str):
    df = pd.read_csv(data_read_path, index_col=0)
    df = df.reset_index()

    calculator = PerformanceIndicatorsCalculator(df=df, fig=make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02))

    calculator.add_drawdown(method="percent", offset=0)
    calculator.add_log_return(length=1, cumulative=False, offset=0)
    calculator.add_percent_return(length=1, cumulative=False, offset=0)

    calculator.df.to_csv(data_write_path)


if __name__ == "__main__":
    main(
        data_read_path=r"/home/alex/BitcoinScalper/dataframes/TSLA.csv",
        data_write_path=r"/home/alex/BitcoinScalper/dataframes/TSLA_performance.csv",
        chart_path=r"/home/alex/BitcoinScalper/html_charts/TSLA_performance.html"
    )