import pandas_ta as ta
import pandas as pd
from plotly._subplots import make_subplots

class VolatilityIndicatorsCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def bbands(self, length=20, std=2, talib=False, offset=None):
        self.df = pd.concat([self.df, ta.bbands(
            close=self.df['Close'],
            length=length,
            std=std,
            talib=talib,
            offset=offset)], axis='columns')
        
    def atr(self, length=14, scalar=1.0, offset=None):
        self.df["atr"] = ta.atr(
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            length=length,
            scalar=scalar,
            offset=offset
        )
        
    def kc(self, length=20, scalar=1.5, offset=None):
        self.df = pd.concat([self.df, ta.kc(
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            length=length,
            scalar=scalar,
            offset=offset)], axis='columns')
        
    def donchian(self, length=20, offset=None):
        self.df = pd.concat([self.df, pd.DataFrame(ta.donchian(
            high=self.df['High'],
            low=self.df['Low'],
            length=length,
            offset=offset))], axis=1)
        
    def accbands(self, length=20, scalar=1.0, offset=None):
        self.df = pd.concat([self.df,ta.accbands(
            close=self.df['Close'],
            high=self.df['High'],
            low=self.df['Low'],
            length=length,
            scalar=scalar,
            offset=offset)], axis='columns')
        
    def rvi(self, length=14, scalar=100, offset=None):
        self.df["rvi"] = ta.rvi(
            close=self.df['Close'],
            length=length,
            scalar=scalar,
            offset=offset
        )
        
    def hwc(self, na=2, nb=3, nc=5, nd=1, scalar=1.0, channel_eval=None, offset=0):
        self.df["hwc"] = ta.hwc(
            close=self.df['Close'],
            na=na,
            nb=nb,
            nc=nc,
            nd=nd,
            scalar=scalar,
            channel_eval=channel_eval,
            offset=offset
        )
        
    def true_range(self, offset=None):
        self.df["true_range"] = ta.true_range(
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            offset=offset
        )
        
    def massi(self, length=25, offset=None):
        self.df["massi"] = ta.massi(
            high=self.df['High'],
            low=self.df['Low'],
            length=length,
            offset=offset
        )
        
    def natr(self, length=14, scalar=100, offset=None):
        self.df["natr"] = ta.natr(
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            length=length,
            scalar=scalar,
            offset=offset
        )
        
def main(df: pd.DataFrame, data_write_path: str=''):
    calculator = VolatilityIndicatorsCalculator(df=df)
    
    calculator.bbands(length=20, std=2, talib=False, offset=0)
    calculator.atr(length=14, scalar=1.0, offset=0)
    calculator.kc(length=20, scalar=1.5, offset=0)
    calculator.donchian(length=20, offset=0)
    calculator.accbands(length=20, scalar=1.0, offset=0)
    calculator.rvi(length=14, scalar=100, offset=0)
    # calculator.hwc(na=2, nb=3, nc=5, nd=1, scalar=1.0, channel_eval=None, offset=0) # too long to wait
    calculator.true_range(offset=0)
    calculator.massi(length=25, offset=0)
    calculator.natr(length=14, scalar=100, offset=0)

    if data_write_path != '':
        calculator.df.to_csv(data_write_path)
    return calculator.df.dropna()
