import pandas as pd
import pandas_ta as ta

class CyclesIndicatorsCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def ebsw(self, length, fast, slow, signal, scalar, offset):
        self.df["ebsw"] = ta.ebsw(
            close=self.df["Close"],  
            length=length,         
            fast=fast,             
            slow=slow,            
            signal=signal,           
            scalar=scalar,         
            offset=offset            
        )

def main(df:pd.DataFrame, data_write_path: str = ''):
    calculator = CyclesIndicatorsCalculator(df)
    calculator.ebsw(length=20, fast=4, slow=10, signal=6, scalar=1.0, offset=0)
    if data_write_path != '':
        calculator.df.to_csv(data_write_path)
    
    return calculator.df

