import json
import os

directory = "reports_data"
files = os.listdir(directory)

class GlassAnalyzer:
    def __init__(self):
        self.history = []
        for f in files:
            self.history.append(json.load(open(f, 'r')))

    def get_data_by_time(self, time_begin, time_end):
        

    def indicators(self, data):
        indicators = {'Buy': {},
                      'Sell': {},
                      'Both': {}}
        for side in 'Buy', 'Sell':
            indicators[side]['money/btc'] = data[side]['amount_btc'] / data[side]['amount_money'] 
            indicators[side]['money/deal'] = data[side]['amount_money'] / data[side]['deals_number']
            indicators[side]['money/btc'] = data[side]['amount_btc'] / data[side]['deals_number']
        indicators['Both']['amount_btc_Buy/Sell'] = data['Buy']['amount_btc'] / data['Sell']['amount_btc']
        indicators['Both']['amount_money_Buy/Sell'] = data['Buy']['amount_money'] / data['Sell']['amount_money']
        return indicators
