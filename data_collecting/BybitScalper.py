import time
import datetime
from datetime import datetime as dt
import telebot
import json

from pybit.unified_trading import HTTP

import os
from dotenv import load_dotenv, dotenv_values


class BybitGlassScalper:
    def __init__(self, chat_id, token, sleeptime_sec):
        load_dotenv()
        self.CHAT_ID = chat_id
        self.session = HTTP(api_key=os.getenv('bybit_api_key'),
                            api_secret=os.getenv('bybit_api_secret'))
        self.bot = telebot.TeleBot(token)

        self.sleeptime_sec = sleeptime_sec
        self.deals_counter = 0
        self.session_begin = dt.now()
        self.written_deals = []
        self.dataset_deals = []

    def get_new_deals(self):
        data = self.session.get_public_trade_history(
            category="spot",
            symbol="BTCUSDT",
            limit=60,
            )['result']['list']
        total = []
        print(len(data), end=' ')
        repeated_deals = 0
        for i in data:
            i['execId'] = int(i['execId'])
            if i['execId'] not in self.written_deals:
                self.written_deals.append(i['execId'])

                i['price'] = float(i['price'])
                i['size'] = float(i['size'])
                i['time'] = int(i['time'])
                total.append(i)
            else:
                repeated_deals += 1
        print(f'where repeated {repeated_deals} deals, new deals {60 - repeated_deals}', dt.now())
        return total

    def compress_dataset(self):
        data = os.listdir(r'/home/alex/BitcoinScalper/data_collecting/deals_history')
        total = []
        for i in data:
            try:
                with open(rf'/home/alex/BitcoinScalper/data_collecting/deals_history/{i}') as f:
                    file = json.load(f)
                    if file != []:
                        total.extend(file)
            except Exception as e:
                print(e, i)
        filepath = rf"/home/alex/BitcoinScalper/data_collecting/training_data/{total[0]['time']}_{total[-1]['time']}.json"
        print(filepath)
        with open(filepath, 'w') as f:
            f.write(json.dumps(total))
        for i in data:
            os.remove(f'/home/alex/BitcoinScalper/data_collecting/deals_history/{i}')

    def send_data(self, data: dict, message=''):
        filepath = f'/home/alex/BitcoinScalper/data_collecting/reports_data/{self.session_begin.strftime('%Y-%m-%d_%H-%M-%S')}.json'
        with open(filepath, 'w') as f:
            f.write(json.dumps(data))
        # self.bot.send_document(self.CHAT_ID, open(filepath, 'rb'))
        for key, value in data.items():
            message += f'{key}: {value}\n\n'
        self.bot.send_message(self.CHAT_ID, message)

    def analytycs(self, data: list):
        result = {'Buy': {'open': -1,
                        'close': -1,
                        'max': -1,
                        'min': 10 ** 8,
                        'amount_money': 0,
                        'amount_btc': 0,
                        'deals_number': 0,
                        },
                'Sell': {'open': -1,
                        'close': -1,
                        'max': -1,
                        'min': 10 ** 8,
                        'amount_money': 0,
                        'amount_btc': 0,
                        'deals_number': 0,
                        },
                'Total': {'open': -1,
                        'close': -1,
                        'max': -1,
                        'min': 10 ** 8,
                        'amount_money': 0,
                        'amount_btc': 0,
                        'deals_number': 0
                        },
                'session_end': str(dt.now())}
        first = True
        for deal in data:
            side = deal['side']
            price = float(deal['price'])
            size = float(deal['size'])

            result['Total']['close'] = price
            result[side]['close'] = price
            result[side]['amount_money'] += price * size
            result[side]['amount_btc'] += size
            result[side]['deals_number'] += 1
            
            if result[side]['open'] == -1:
                result[side]['open'] = price
            if price < result[side]['min']:
                result[side]['min'] = price
            elif price > result[side]['max']:
                result[side]['max'] = price
            if first:
                result['Total']['open'] = price
                first = False
        result['Total']['max'] = max(result['Buy']['max'], result['Sell']['max'])
        result['Total']['min'] = min(result['Buy']['min'], result['Sell']['min'])
        result['Total']['amount_money'] = result['Buy']['amount_money'] + result['Sell']['amount_money']
        result['Total']['amount_btc'] = result['Buy']['amount_btc'] + result['Sell']['amount_btc']
        result['Total']['deals_number'] = result['Buy']['deals_number'] + result['Sell']['deals_number']
        return result

    def run(self):
        iterations_counter = 0
        response = []
        now_hour = dt.now().hour
        prev_hour = dt.now().hour
        while True:
            now = dt.now()
            if now.hour == 12 or now.hour == 0:
                now_hour = now.hour
            try:
                total = self.get_new_deals()
                with open(f'/home/alex/BitcoinScalper/data_collecting/deals_history/deals_{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}.json', 'w') as f:
                    f.write(json.dumps(total))

                response.extend(total)
            except Exception:
                pass
            iterations_counter += 1
            if iterations_counter >= 10:
                try:
                    self.written_deals = self.written_deals[-70:]
                    print('repeated_deals cache cleared')
                    iterations_counter = 0
                except IndexError:
                    pass

            if now_hour != prev_hour:
                data = self.analytycs(response)
                self.send_data(data=data)
                # ite data to file and send filepath to function def send_data
                response = []
                self.session_begin = dt.now()
                print(f'session ended, {dt.now()}, new begins now')
                self.compress_dataset()
                print('dataset formed successfully')
            prev_hour = now_hour
            time.sleep(self.sleeptime_sec)

if __name__ == '__main__':
    config = dotenv_values("../.env")
    load_dotenv()
    my_scalper = BybitGlassScalper(chat_id=1145759852,
                                   token=os.getenv('bitscalperasad_token'),
                                   sleeptime_sec=1)
    #my_scalper.run()
    my_scalper.compress_dataset()