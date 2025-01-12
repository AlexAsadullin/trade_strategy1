import telebot # для работы с ботом
from telebot import types # для создания кнопок

import plotly.graph_objects as go 
from collections import defaultdict
from datetime import datetime as dt
from dotenv import load_dotenv
import os

import pybit
#from data_collecting.tinkoff_data import 

load_dotenv()
TOKEN = os.getenv('bitscalperasad_token')
bot = telebot.TeleBot(TOKEN)

USER_DATA = {} 
SYMBOL = 'BTCUSDT'
LIMIT = 10
CATEGORY = 'linear'

def glass_plot(data: dict, key):
    """
    key: 'a' - ask or 'b' - bid
    """
    res = defaultdict(lambda: 0)
    for price, amount in data[key]:
        res[float(price)] += float(amount)
    x_data = list(res.keys())
    y_data = list(res.values())
    figure = go.Figure(
        data=go.Bar(
            x=x_data,
            y=y_data,))
    filepath = f"/home/alex/BitcoinScalper/html_charts/telegram/{data['s']}_{data['ts']}_{key}.html"
    figure.update_layout(
        title=f"Amount vs Price, key='{key}'",
        xaxis_title="Price (USDT)",
        yaxis_title="Amount (BTC)", yaxis_type='log')
    figure.write_html(filepath)
    return filepath

@bot.message_handler(commands=['start'])
def hello(message): 
    USER_DATA['chat_id'] = message.chat.id
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("/glass")
    btn4 = types.KeyboardButton("/history")
    markup.add(btn1, btn4)
    bot.send_message(message.chat.id, f'Hi, choose your instrument for today',
                     reply_markup=markup)  

@bot.message_handler(commands=['history'])
def history(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("")
    btn5 = types.KeyboardButton('/start')
    markup.add(btn1, btn5)
    bot.send_message(message.chat.id, 'Bid panel, choose option:', reply_markup=markup)

@bot.message_handler(commands=['glass'])
def glass_live(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("")
    btn5 = types.KeyboardButton('/start')
    markup.add(btn1, btn5)

    SYMBOL, LIMIT, CATEGORY = 'BTCUSDT', 10, 'linear'
    # pybit
    data = load_data.get_bybit_glass(symbol=SYMBOL, limit=LIMIT, category=CATEGORY)['result']
    SYMBOL, TIMESTAMP = data['s'], data['ts']/1000
    # графики цена\кол-во ask/bid
    bot.send_message(message.chat.id, f'{data['s']} glass at {dt.fromtimestamp(TIMESTAMP)}')
    filepath = glass_plot(data=data, key='a')
    bot.send_document(message.chat.id, open(filepath, 'rb'))
    filepath = glass_plot(data=data, key='b')
    bot.send_document(message.chat.id, open(filepath, 'rb'))
    # итого ask/bid (money, bitcoin)
    money_a = sum([float(x[0]) * float(x[-1]) for x in data['a']])
    money_b = sum([float(x[0]) * float(x[-1]) for x in data['b']])
    bot.send_message(message.chat.id, f"money ask/bid = {money_a / money_b}\nUSDT amount:\nask: {money_a}\nbid: {money_b}")
    btc_a = sum([float(x[-1]) for x in data['a']])
    btc_b = sum([float(x[-1]) for x in data['b']])
    bot.send_message(message.chat.id, f"btc ask/bid{btc_a / btc_b}\nBTC amount:\nask: {btc_a}\nbid: {btc_b}", reply_markup=markup)

# TODO: в одной команде get_current_glass (перименовать) - весь целостный анализ

@bot.message_handler(content_types=['text'])
def all_buttons_response(message):
    bot.send_message(message.chat.id, text="wrong command, please choose from appeared ones")

bot.enable_save_next_step_handlers(delay=1)
bot.load_next_step_handlers()

bot.infinity_polling()

def main():
    print('bot started')