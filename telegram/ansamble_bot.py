import telebot # для работы с ботом
from telebot import types # для создания кнопок

import pandas as pd
import plotly.graph_objects as go 
from collections import defaultdict
from datetime import datetime as dt
from dotenv import load_dotenv
import os
import sys
from tinkoff.invest import CandleInterval

load_dotenv()
TOKEN = os.getenv('ansamble_invest_bot')
bot = telebot.TeleBot(TOKEN)
USER_DATA = {}
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ML.ansamble.pipeline import load_tinkoff
from ML.ansamble.ai import ensemble_predict

def get_figi(name: str, instrument: str):
    df = pd.read_csv(r'/home/alex/BitcoinScalper/data_collecting/tinkoff_data/tickers_figi.csv')
    if instrument == 'stocks':
        result = df.loc[(df['Type'] == 'shares') & (df['Name'].str.contains(name, case=False, na=False)), ['Name', 'Figi', 'Ticker']]
    elif instrument == 'bonds':
        result = df.loc[(df['Type'] == 'bonds') & (df['Name'].str.contains(name, case=False, na=False)), ['Name', 'Figi', 'Ticker']]
    elif instrument == 'etfs':
        result = df.loc[(df['Type'] == 'etfs') & (df['Name'].str.contains(name, case=False, na=False)), ['Name', 'Figi', 'Ticker']]
    else: return

    return result.values[0] if len(result) > 0 else None


@bot.message_handler(commands=['start'])
def hello(message): 
    global USER_DATA
    USER_DATA['chat_id'] = message.chat.id
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("/stocks") # акции
    btn2 = types.KeyboardButton("/bonds") # облигации
    btn3 = types.KeyboardButton("/etfs") # фонды - ПИФ
    markup.add(btn1, btn2, btn3)
    bot.send_message(message.chat.id, f'Привет! выбери инструмент',
                     reply_markup=markup)  

@bot.message_handler(commands=['stocks'])
def stocks(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn3 = types.KeyboardButton("/start")
    markup.add(btn3)
    global USER_DATA
    USER_DATA['instrument_type'] = 'stocks'
    msg = bot.send_message(message.chat.id, 'Введи имя компании:', reply_markup=markup)
    bot.register_next_step_handler(msg, find_figi)

@bot.message_handler(commands=['bonds'])
def bonds(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn3 = types.KeyboardButton("/start")
    markup.add(btn3)
    global USER_DATA
    USER_DATA['instrument_type'] = 'bonds'
    msg = bot.send_message(message.chat.id, 'Введи имя компании:', reply_markup=markup)
    bot.register_next_step_handler(msg, find_figi)

@bot.message_handler(commands=['etfs'])
def etfs(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn3 = types.KeyboardButton("/start")
    markup.add(btn3)
    global USER_DATA
    USER_DATA['instrument_type'] = 'etfs'
    msg = bot.send_message(message.chat.id, 'Введи имя компании:', reply_markup=markup)
    bot.register_next_step_handler(msg, find_figi)

def find_figi(message):
    global USER_DATA
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn2 = types.KeyboardButton("/period") 
    btn3 = types.KeyboardButton("/start") 
    figi = get_figi(message.text, USER_DATA['instrument_type'])
    
    if figi is not None:
        markup.add(btn3, btn2)
        USER_DATA['company'] = figi[0]
        USER_DATA['figi'] = figi[1]
        USER_DATA['ticker'] = figi[2]
        print(figi[1])
        bot.send_message(message.chat.id, f'{figi}\n/period - продолжить\n/start - отмена', reply_markup=markup)
    else:
        markup.add(btn3)
        bot.send_message(message.chat.id, 'Ничего не найдено!\n/start - отмена', reply_markup=markup)

@bot.message_handler(commands=['period'])
def period(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn3 = types.KeyboardButton("/start")
    markup.add(btn3)
    msg = bot.send_message(message.chat.id, 'Введите период (число - количество дней):', reply_markup=markup)
    bot.register_next_step_handler(msg, find_period)

def find_period(message):
    try:
        global USER_DATA
        USER_DATA['n_days'] = int(message.text)

        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn2 = types.KeyboardButton("/10MIN") 
        btn3 = types.KeyboardButton("/30MIN")
        btn4 = types.KeyboardButton("/1HOUR")
        btn5 = types.KeyboardButton("/2HOUR")
        btn6 = types.KeyboardButton("/1DAY")
        btn7 = types.KeyboardButton("/1WEEK")
        btn1 = types.KeyboardButton("/start")
        markup.add(btn2, btn3, btn4, btn5, btn6, btn7, btn1)
        
        bot.send_message(message.chat.id, 'Выбери интервал Таймфрейма', reply_markup=markup)
    except Exception as e:
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton("/start")
        markup.add(btn1)
        bot.send_message(message.chat.id, 'Пожалуйста, введите число', reply_markup=markup)

@bot.message_handler(commands=['10MIN'])
def ten_min(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn2 = types.KeyboardButton("/predict10min")
    btn1 = types.KeyboardButton("/start")
    global USER_DATA
    USER_DATA['interval'] = CandleInterval.CANDLE_INTERVAL_10_MIN
    markup.add(btn2, btn1)
    bot.send_message(message.chat.id, '/predict - получить предсказание модели\n/start - отмена', reply_markup=markup)

@bot.message_handler(commands=['30MIN'])
def thirty_min(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn2 = types.KeyboardButton("/predict30min")
    btn1 = types.KeyboardButton("/start")
    global USER_DATA
    USER_DATA['interval'] = CandleInterval.CANDLE_INTERVAL_30_MIN
    markup.add(btn2, btn1)
    bot.send_message(message.chat.id, '/predict - получить предсказание модели\n/start - отмена', reply_markup=markup)

@bot.message_handler(commands=['1HOUR'])
def one_hour(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn2 = types.KeyboardButton("/predict1hour")
    btn1 = types.KeyboardButton("/start")
    global USER_DATA
    USER_DATA['interval'] = CandleInterval.CANDLE_INTERVAL_HOUR
    markup.add(btn2, btn1)
    bot.send_message(message.chat.id, '/predict - получить предсказание модели\n/start - отмена', reply_markup=markup)

@bot.message_handler(commands=['2HOUR'])
def two_hour(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn2 = types.KeyboardButton("/predict2hour")
    btn1 = types.KeyboardButton("/start")
    global USER_DATA
    USER_DATA['interval'] = CandleInterval.CANDLE_INTERVAL_2_HOUR
    markup.add(btn2, btn1)
    bot.send_message(message.chat.id, '/predict - получить предсказание модели\n/start - отмена', reply_markup=markup)

@bot.message_handler(commands=['1DAY'])
def one_day(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn2 = types.KeyboardButton("/predict1day")
    btn1 = types.KeyboardButton("/start")
    global USER_DATA
    USER_DATA['interval'] = CandleInterval.CANDLE_INTERVAL_DAY
    markup.add(btn2, btn1)
    bot.send_message(message.chat.id, '/predict - получить предсказание модели\n/start - отмена', reply_markup=markup)

@bot.message_handler(commands=['1WEEK'])
def one_week(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn2 = types.KeyboardButton("/predict1week")
    btn1 = types.KeyboardButton("/start")
    global USER_DATA
    USER_DATA['interval'] = CandleInterval.CANDLE_INTERVAL_WEEK
    markup.add(btn1, btn2)
    bot.send_message(message.chat.id, '/predict - получить предсказание модели\n/start - отмена', reply_markup=markup)

@bot.message_handler(commands=['predict10min'])
def min_10(message):
    predict(r"/home/alex/BitcoinScalper/ML/ansamble/10_min")

@bot.message_handler(commands=['predict30min'])
def min_30(message):
    predict(r"/home/alex/BitcoinScalper/ML/ansamble/")

@bot.message_handler(commands=['predict1hour'])
def hour_1(message):
    predict(r"/home/alex/BitcoinScalper/ML/ansamble/")

@bot.message_handler(commands=['predict2hour'])
def hour_2(message):
    predict(r"/home/alex/BitcoinScalper/ML/ansamble/2_hour")

@bot.message_handler(commands=['predict1day'])
def day_1(message):
    predict(r"/home/alex/BitcoinScalper/ML/ansamble/1_day")

@bot.message_handler(commands=['predict1week'])
def week_1(message):
    predict(r"/home/alex/BitcoinScalper/ML/ansamble/1_week")


def predict(models_dir_path):
    global USER_DATA
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("/start")
    markup.add(btn1)
    bot.send_message(USER_DATA['chat_id'], 'минутку, скачиваю данные...', reply_markup=markup)
    df = load_tinkoff(figi=USER_DATA['figi'], days_back_begin=USER_DATA['n_days'], interval=USER_DATA['interval'])

    make_plots(df)
    bot.send_message(USER_DATA['chat_id'], 'начинаю рассчет...', reply_markup=markup)
    predictions, final_desicion = launch_ansamble(df, models_dir_path) # '/home/alex/BitcoinScalper/ML/ansamble/trained_models'
    bot.send_message(USER_DATA['chat_id'], f"финальное решение моделей: {final_desicion}\nголоса: {list(predictions)}",
                     reply_markup=markup)

def make_plots(df: pd.DataFrame):
    global USER_DATA
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode='lines',
        line=dict(color="blue"), name='Цена закрытия'))
    if USER_DATA['instrument_type'] == 'stocks': ins_type = 'акций'
    elif USER_DATA['instrument_type'] == 'bonds': ins_type = 'облигаций'
    else: ins_type = 'ETF фонда'
    fig.update_layout(title=f"График цены {ins_type} \"{USER_DATA['company']}\" (Close)",
    xaxis_title=f"Дата (0 = {USER_DATA['n_days']} дней назад)", yaxis_title="Цена",
    legend_title="Обозначения")

    df_path = rf'/home/alex/BitcoinScalper/telegram/temp/{USER_DATA['ticker']}_{USER_DATA['n_days']}_{USER_DATA['instrument_type']}_{USER_DATA['figi']}.csv'
    fig_path = rf'/home/alex/BitcoinScalper/telegram/temp/{USER_DATA['ticker']}_{USER_DATA['n_days']}_{USER_DATA['instrument_type']}_{USER_DATA['figi']}.html'
    df.to_csv(df_path)
    fig.write_html(fig_path)

    with open(df_path, 'rb') as f:
        bot.send_document(USER_DATA['chat_id'], f)
    with open(fig_path, 'rb') as f:
        bot.send_document(USER_DATA['chat_id'], f)
    os.remove(df_path)
    os.remove(fig_path)

def launch_ansamble(df: pd.DataFrame, models_dir_path: str):
    predictions, final_desicion = ensemble_predict(
        df=df,
        models_dir_path=models_dir_path
    )
    return predictions, final_desicion


bot.enable_save_next_step_handlers(delay=1)
bot.load_next_step_handlers()

bot.infinity_polling()