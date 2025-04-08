from enum import Enum
from tinkoff.invest import CandleInterval

# Маппинг числовых значений в CandleInterval (если нужно поддерживать оба варианта)
CANDLE_INTERVAL_MAP = {
    "5m": CandleInterval.CANDLE_INTERVAL_5_MIN,
    "10m": CandleInterval.CANDLE_INTERVAL_10_MIN,
    "15m": CandleInterval.CANDLE_INTERVAL_15_MIN,
    "30m": CandleInterval.CANDLE_INTERVAL_30_MIN,
    "1H": CandleInterval.CANDLE_INTERVAL_HOUR,
    "2H": CandleInterval.CANDLE_INTERVAL_2_HOUR,
    "4H": CandleInterval.CANDLE_INTERVAL_4_HOUR,
    "1D": CandleInterval.CANDLE_INTERVAL_DAY,
    "1W": CandleInterval.CANDLE_INTERVAL_WEEK,
    "1M": CandleInterval.CANDLE_INTERVAL_MONTH,
}
