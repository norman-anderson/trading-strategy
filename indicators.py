import datetime as dt
import pandas as pd
from util import get_data

def author():
    return "nanderson83"


def ema(sd, ed, symbol, window_size=20):
    refined_sd = sd - dt.timedelta(window_size * 2)
    df_price = get_data([symbol], pd.date_range(refined_sd, ed))
    df_price = df_price[[symbol]]
    df_price = df_price.ffill().bfill()
    df_ema = df_price.ewm(span=window_size).mean()
    df_ema = df_ema.truncate(before=sd)

    return df_ema


def macd(sd, ed, symbol):
    refined_sd = sd - dt.timedelta(52)
    df_price = get_data([symbol], pd.date_range(refined_sd, ed))
    df_price = df_price[[symbol]]
    df_price = df_price.ffill().bfill()

    ema_12 = df_price.ewm(span=12, adjust=False).mean()
    ema_26 = df_price.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd = macd.truncate(before=sd)
    macd_signal = macd_signal.truncate(before=sd)

    return macd, macd_signal


def roc(sd, ed, symbol):
    refined_sd = sd - dt.timedelta(70)
    df_price = get_data([symbol], pd.date_range(refined_sd, ed))
    df_price = df_price[[symbol]]
    df_price = df_price.ffill().bfill()

    diff = df_price - df_price.shift(1)
    ema_25 = diff.ewm(span=25, adjust=False).mean()
    ema_13 = ema_25.ewm(span=13, adjust=False).mean()

    abs_diff = abs(diff)
    abs_ema_25 = abs_diff.ewm(span=25, adjust=False).mean()
    abs_ema_13 = abs_ema_25.ewm(span=13, adjust=False).mean()

    roc = ema_13 / abs_ema_13
    roc = roc.truncate(before=sd)

    return roc


