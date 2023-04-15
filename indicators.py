import datetime as dt
import numpy as np
import pandas as pd
from util import get_data, plot_data
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt

def author():
    return "nanderson83"


def sma(df_prices):
    sma = df_prices.rolling(window=20).mean()
    sma.fillna(method="bfill", inplace=True)
    sma_ratio = df_prices/sma
    return sma, sma_ratio


def bollinger_band_percentage(df_prices):
    rolling_mean = df_prices.rolling(window=20, center=False).mean()
    rolling_mean.fillna(method="bfill", inplace=True)
    stdev = df_prices.rolling(window=10,center=False).std()
    stdev.fillna(method="bfill", inplace=True)
    upper_band = rolling_mean + stdev * 2
    lower_band = rolling_mean - stdev * 2

    bb_percentage = (df_prices - lower_band)/(upper_band - lower_band)

    return upper_band, lower_band, bb_percentage


def rate_of_change(df_prices, window=20):
    roc = df_prices / df_prices.shift(window) - 1
    roc.fillna(method="bfill", inplace=True)
    return roc


