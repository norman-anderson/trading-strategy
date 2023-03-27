import pandas as pd


def author():
    return "nanderson83"


def sma(df_prices):
    df_prices /= df_prices[0]
    df_indicators = pd.DataFrame(index=df_prices.index)
    df_indicators['price'] = df_prices
    df_indicators['SMA'] = df_prices.rolling(window=20).mean()
    df_indicators.dropna()
    return df_indicators


def ema(df_prices):
    print(df_prices.iloc[0, 0])
    df_prices /= df_prices.iloc[0, 0]
    df_indicators = pd.DataFrame(index=df_prices.index)
    df_indicators['price'] = df_prices
    df_indicators['EMA 20 days'] = df_prices.ewm(span=20).mean()
    df_indicators['EMA 50 days'] = df_prices.ewm(span=50).mean()
    df_indicators.dropna()
    return df_indicators


def macd(df_prices):
    ema_12, ema_26 = df_prices.ewm(span=12).mean(), df_prices.ewm(span=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9).mean()
    df_indicators = pd.DataFrame(index=df_prices.index)
    #df_indicators['price'] = df_prices
    df_indicators['MACD'] = macd
    df_indicators['MACD Signal'] = macd_signal
    return df_indicators


def bollinger_band_percentage(df_prices):
    rolling_mean = df_prices.rolling(window=10,center=False).mean()
    stdev = df_prices.rolling(window=10,center=False).std()
    upper_band = rolling_mean + stdev * 2
    lower_band = rolling_mean - stdev * 2
    df_indicators = pd.DataFrame(index=df_prices.index)
    bollinger_band = pd.DataFrame(index=df_prices.index)
    bollinger_band['Upper Band'] = upper_band
    bollinger_band['Lower Band'] = lower_band
    bollinger_band['Price'] = df_prices

    bb_percentage = (df_prices - lower_band)/(upper_band - lower_band) * 100
    df_indicators['BB Percentage'] = bb_percentage
    return df_indicators, bollinger_band


def rate_of_change(df_prices, window=14):
    roc = (df_prices - df_prices.shift(window)) / df_prices.shift(window) * 100
    return roc
