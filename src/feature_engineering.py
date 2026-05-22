import pandas as pd
import numpy as np


def calculate_ema(close_price, period):
    n = close_price.shape[0]
    ema = np.empty(n)
    ema[0] = close_price.iloc[0]

    #Hệ số làm mượt
    s = 2

    for i in range(1, n):
        ema[i] = (close_price.iloc[i] * (s/(1+period))) + ema[i-1] * (1 - (s/(1+period)))

    return pd.Series(ema)

def calculate_sma(close_price, period):
    n = close_price.shape[0]

    sma = close_price.rolling(window=period).mean()
    sma = pd.Series(sma).fillna(0)

    return sma

def calculate_macd(ema12, ema26):
    return ema12 - ema26

def calculate_rsi(close_price, period):
    n = close_price.shape[0]

    delta = close_price.diff()

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return pd.Series(rsi).fillna(0)
        
def calculate_bollinger_bands(close_price, middle_band, period):
    rolling_std = close_price.rolling(window=period).std()

    upper_band = middle_band + (2 * rolling_std)
    lower_band = middle_band - (2 * rolling_std)

    return upper_band, lower_band


def calculate_atr(df, period):
    n = df.shape[0]
    tr = np.empty(n)
    tr[0] = df['high'].iloc[0] - df['low'].iloc[0]

    for i in range(1, n):
        tr1 = df['high'].iloc[i] - df['low'].iloc[i]
        tr2 = abs(df['high'].iloc[i] - df['close'].iloc[i-1])
        tr3 = abs(df['low'].iloc[i] - df['close'].iloc[i-1])

        tr = max([tr1, tr2, tr3])

    atr = pd.Series(tr).rolling(window=period).mean()
    return atr


def calculate_indices(df, periods_list):
    result = []

    return ema_df

