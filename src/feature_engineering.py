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

def create_ema_list(close_price, periods_list):
    result = []
    for i in periods_list:
        ema_series = calculate_ema(close_price, i)
        ema_series.name = f'ema_{i}'
        ema_series.index = close_price.index
        result.append(ema_series)

    ema_df = pd.concat(result, axis=1)
    return ema_df