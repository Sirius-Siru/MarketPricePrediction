import numpy as np
import pandas as pd
import talib


# Trend 
def EMA(df, periods_list):
    ema_df = pd.DataFrame(index=df.index)
    for period in periods_list:
        ema_df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    return ema_df


# Momentum
def RSI(df, lengths):
    rsi_df = pd.DataFrame(index=df.index)
    for length in lengths:
        rsi_df[f'RSI_{length}'] = talib.RSI(df['close'], timeperiod=length)
    return rsi_df


def MACD(df, fast, slow, signal):
    macd_df = pd.DataFrame(index=df.index)
    macd_df['MACD'], macd_df['Signal'], macd_df['Hist'] = talib.MACD(
        df['close'],
        fastperiod=fast,
        slowperiod=slow,
        signalperiod=signal,
    )
    return macd_df


def ROC(df, periods):
    roc_df = pd.DataFrame(index=df.index)
    for period in periods:
        roc_df[f'ROC_{period}'] = talib.ROC(df['close'], timeperiod=period)
    return roc_df


# Volatility
def ATR(df, periods):
    high, low, close = df['high'], df['low'], df['close']
    atr_df = pd.DataFrame(index=df.index)
    for period in periods:
        atr_df[f'ATR_{period}'] = talib.ATR(high, low, close, timeperiod=period)
    return atr_df


def rolling_std(df, periods):
    rstd_df = pd.DataFrame(index=df.index)
    for period in periods:
        rstd_df[f'Rolling_Std_{period}'] = talib.STDDEV(df['close'], timeperiod=period, nbdev=1.0)
    return rstd_df


def bb_width(df, periods):
    bb_df = pd.DataFrame(index=df.index)
    for period in periods:
        upper, middle, lower = talib.BBANDS(
            df['close'], timeperiod=period, nbdevup=2, nbdevdn=2, matype=0
        )
        bb_df[f'BB_Width_{period}'] = (upper - lower) / middle
    return bb_df


# Volume
def VOL(df, periods):
    vol_df = pd.DataFrame(index=df.index)
    vol_df['Vol_Change_Pct'] = df['volume'].pct_change() * 100
    vol_df['Vol_Change_Abs'] = df['volume'].diff()

    for period in periods:
        vol_df[f'Vol_SMA_{period}'] = talib.SMA(df['volume'], timeperiod=period)
    return vol_df


# Not indicator
def RETURN(df, periods):
    return_df = pd.DataFrame(index=df.index)
    return_df['High_Low_Range_Pct'] = ((df['high'] - df['low']) / df['low']) * 100
    return_df['Close_Open_Return_Pct'] = ((df['close'] - df['open']) / df['open']) * 100
    
    for n in periods:
        return_df[f'Simple_Return_{n}'] = df['close'].pct_change(-n) * -1
        return_df[f'Log_Return_{n}'] = np.log(df['close'].shift(-n) / df['close'])
    return return_df


def MFE(df, periods):
    MFE_df = pd.DataFrame(index=df.index)
    for N in periods:
        MFE_df[f'MFE_{N}'] = (df['high'].rolling(N).max().shift(-N+1) - df['close']) / df['close']
    return MFE_df


def MAE(df, periods):
    MAE_df = pd.DataFrame(index=df.index)
    for N in periods:
        MAE_df[f'MAE_{N}'] = (df['close'] - df['low'].rolling(N).min().shift(-N+1)) / df['close']
    return MAE_df


def ADX(df):
    ADX_df = pd.DataFrame(index=df.index)
    ADX_df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    return ADX_df


# Primary Signal Generation
def generate_robust_primary_signals(df):
    macd = df['MACD']
    macd_hist = df['Hist']
    rsi_14 = df['RSI_14']

    macd_q20 = macd.rolling(200).quantile(0.20)
    macd_q80 = macd.rolling(200).quantile(0.80)
    is_macd_neutral = (macd >= macd_q20) & (macd <= macd_q80)

    long_cond = (is_macd_neutral & (macd_hist > 0)) | (rsi_14 >= 70)
    short_cond = (rsi_14 <= 30) & (macd_hist < 0)

    # Avoid conflict by checking conditions explicitly
    conditions = [long_cond & ~short_cond, short_cond & ~long_cond]
    choices = [1, -1]
    
    primary_signal = pd.Series(
        np.select(conditions, choices, default=0), 
        index=df.index, 
        name='Primary_Signal'
    )

    return pd.DataFrame({'Primary_Signal': primary_signal}, index=df.index)


# Optimized Triple Barrier with NumPy
def apply_triple_barrier_explicit(df, signals_df, ptp=2.0, psl=1.0, horizon=14):
    signals = signals_df['Primary_Signal'].values
    atr = df['ATR_14'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n = len(df)

    barrier_labels = np.full(n, np.nan)
    signal_indices = np.where(signals != 0)[0]

    for idx in signal_indices:
        sig = signals[idx]
        entry = close[idx]
        a = atr[idx]

        if np.isnan(a) or a == 0:
            continue

        if sig == 1:
            tp = entry + (ptp * a)
            sl = entry - (psl * a)
        else:
            tp = entry - (ptp * a)
            sl = entry + (psl * a)

        end_idx = min(idx + 1 + horizon, n)
        sub_high = high[idx + 1:end_idx]
        sub_low = low[idx + 1:end_idx]

        final_label = 0  # Timeout

        for h, l in zip(sub_high, sub_low):
            if sig == 1:
                if l <= sl:
                    final_label = -1
                    break
                if h >= tp:
                    final_label = 1
                    break
            elif sig == -1:
                if h >= sl:
                    final_label = -1
                    break
                if l <= tp:
                    final_label = 1
                    break

        barrier_labels[idx] = final_label

    barrier_df = pd.DataFrame({
        'Barrier_Label': barrier_labels,
        'Meta_Target_Binary': np.where(
            np.isnan(barrier_labels), 
            np.nan, 
            np.where(barrier_labels == 1, 1, 0)
        )
    }, index=df.index)

    return barrier_df


def extract_ema_trend_features(df):
    close = df['close']
    atr = df['ATR_14']
    ema_fast = df['EMA_12']
    ema_slow = df['EMA_250']

    return pd.DataFrame({
        'EMA_Distance_12_250': (ema_fast - ema_slow) / atr,
        'EMA_250_Slope': (ema_slow - ema_slow.shift(5)) / atr,
        'Price_to_EMA250': (close - ema_slow) / atr
    }, index=df.index)