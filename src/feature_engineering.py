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
    high = df['high']
    low = df['low']
    close = df['close']
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
            df['close'], timeperiod=period, nbdevup=2, nbdevdn=2, matype=0,
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

    return_df['High_Low_Range_Pct'] = (
        (df['high'] - df['low']) / df['low']
    ) * 100

    return_df['Close_Open_Return_Pct'] = (
        (df['close'] - df['open']) / df['open']
    ) * 100
    
    for n in periods:
        # Simple Return
        return_df[f'Simple_Return_{n}'] = df['close'].pct_change(-n) * -1

        # Log Return 
        return_df[f'Log_Return_{n}'] = np.log(
            df['close'].shift(-n) / df['close']
        ) 

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

# Primary Signal Generation from MACD & RSI
def generate_robust_primary_signals(df):
    """
    Tạo tín hiệu Mua/Bán thô và chỉ báo liên quan dưới dạng DataFrame độc lập.
    Không chỉnh sửa DataFrame df truyền vào.
    """
    close = df['close']
    high = df['high']
    low = df['low']

    # 1. Tính toán các chỉ báo bằng Series độc lập
    macd = df['MACD']
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    # RSI 14
    rsi_14 = df['RSI_14']

    # ATR 14
    atr_14 = df['ATR_14']

    # 2. Định nghĩa MACD Bin Q3 (Neutral)
    macd_q20 = macd.rolling(200).quantile(0.20)
    macd_q80 = macd.rolling(200).quantile(0.80)
    is_macd_neutral = (macd >= macd_q20) & (macd <= macd_q80)

    # 3. Kích hoạt Primary Signal
    primary_signal = pd.Series(0, index=df.index, name='Primary_Signal')

    long_condition = (is_macd_neutral & (macd_hist > 0)) | (rsi_14 >= 70)
    short_condition = (rsi_14 <= 30) & (macd_hist < 0)

    primary_signal.loc[long_condition] = 1
    primary_signal.loc[short_condition] = -1

    # 4. Xuất ra DataFrame riêng biệt
    signals_df = pd.DataFrame({
        'MACD_Hist': macd_hist,
        'Primary_Signal': primary_signal
    }, index=df.index)

    return signals_df

# Tripple Barrier
def apply_triple_barrier_explicit(df, signals_df, ptp=2.0, psl=1.0, horizon=14):
    """
    Gán nhãn Triple Barrier và trả về DataFrame k ết quả độc lập.
    Yêu cầu truyền vào:
    - df: chứa giá (close, high, low)
    - signals_df: chứa 'Primary_Signal' và 'ATR_14'
    """
    signals = signals_df['Primary_Signal']
    atr_series = df['ATR_14']
    signal_indices = signals[signals != 0].index

    labels = []

    for idx in signal_indices:
        signal = signals.loc[idx]
        entry_price = df.loc[idx, 'close']
        atr = atr_series.loc[idx]

        if pd.isna(atr) or atr == 0:
            labels.append((idx, np.nan))
            continue

        # Cài đặt TP/SL
        if signal == 1:
            tp_price = entry_price + (ptp * atr)
            sl_price = entry_price - (psl * atr)
        else:
            tp_price = entry_price - (ptp * atr)
            sl_price = entry_price + (psl * atr)

        loc_i = df.index.get_loc(idx)
        future_df = df.iloc[loc_i + 1 : loc_i + 1 + horizon]

        final_label = 0  # Default: Timeout

        for _, row in future_df.iterrows():
            high, low = row['high'], row['low']

            if signal == 1:
                if low <= sl_price:
                    final_label = -1
                    break
                if high >= tp_price:
                    final_label = 1
                    break
            elif signal == -1:
                if high >= sl_price:
                    final_label = -1
                    break
                if low <= tp_price:
                    final_label = 1
                    break

        labels.append((idx, final_label))

    # Tạo DataFrame kết quả và căn chỉnh chuẩn Index theo df gốc
    barrier_df = pd.DataFrame(labels, columns=['index', 'Barrier_Label']).set_index('index')
    barrier_df = barrier_df.reindex(df.index)

    # Phân loại Nhị phân cho Machine Learning Filter (1: Win, 0: Loss/Timeout)
    barrier_df['Meta_Target_Binary'] = barrier_df['Barrier_Label'].apply(
        lambda x: 1 if x == 1 else (0 if pd.notna(x) else np.nan)
    )

    return barrier_df