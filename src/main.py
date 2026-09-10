import pandas as pd
import gc
from feature_engineering import *

DATA_PATH = '../data/BTCUSDT_1h.parquet'

# 1. Load Data
df = pd.read_parquet(DATA_PATH)
df['open_time'] = pd.to_datetime(df['open_time'])
df['unix_time'] = df['open_time'].astype('int64') // 10**9
df = df.set_index('unix_time')

cols_to_keep = ['high', 'low', 'close', 'open', 'volume']
df = df[cols_to_keep].copy()

# 2. Base Features Calculation
ema_periods = [3, 5, 8, 10, 12, 14, 15, 20, 21, 26, 30, 34, 40, 50, 100, 150, 200, 250]
ema_df = EMA(df, ema_periods)
rsi_df = RSI(df, [14])
macd_df = MACD(df, 12, 26, 9)
roc_df = ROC(df, [10])

atr_df = ATR(df, [14])
rstd_df = rolling_std(df, [20])
bb_df = bb_width(df, [20])
ADX_df = ADX(df)
vol_df = VOL(df, [20])

return_periods = [1, 3, 6, 12, 24, 48, 96, 192, 384, 720]
return_df = RETURN(df, return_periods)
MAE_df = MAE(df, return_periods)
MFE_df = MFE(df, return_periods)

# Concatenate Base Features
features_df = pd.concat([
    df, ema_df, rsi_df, macd_df, roc_df,
    atr_df, rstd_df, bb_df, vol_df,
    return_df, MAE_df, MFE_df, ADX_df
], axis=1)

# 3. Derived Features & Target Generation
ema_trend_df = extract_ema_trend_features(features_df)
signals_df = generate_robust_primary_signals(features_df)
barrier_df = apply_triple_barrier_explicit(features_df, signals_df)

target_df = pd.concat([signals_df, barrier_df], axis=1)

# 4. Assemble Final Dataset
df = pd.concat([features_df, ema_trend_df, target_df], axis=1)

# 5. Extract Raw Baseline
baseline_summary = extract_raw_baseline(df['Meta_Target_Binary'])

# Memory Cleanup
del ema_df, rsi_df, macd_df, roc_df, atr_df, rstd_df, bb_df, vol_df, return_df, MAE_df, MFE_df, ADX_df, ema_trend_df, signals_df, barrier_df, target_df, features_df
gc.collect()