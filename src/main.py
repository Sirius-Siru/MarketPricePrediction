import pandas as pd
import gc
from feature_engineering import *

DATA_PATH = '../data/BTCUSDT_1h.parquet'

df = pd.read_parquet(DATA_PATH)

################### Feature Selection ############

df['open_time'] = pd.to_datetime(df['open_time'])
df['unix_time'] = (df['open_time'].astype('int64') // 10**9)
df = df.set_index('unix_time')

cols_to_keep = ['high', 'low', 'close', 'open', 'volume']
df = df[cols_to_keep].copy()

################### Feature Engineering ###########

######### Trend

periods = [3,5,8,10,12,14,15,20,21,26,30,34,40,50,100,150,200,250]
ema_df = EMA(df, periods) # EMA

######### Momentum

periods = [14]
rsi_df = RSI(df, periods) # RSI

macd_df = MACD(df, 12, 26, 9) # MACD

periods = [10]
roc_df = ROC(df, periods) # ROC

######### Volatility

periods = [14]
atr_df = ATR(df, periods) # ATR

periods = [20]
rstd_df = rolling_std(df, periods) # Rolling STD

periods = [20]
bb_df = bb_width(df, periods) # Bollinger BandWidth

ADX_df = ADX(df)

######### Volume

periods = [20]
vol_df = VOL(df, periods)

######### Target

periods = [1, 3, 6, 12, 24, 48, 96, 192, 384, 720]
return_df = RETURN(df, periods)
MAE_df = MAE(df, periods)
MFE_df = MFE(df, periods)

######### Merge Indices
df = pd.concat([df, ema_df, rsi_df, macd_df, roc_df,
                atr_df, rstd_df, bb_df, vol_df,
                return_df, MAE_df, MFE_df,
                ADX_df,
                ], axis=1
)

######### EMA Trend Extract
features_df = extract_ema_trend_features(df)

# Primary Signal Generation
signals_df = generate_robust_primary_signals(df)
barrier_df = apply_triple_barrier_explicit(df, signals_df)

######### Add target
df = pd.concat([df, signals_df, barrier_df, features_df,], axis=1)

######### Feature
target_df = pd.concat([signals_df, barrier_df], axis=1)
features_df = pd.concat([df, ema_df, rsi_df, macd_df, roc_df,
                atr_df, rstd_df, bb_df, vol_df,
                return_df, MAE_df, MFE_df,
                ADX_df, features_df,
                ], axis=1
)

del ema_df, rsi_df, macd_df, roc_df, atr_df, rstd_df, bb_df, vol_df, return_df, MAE_df, MFE_df, ADX_df, periods, signals_df, barrier_df, features_df

gc.collect()
