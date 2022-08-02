import pandas as pd
import os
from momentum_indicators import *
from overlap_indicators import *
from trend_indicators import *
from volatility_indicators import *
from volume_indicators import *

####################################################################
######################### Technical indicators and preprocess ######
####################################################################

def extract_technical_indicator_features(input_file: str, output_file: str):
    ## read files
    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace("/feature_extraction", ""), input_file)
    df = pd.read_csv(data_file, index_col="date", parse_dates=True)

    ## momentum indicators
    df = ao(df)
    df = cmo(df)
    df = fisher(df)
    df = macd(df)
    df = mom(df)
    df = ppo(df)
    df = roc(df)
    df = rsi(df)
    df = stc(df)
    df = stoch(df)
    df = trix(df)
    df = tsi(df)
    df = willsr(df)

    ## overlap indicators
    df = ema(df, 50)
    df = wma(df)
    df = er(df)

    ## trend indicators
    df = psar(df)
    df = qstick(df)

    ## volatility indicators
    df = tr(df)
    df = donchian(df)

    ## volume indicators
    df = efi(df)
    df = tp(df)

    # write file
    clean_data(df, output_file)

def clean_data(df, output_file: str):
	
    df = df.drop(labels=['tradecount','Volume USDT', 'psarbull', 'psarbear','14 period qstick.', 'unix', 'symbol','macd_signal','ppo_signal','ppo_histo','tsi_signal', 'donchian_lower', 'donchian_uper', '13 period Force Index'], axis=1)
    df = df.dropna(axis=0, how='any')

    df.rename(columns = {'10 period fish.':'fisht', '14 period rsi':'rsi', '10 period stc':'stc','14 period stoch %K':'stoch', '20 period trix':'trix', '14 williams %R':'williams_r', '50 period ema':'ema', '9 period wma.':'wma', '10 period er':'er', 'donchian_middle':'donchian'}, inplace = True)

    df.fillna(method='bfill',inplace=True)

    output_data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)).replace("/feature_extraction", ""), output_file)
    df.to_csv(output_data_file)

if __name__ == "__main__":
    extract_technical_indicator_features("data/raw/Binance_BTCUSDT_minute.csv", "data/processed/Binance_BTCUSDT_minute_processed.csv")
    extract_technical_indicator_features("data/raw/Binance_ETHUSDT_minute.csv", "data/processed/Binance_ETHUSDT_minute_processed.csv")
    extract_technical_indicator_features("data/raw/Binance_LTCUSDT_minute.csv", "data/processed/Binance_LTCUSDT_minute_processed.csv")
   