# coding:utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from feature_selection.mcts import MCTS
import time
import os
from feature_selection.feature_selection_env import FeatureSelectionEnv
from constants.market_symbol import *

####################################################################
######################### feature selector #########################
####################################################################

if __name__ == "__main__":

    seed = 12345
    np.random.seed(seed)
    datasets_files = ["data/processed/Binance_BTCUSDT.csv"]
    # datasets_files = ["data/processed/Binance_BTCUSDT.csv", "data/processed/Binance_ETHUSDT.csv", "data/processed/Binance_LTCUSDT.csv"]
    marketSymbols = [MarketSymbol.BTC, MarketSymbol.ETH, MarketSymbol.LTC]
    index = 0
    for filename in datasets_files:
        print(filename)
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        df = pd.read_csv(file_path,header = 0,index_col="date", parse_dates=True)
    
        start = time.time()
        initialState = FeatureSelectionEnv(df.columns)
        mcts = MCTS(iterationLimit=50,X=df,marketSymbol=marketSymbols[index])
        best_subset = mcts.search(initialState=initialState)

        print('best_subset=', best_subset)

        elapsed = time.time()-start
        print('time=',elapsed)
        index += 1
