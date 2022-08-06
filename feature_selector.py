# coding:utf-8
import numpy as np
import pandas as pd
from constants.agent_type import AgentType
from feature_selection.mcts import MCTS
import time
import os
from feature_selection.feature_selection_env import FeatureSelectionEnv
from constants.market_symbol import *
from util.file_util import *
from random import choice

####################################################################
######################### feature selector #########################
####################################################################

def tune_hyperparameters():
    """
    Do the feature selection
    """
    iterations = [50, 100, 150, 200, 250]
    expand_widths = [3, 5,  8,  10,  12]
    search_depths = [10, 15, 18, 20, 22]

    #set seed for MCTS to randomly expand the search tree
    total_time_steps = 10000 * 100 # default 1 million

    datasets_files = ["data/processed/Binance_BTCUSDT_minute_processed.csv", "data/processed/Binance_ETHUSDT_minute_processed.csv", "data/processed/Binance_LTCUSDT_minute_processed.csv"]
    marketSymbols = [MarketSymbol.BTC, MarketSymbol.ETH, MarketSymbol.LTC]
    markers_index = 0
    for filename in datasets_files:
        print(filename)
        
        agents = [AgentType.DQN, AgentType.MULTI_AGENT]
        learning_rates = [1e-4, 7e-4]

        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        df = pd.read_csv(file_path,header = 0,index_col="date", parse_dates=True)
        total_time_steps = len(df.columns)

        agents_index = 0
        for agent_type in agents:
            result_dict = {}
            for i in range(20):
                iteration = choice(iterations)
                expand_width = choice(expand_widths)
                search_depth = choice(search_depths)
                best_score_best_features = {}

                start = time.time()
                initialState = FeatureSelectionEnv(df.columns, agent_type, total_time_steps,learning_rates[agents_index], search_depth=search_depth)
                mcts = MCTS(iterationLimit=iteration,X=df,marketSymbol=marketSymbols[markers_index], expand_width=expand_width)
                best_feature_subset,best_score, features_and_reward = mcts.search(initialState=initialState)
                
                elapsed = time.time() - start
                print('time=',elapsed)

                best_score_best_features["iteration"] = iteration
                best_score_best_features["expand_width"] = expand_width
                best_score_best_features["search_depth"] = search_depth
                best_score_best_features["best_score"] = best_score
                best_score_best_features["time"] = elapsed

                result_dict["epoch-{}".format(i+1)] = best_score_best_features

            agents_index += 1
            save_to_file("results/mcts/hypers/{}{}{}{}".format(marketSymbols[markers_index].name,"_", agent_type.name,"_mcts_hypers.txt"), result_dict)
        markers_index += 1


def run_feature_selection():
    """
    Do the feature selection
    """
    #set seed for MCTS to randomly expand the search tree
    seed = 12345
    np.random.seed(seed)
    total_time_steps = 10000 * 100 # default 1 million

    datasets_files = ["data/processed/Binance_BTCUSDT_minute_processed.csv", "data/processed/Binance_ETHUSDT_minute_processed.csv", "data/processed/Binance_LTCUSDT_minute_processed.csv"]
    marketSymbols = [MarketSymbol.BTC, MarketSymbol.ETH, MarketSymbol.LTC]
    markers_index = 0
    for filename in datasets_files:
        print(filename)
        
        agents = [AgentType.DQN, AgentType.MULTI_AGENT]
        learning_rates = [1e-4, 7e-4]

        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        df = pd.read_csv(file_path,header = 0,index_col="date", parse_dates=True)
        total_time_steps = len(df.columns)

        agents_index = 0
        for agent_type in agents:
            best_score_best_features = {}

            start = time.time()
            initialState = FeatureSelectionEnv(df.columns, agent_type, total_time_steps,learning_rates[agents_index])
            mcts = MCTS(iterationLimit=1000,X=df,marketSymbol=marketSymbols[markers_index])
            best_feature_subset,best_score, features_and_reward = mcts.search(initialState=initialState)
            
            elapsed = time.time() - start
            print('time=',elapsed)

            best_score_best_features["best_features"] = best_feature_subset
            best_score_best_features["best_score"] = best_score
            best_score_best_features["time"] = elapsed

            # best_features_df = pd.DataFrame.from_dict(best_score_best_features)
            # best_features_file_name = "results/mcts/{}{}{}".format(marketSymbols[markers_index],agent_type.name,"_best_features.csv")
            # best_features_df.to_csv(best_features_file_name)
    
            # features_and_reward_df = pd.DataFrame.from_dict(features_and_reward)
            # features_and_reward_file_name = "results/mcts/{}{}{}".format(marketSymbols[markers_index],agent_type.name,"_with_reward.csv")
            # features_and_reward_df.to_csv(features_and_reward_file_name)
            save_to_file("results/mcts/{}{}{}{}".format(marketSymbols[markers_index].name,"_", agent_type.name,"_best_features.txt"),best_score_best_features)
            save_to_file("results/mcts/{}{}{}{}".format(marketSymbols[markers_index].name, "_", agent_type.name,"_with_reward.txt"), features_and_reward)
            agents_index += 1
        markers_index += 1

if __name__ == "__main__":
    # run_feature_selection()
    tune_hyperparameters()
