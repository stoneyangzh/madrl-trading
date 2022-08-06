import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

from util.file_util import *

def process_mcts_parameters_result_file():
    file_name = ["results/mcts/hypers/BTC_DQN_mcts_hypers.txt"
    ,"results/mcts/hypers/BTC_MULTI_AGENT_mcts_hypers.txt"
    ,"results/mcts/hypers/ETH_DQN_mcts_hypers.txt"
    ,"results/mcts/hypers/ETH_MULTI_AGENT_mcts_hypers.txt",
    "results/mcts/hypers/LTC_DQN_mcts_hypers.txt",
    "results/mcts/hypers/LTC_MULTI_AGENT_mcts_hypers.txt"]
    names = ["BTC_DQN", "BTC_MULTI_AGENT", "ETH_DQN", "ETH_MULTI_AGENT","LTC_DQN", "LTC_MULTI_AGENT",]
    the_df = pd.DataFrame.from_dict({})
    index = 0
    for filename in file_name:
        result_str = read_file(filename)
        btc_dqn_mcts_hypers = ast.literal_eval(result_str)
        for iter in btc_dqn_mcts_hypers:
            next_df = pd.DataFrame(btc_dqn_mcts_hypers[iter],index=[names[index]])
            the_df = pd.concat([the_df, next_df])
        index+=1
    the_df.to_csv("results/mcts/hypers/results.csv")

def plot_mcts_hyper_parameters():
    df = pd.read_csv("results/mcts/hypers/results.csv")
   
    print(df.loc[df['market'] == "BTC_DQN"]["best_score"].max())
    print(df.loc[df['market'] == "ETH_DQN"]["best_score"].max())
    print(df.loc[df['market'] == "LTC_DQN"]["best_score"].max())

    print(df.loc[df['market'] == "BTC_MULTI_AGENT"]["best_score"].max())
    print(df.loc[df['market'] == "ETH_MULTI_AGENT"]["best_score"].max())
    print(df.loc[df['market'] == "LTC_MULTI_AGENT"]["best_score"].max())

def calculate_dr():
    number_features = len("psar, tsi, fisht, tp, volume, open, stoch, ppo,williams r, mom, macd, cmo, stc, wma, low,trix, roc, donchian, tr, rsi, ao, close".split(","))
    print("BTC-DQN,",100*(1-(number_features/26)))
    number_features = len("volume, stc, stoch, low, cmo, macd, psar, tr,williams r, wma, trix, donchian, ao, ema, close,tsi,roc, high, tp, fisht, ppo, rsi".split(","))
    print("ETH-DQN,",100*(1-(number_features/26)))
    number_features = len("fisht, rsi, ema, williams r, stoch, mom, er, trix,cmo, macd, roc, wma, low, high, tsi, tr, tp, open,ppo, stc, psar, donchian, close".split(","))
    print("LTC-DQN,",100*(1-(number_features/26)))

    number_features = len("williams r, cmo, ema, trix, high, macd, psar,stc, ppo, close, tr, volume, tp, low, fisht, wma,donchian, mom, ao, er, rsi, stoch".split(","))
    print("BTC-MULTI-AGENT,",100*(1-(number_features/26)))
    number_features = len("volume, macd, fisht, high, stoch, cmo, ppo, tp,trix, rsi, roc, er, williams r, tsi, close, tr, ao,psar, stc, wma, mom, donchian".split(","))
    print("ETH-MULTI-AGENT,",100*(1-(number_features/26)))
    number_features = len("open, er, stc, fisht, ppo, tsi, close, ao, mom,low, stoch, roc, wma, tp, psar, trix, macd, cmo,williams r, ema, rsi, donchian".split(","))
    print("LTC-MULTI-AGENT,",100*(1-(number_features/26)))

def plot_dr():
    


    df = pd.read_csv("results/mcts/dr.csv")
    np.random.seed(19680801)


    # plt.rcdefaults()
    fig, ax = plt.subplots()
    names = list(df["agent-data"])
    values = list(df["dr"])
    fig, axs = plt.subplots(1, 1, figsize=(12, 8), sharey=True)
    axs.bar(names, values, width=0.3)

    
    # axs[1].scatter(names, values)
    # axs[2].plot(names, values)
    # fig.suptitle('Categorical Plotting')
    axs.set_ylabel("Dimension Reduction Ratio(%)")
    axs.set_xlabel("Dataset and trading agent")

if __name__ == "__main__":
    #process_mcts_parameters_result_file()
    # plot_mcts_hyper_parameters()
    # calculate_dr()
    plot_dr()