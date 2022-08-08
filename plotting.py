import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

from util.file_util import *
import matplotlib.dates as mdates

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
def process_hyper_parameters_agents_result_file():
    file_name = ["results/trading/hypers/BTC_DQN_trading_hypers.txt"
    ,"results/trading/hypers/BTC_MULTI_AGENT_trading_hypers.txt"
    ,"results/trading/hypers/ETH_DQN_trading_hypers.txt"
    ,"results/trading/hypers/ETH_MULTI_AGENT_trading_hypers.txt",
    "results/trading/hypers/LTC_DQN_trading_hypers.txt",
    "results/trading/hypers/LTC_MULTI_AGENT_trading_hypers.txt"]
    names = ["BTC_DQN", "BTC_MULTI_AGENT", "ETH_DQN", "ETH_MULTI_AGENT","LTC_DQN", "LTC_MULTI_AGENT"]
    the_df = pd.DataFrame.from_dict({})
    index = 0
    for filename in file_name:
        result_str = read_file(filename)
        btc_dqn_agents_hypers = ast.literal_eval(result_str)
        for iter in btc_dqn_agents_hypers:
            epoch = btc_dqn_agents_hypers[iter]
            per_table = epoch["performance_table"]
            hyper_params = "learning_rate:{}-dicount_factor:{}-total_time_step:{}".format( epoch["learning_rate"], epoch["dicount_factor"], epoch["total_time_step"])
            epoch["hyper_parameters"] = hyper_params
            epoch["sharp_ratio"] = per_table[3][1]
            epoch["training_mins"] = epoch["time"]
            del epoch["performance_table"]
            del epoch["time"]
            next_df = pd.DataFrame(epoch,index=[names[index]])
            the_df = pd.concat([the_df, next_df])
        index+=1
    the_df.to_csv("results/trading/hypers/results.csv")
def plot_trading_activities_agents_(without_features=False):
    if without_features is False:
        file_name = ["results/trading/BTC_DQN_features.csv"
        ,"results/trading/BTC_MULTI_AGENT_features.csv"
        ,"results/trading/ETH_DQN_features.csv"
        ,"results/trading/ETH_MULTI_AGENT_features.csv",
        "results/trading/LTC_DQN_features.csv",
        "results/trading/LTC_MULTI_AGENT_features.csv"]
    else:
         file_name = ["results/trading/BTC_DQN_without_features.csv"
        ,"results/trading/BTC_MULTI_AGENT_without_features.csv"
        ,"results/trading/ETH_DQN_without_features.csv"
        ,"results/trading/ETH_MULTI_AGENT_without_features.csv",
        "results/trading/LTC_DQN_without_features.csv",
        "results/trading/LTC_MULTI_AGENT_without_features.csv"]
    names = ["BTC_DQN", "BTC_MULTI_AGENT", "ETH_DQN", "ETH_MULTI_AGENT","LTC_DQN", "LTC_MULTI_AGENT"]

    index = 0
    for filename in file_name:
        
        for i in range(2):
            df = pd.read_csv(filename, index_col="date", parse_dates=True)
            fig, ax = plt.subplots(figsize=(20, 8))

            half_year_locator = mdates.MonthLocator(interval=10)
            ax.xaxis.set_major_locator(half_year_locator)
            if i == 0:
                lable = "close"
                legend = "Price"
            else:
                lable = "Balance"
                legend = "Balance"
            ax.plot(df[lable].index, df[lable])
            ax.plot(df.loc[df['Action'] == 1.0].index, 
                        df[lable][df['Action'] == 1.0],
                        '^', markersize=5, color='green')   
            ax.plot(df.loc[df['Action'] == -1.0].index, 
                        df[lable][df['Action'] == -1.0],
                        'v', markersize=5, color='red')
            plt.legend([legend, "Buy",  "Sell"])
            plt.xlabel('Date time')
            plt.ylabel(legend)
            if without_features is False:
                plt.savefig(''.join(['figures/trading/', str(names[index]),'_',legend, '.png']))
            else:
                plt.savefig(''.join(['figures/trading/without_features/', str(names[index]),'_',legend, '_without.png']))
        index += 1

if __name__ == "__main__":
    #process_mcts_parameters_result_file()
    # plot_mcts_hyper_parameters()
    # calculate_dr()
    #plot_dr()
    #process_hyper_parameters_agents_result_file()
    # plot_trading_activities_agents_()
    #plot_trading_activities_agents_(without_features=True)

    print((1162096.33)--50645.0)
