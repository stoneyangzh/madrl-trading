import pandas as pd
import os
import time
from constants.market_symbol import MarketSymbol
from trading_agents.A2C_agent import train_A2C
from trading_agents.DDPG_agent import train_DDPG
from trading_agents.DQN_agent import train_DQN
from constants.agent_type import AgentType

from trading_env.trading_env import CryptoTradingEnv
from evaluation.trading_evaluation import Evaluator
from trading_env.trading_env_discret import CryptoTradingDiscretEnv
from constants.settings import *
from util.file_util import *
from random import choice

def run_multi_agent1(marketSymbol:MarketSymbol.BTC, train_dataset, selected_features, timesteps, dicount_factor=0.9, learning_rate=7e-4):
    # Train A2C agent
    a2c_agent, a2c_sharpe_ratio,data = trainig(marketSymbol, train_dataset, selected_features, agent_type=AgentType.A2C, timesteps=timesteps,learning_rate=learning_rate, dicount_factor=dicount_factor)
    return a2c_agent, a2c_sharpe_ratio,data
def run_multi_agent(marketSymbol:MarketSymbol.BTC, train_dataset, selected_features, timesteps, dicount_factor=0.9, learning_rate=7e-4):
    # Train A2C agent
    a2c_agent, a2c_sharpe_ratio,data = trainig(marketSymbol, train_dataset, selected_features, agent_type=AgentType.A2C, timesteps=timesteps,dicount_factor=dicount_factor)
    print("======sharpe for a2c:", a2c_sharpe_ratio)
    # Train DDPG agent
    ddpg_agent, ddpg_sharpe_ratio,data = trainig(marketSymbol, train_dataset, selected_features, agent_type=AgentType.DDPG, timesteps=timesteps,dicount_factor=dicount_factor)
    print("======sharpe for DDPG:", ddpg_sharpe_ratio)
    if a2c_sharpe_ratio > ddpg_sharpe_ratio:
        print("Best agent is A2C")
        a2c_agent.save(f"{TRAINED_AGENT_DIR}/{AgentType.A2C.name}")
        return a2c_agent,a2c_sharpe_ratio,data
    else:
        print("Best agent is DDPG")
        ddpg_agent.save(f"{TRAINED_AGENT_DIR}/{AgentType.DDPG.name}")
        return ddpg_agent, ddpg_sharpe_ratio,data

def trainig(marketSymbol:MarketSymbol.BTC, train_dataset, selected_features, agent_type=AgentType.A2C, timesteps=5000, learning_rate=7e-4, dicount_factor=0.9):
    # Create trading A2C agent
    if agent_type is AgentType.A2C:
        env_train = CryptoTradingDiscretEnv(train_dataset, marketSymbol.name, selected_features=selected_features, agent_type=agent_type)
        agent = train_A2C(env_train, timesteps=timesteps, learning_rate=learning_rate,dicount_factor=dicount_factor)
    elif agent_type is AgentType.DDPG:
        env_train = CryptoTradingEnv(train_dataset, marketSymbol.name, selected_features=selected_features, agent_type=agent_type)
        agent = train_DDPG(env_train, timesteps=timesteps, learning_rate=learning_rate,dicount_factor=dicount_factor)
    else:
        env_train = CryptoTradingDiscretEnv(train_dataset, marketSymbol.name, selected_features=selected_features, agent_type=agent_type)
        agent = train_DQN(env_train, timesteps=timesteps, learning_rate=learning_rate,dicount_factor=dicount_factor)
    #env_train.render()

    return agent,getSharpeRatio(env_train.data),env_train.data,

def testing(marketSymbol:MarketSymbol.BTC, validation_dataset, selected_features, agent, agent_type=AgentType.A2C):
    env_testing= CryptoTradingEnv(validation_dataset, marketSymbol.name, selected_features=selected_features,agent_type=agent_type)
    obs_testing = env_testing.reset()
    # Validation
    while not done:
        action, _states = agent.predict(obs_testing)
        obs, rewards, done, info = env_testing.step(action)

    sharpe_ratio = getSharpeRatio()

    return agent,sharpe_ratio,

def getSharpeRatio(data):
    evaluator = Evaluator(data)
    return evaluator.calculateSharpeRatio()

def getPerformance_table(data):
    evaluator = Evaluator(data)
    return evaluator.calculatePerformance()


def data_split(df,start,end):
    data = df[start:end]
    return data

def hyper_paramters_tuning():
    timesteps = 5000
    epoch = 100000
    balance = 100000
   
    # Create feature selection agent
    
    datasets_files = ["data/processed/Binance_BTCUSDT_minute_processed.csv", "data/processed/Binance_ETHUSDT_minute_processed.csv", "data/processed/Binance_LTCUSDT_minute_processed.csv"]
    marketSymbols = [MarketSymbol.BTC, MarketSymbol.ETH, MarketSymbol.LTC]
    markers_index = 0
    for filename in datasets_files:
        
        agents = [AgentType.DQN, AgentType.MULTI_AGENT]
        learning_rates_dqn = [1e-1, 1e-2,1e-3,1e-4,1e-5]
        learning_rates_multi = [7e-1,7e-3,7e-3,7e-4,7e-5]
        dicount_factors = [0.8, 0.82, 0.84, 0.86, 0.9]
        total_time_steps = [1000, 2000,3000, 4000, 5000]

        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        df = pd.read_csv(file_path,header = 0,index_col="date", parse_dates=True)

        agents_index = 0
        for agent_type in agents:
            result_dict = {}
            for i in range(20):
                dicount_factor = choice(dicount_factors)
                total_time_step = choice(total_time_steps)
                best_score_best_features = {}

                start = time.time()
                if agent_type is AgentType.DQN:
                    learning_rate = choice(learning_rates_dqn)
                    agent, sharpRatio,data = trainig(marketSymbols[markers_index], df, df.columns, AgentType.DQN,timesteps=total_time_step, learning_rate=learning_rate, dicount_factor=dicount_factor)
                else:
                    learning_rate = choice(learning_rates_multi)
                    agent, sharpRatio,data = run_multi_agent1(marketSymbols[markers_index], df, df.columns, timesteps=total_time_step, dicount_factor=dicount_factor,learning_rate=learning_rate)
                
                elapsed = time.time() - start
                print('time=',elapsed)

                best_score_best_features["learning_rate"] = learning_rate
                best_score_best_features["dicount_factor"] = dicount_factor
                best_score_best_features["total_time_step"] = total_time_step
                best_score_best_features["performance_table"] = getPerformance_table(data)
                best_score_best_features["time"] = elapsed

                result_dict["epoch-{}".format(i+1)] = best_score_best_features

            agents_index += 1
            save_to_file("results/trading/hypers/{}{}{}{}".format(marketSymbols[markers_index].name,"_", agent_type.name,"_trading_hypers.txt"), result_dict)
        markers_index += 1


if __name__ == "__main__":
    hyper_paramters_tuning()
    
