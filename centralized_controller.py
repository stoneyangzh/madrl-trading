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

def training(marketSymbol:MarketSymbol.BTC, train_dataset, selected_features, agent_type=AgentType.A2C, timesteps=1):
    # Vectorized environments for the agent
    
    # Create trading A2C agent
    if agent_type is AgentType.A2C:
        env_train = CryptoTradingDiscretEnv(train_dataset, marketSymbol.name, selected_features=selected_features, agent_type=agent_type)
        agent = train_A2C(env_train, timesteps=timesteps)
    elif agent_type is AgentType.DDPG:
        env_train = CryptoTradingEnv(train_dataset, marketSymbol.name, selected_features=selected_features, agent_type=agent_type)
        agent = train_DDPG(env_train, timesteps=timesteps)
    else:
        env_train = CryptoTradingDiscretEnv(train_dataset, marketSymbol.name, selected_features=selected_features, agent_type=agent_type)
        agent = train_DQN(env_train, timesteps=timesteps)
    env_train.render()

    return agent,getSharpeRatio(env_train.data),

def testing(marketSymbol:MarketSymbol.BTC, validation_dataset, selected_features, agent, agent_type=AgentType.A2C):
    # Vectorized environments for the agent
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

def data_split(df,start,end):
    data = df[start:end]
    return data

def tuning_learning_rate():
    pass

def tuning_timesteps():
    pass

def run_feature_selector():
    pass

def run_multi_agents(marketSymbol:MarketSymbol.BTC, train_dataset, selected_features, agent_type, timesteps):
    # Train A2C agent
    a2c_agent, a2c_sharpe_ratio = training(marketSymbol, train_dataset, selected_features, agent_type=AgentType.A2C, timesteps=timesteps)
    print("======sharpe for a2c:", a2c_sharpe_ratio)
    # Train DDPG agent
    ddpg_agent, ddpg_sharpe_ratio = training(marketSymbol, train_dataset, selected_features, agent_type=AgentType.DDPG, timesteps=timesteps)
    print("======sharpe for DDPG:", ddpg_sharpe_ratio)
    if a2c_sharpe_ratio > ddpg_sharpe_ratio:
        print("Best agent is A2C")
        a2c_agent.save(f"{TRAINED_AGENT_DIR}/{AgentType.A2C.name}")
    else:
        print("Best agent is DDPG")
        ddpg_agent.save(f"{TRAINED_AGENT_DIR}/{AgentType.DDPG.name}")
if __name__ == "__main__":
    btc_path = "data/processed/Binance_BTCUSDT_minute_processed.csv"
    eth_path = "data/processed/Binance_ETHUSDT_minute_processed.csv"
    ltc_path = "data/processed/Binance_LTCUSDT_minute_processed.csv"

    test_data= "data/processed/Binance_BTCUSDT.csv"

    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_data)

    df = pd.read_csv(data_file, index_col=0)
    
    train_dataset = df
    validation_dataset = df
    timesteps = 1400
    epoch = 100000
    balance = 100000
   
    # Create feature selection agent
    selected_features = df.columns
    
    a2c_agent, a2c_sharpe_ratio = training(MarketSymbol.BTC, train_dataset, selected_features, agent_type=AgentType.DQN, timesteps=timesteps)
    print("======sharpe for DQN:", a2c_sharpe_ratio)

    
