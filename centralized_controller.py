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
    return agent,getSharpeRatio(env_train.get_data()),env_train.get_data(),

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
    datasets_files = ["data/processed/Binance_BTCUSDT_minute_processed.csv", 
    "data/processed/Binance_ETHUSDT_minute_processed.csv", "data/processed/Binance_LTCUSDT_minute_processed.csv"]
    marketSymbols = [MarketSymbol.BTC, MarketSymbol.ETH, MarketSymbol.LTC]
    markers_index = 0
    for filename in datasets_files:
        agents = [AgentType.DQN, AgentType.MULTI_AGENT]
        learning_rates_dqn = [1e-1, 1e-2,1e-3,1e-4,1e-5]
        learning_rates_multi = [7e-1,7e-3,7e-3,7e-4,7e-5]
        dicount_factors = [0.8, 0.82, 0.84, 0.86, 0.9]
        total_time_steps = [100000, 200000,300000, 400000, 500000]

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
                    agent, sharpRatio,data = run_multi_agent(marketSymbols[markers_index], df, df.columns, timesteps=total_time_step, dicount_factor=dicount_factor,learning_rate=learning_rate)
                
                elapsed = time.time() - start
                print('time=',elapsed/60)

                best_score_best_features["learning_rate"] = learning_rate
                best_score_best_features["dicount_factor"] = dicount_factor
                best_score_best_features["total_time_step"] = total_time_step
                best_score_best_features["performance_table"] = getPerformance_table(data)
                best_score_best_features["time"] = elapsed

                result_dict["epoch-{}".format(i+1)] = best_score_best_features

            agents_index += 1
            save_to_file("results/trading/hypers/{}{}{}{}".format(marketSymbols[markers_index].name,"_", agent_type.name,"_trading_hypers.txt"), result_dict)
        markers_index += 1
def run_trading_agents(feature_seleted="_features"):
    datasets_files = ["data/processed/Binance_BTCUSDT_minute_processed.csv", "data/processed/Binance_ETHUSDT_minute_processed.csv", "data/processed/Binance_LTCUSDT_minute_processed.csv"]
    marketSymbols = [MarketSymbol.BTC, MarketSymbol.ETH, MarketSymbol.LTC]
    selected_feature_subsets= {"BTCDQN":"psar,tsi,fisht,tp,volume,open,stoch,ppo,williams_r,mom,macd,cmo,stc,wma,low,trix,roc,donchian,tr,rsi,ao,close","ETHDQN":"volume,stc,stoch,low,cmo,macd,psar,tr,williams_r,wma,trix,donchian,ao,ema,close,tsi,roc,high,tp,fisht,ppo,rsi","LTCDQN":"fisht,rsi,ema,williams_r,stoch,mom,er,trix,cmo,macd,roc,wma,low,high,tsi,tr,tp,open,ppo,stc,psar,donchian,close","BTCMULTI_AGENT":"williams_r,cmo,ema,trix,high,macd,psar,stc,ppo,close,tr,volume,tp,low,fisht,wma,donchian,mom,ao,er,rsi,stoch","ETHMULTI_AGENT":"volume,macd,fisht,high,stoch,cmo,ppo,tp,trix,rsi,roc,er,williams_r,tsi,close,tr,ao,psar,stc,wma,mom,donchian","LTCMULTI_AGENT":"open,er,stc,fisht,ppo,tsi,close,ao,mom,low,stoch,roc,wma,tp,psar,trix,macd,cmo,williams_r,ema,rsi,donchian"}
    hyper_parameters = {"BTCDQN":[0.1,0.8,100000],"ETHDQN":[0.1,0.86,300000],"LTCDQN":[0.1,0.8,300000],"BTCMULTI_AGENT":[0.7,0.84,500000],"ETHMULTI_AGENT":[0.007,0.82,300000],"LTCMULTI_AGENT":[0.0007,0.84,500000]}
    
    markers_index = 0
    for filename in datasets_files:
        
        agents = [AgentType.MULTI_AGENT, AgentType.DQN]
       
        agents_index = 0
        for agent_type in agents:
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
            df = pd.read_csv(file_path,header = 0,index_col="date", parse_dates=True)

            dataset_agent_name = "{}{}".format(marketSymbols[markers_index].name, agents[agents_index].name)
            feature_subset = selected_feature_subsets[dataset_agent_name].split(",")
            hyper_params = hyper_parameters[dataset_agent_name]
            learning_rate = hyper_params[0]
            dicount_factor = hyper_params[1]
            total_time_step = hyper_params[2]

            best_score_best_features = {}

            start = time.time()
            if agent_type is AgentType.DQN:
                agent, sharpRatio,data = trainig(marketSymbols[markers_index], df, feature_subset, AgentType.DQN,timesteps=total_time_step, learning_rate=learning_rate, dicount_factor=dicount_factor)
            else:
                agent, sharpRatio,data = run_multi_agent(marketSymbols[markers_index], df, feature_subset, timesteps=total_time_step, dicount_factor=dicount_factor,learning_rate=learning_rate)
            
            elapsed = time.time() - start
            print('time=',elapsed/60)

            best_score_best_features["performance_table"] = getPerformance_table(data)
            best_score_best_features["time"] = elapsed
            agents_index += 1
            file_name = "results/trading/{}{}{}{}{}".format(marketSymbols[markers_index].name,"_", agent_type.name,feature_seleted,".txt")
            file_name_csv = "results/trading/{}{}{}{}{}".format(marketSymbols[markers_index].name,"_", agent_type.name,feature_seleted,".csv")

            data.to_csv(file_name_csv)
            save_to_file(file_name, best_score_best_features)
        markers_index += 1

if __name__ == "__main__":
    # hyper_paramters_tuning()
    run_trading_agents()
    run_trading_agents(feature_seleted="_without_features")