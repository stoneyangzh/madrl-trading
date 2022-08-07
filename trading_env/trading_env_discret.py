# coding=utf-8

import gym
import math
import numpy as np
from gym import spaces
from constants.agent_type import AgentType

import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.dates as mdates

from matplotlib import pyplot as plt
###############################################################################
############################## Class CryptoTradingDiscretEnv #########################
###############################################################################

class CryptoTradingDiscretEnv(gym.Env):
    """
    A customized trading environment extended from OpenAI gym.
    """


    def __init__(self, df, marketSymbol, balance=100000, selected_features=None, agent_type=AgentType.A2C, 
                stateLength=30, transactionCosts=0):
        self.data = df
        self.balance = balance
        self.agent_type = agent_type
        print("***********init***************")

        if selected_features is None:
            self.selected_features = df.columns
        else:
            self.selected_features = selected_features

         # State which contains 
        if selected_features is None:
            self.shape_size = len(df.columns) * (stateLength)
        else:
            self.shape_size = len(selected_features) * (stateLength)
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.shape_size+1, ))
        
        self.data['Position'] = 0
        #Buy, Sell or Hold
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        #Initial amount
        self.data['Cash'] = float(balance)
        #Total balance
        self.data['Balance'] = self.data['Holdings'] + self.data['Cash']
        #Return
        self.data['Returns'] = 0.


        self.marketSymbol = marketSymbol
        self.stateLength = stateLength
        self.t = stateLength
        self.numberOfShares = 0
        #To simulate exchange's transaction fee
        self.transactionCosts = transactionCosts
        self.epsilon = 0.1

        # Initialized state for RL agents
        self.nextObservation(0, self.stateLength)
        self.reward = 0.
        self.done = 0

        self.clipingValue = 1

        self.action_space = spaces.Discrete(2)
       
    def nextObservation(self, starting, ending, position=0):
        self.state = []
        for column in self.selected_features:
            self.state += self.data[column][starting: ending].tolist()
        self.state += [position]

    def reset(self):
        # Reset the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = self.data['Cash'][0]
        self.data['Balance'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Reset the state
        self.nextObservation(0, self.stateLength)
        self.reward = 0.
        self.done = 0

        # Reset additional variables related to the trading activity
        self.t = self.stateLength
        self.numberOfShares = 0

        return self.state

    
    def computeLowerBound(self, cash, numberOfShares, price):
        # Computation of the RL action lower bound
        deltaValues = - cash - numberOfShares * price * (1 + self.epsilon) * (1 + self.transactionCosts)
        if deltaValues < 0:
            lowerBound = deltaValues / (price * (2 * self.transactionCosts + (self.epsilon * (1 + self.transactionCosts))))
        else:
            lowerBound = deltaValues / (price * self.epsilon * (1 + self.transactionCosts))
        return lowerBound
    

    def step(self, action):
        t = self.t
        numberOfShares = self.numberOfShares
        customReward = False
        hold = False

        # Buy
        if(action == 1):
            self.data['Position'][t] = 1
            # Buy to Buy
            if(self.data['Position'][t - 1] == 1):
                self.data['Cash'][t] = self.data['Cash'][t - 1]
                self.data['Holdings'][t] = self.numberOfShares * self.data['close'][t]
            #Hold to Buy
            elif(self.data['Position'][t - 1] == 0):
                self.numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['close'][t] * (1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * self.data['close'][t]
                self.data['Action'][t] = 1
            # Sell to Buy Position is "-1"
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['close'][t] * (1 + self.transactionCosts)
                self.numberOfShares = math.floor(self.data['Cash'][t]/(self.data['close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t] - self.numberOfShares * self.data['close'][t] * (1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * self.data['close'][t]
                self.data['Action'][t] = 1
        # Sell
        elif(action == 0):
            self.data['Position'][t] = -1
            # Sell to Sell
            if(self.data['Position'][t - 1] == -1):
                lowerBound = self.computeLowerBound(self.data['Cash'][t - 1], -numberOfShares, self.data['close'][t-1])
                # Hold if lowerBound less or equal to zero
                if lowerBound <= 0:
                    self.data['Cash'][t] = self.data['Cash'][t - 1]
                    self.data['Holdings'][t] =  - self.numberOfShares * self.data['close'][t]
                else:
                    #Buy more
                    numberOfSharesToBuy = min(math.floor(lowerBound), self.numberOfShares)
                    self.numberOfShares -= numberOfSharesToBuy
                    self.data['Cash'][t] = self.data['Cash'][t - 1] - numberOfSharesToBuy * self.data['close'][t] * (1 + self.transactionCosts)
                    self.data['Holdings'][t] =  - self.numberOfShares * self.data['close'][t]
                    customReward = True
            # Hold to sell
            elif(self.data['Position'][t - 1] == 0):
                self.numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * self.data['close'][t] * (1 - self.transactionCosts)
                self.data['Holdings'][t] = - self.numberOfShares * self.data['close'][t]
                self.data['Action'][t] = -1
            # Buy to Sell
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * self.data['close'][t] * (1 - self.transactionCosts)
                self.numberOfShares = math.floor(self.data['Cash'][t]/(self.data['close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t] + self.numberOfShares * self.data['close'][t] * (1 - self.transactionCosts)
                self.data['Holdings'][t] = - self.numberOfShares * self.data['close'][t]
                self.data['Action'][t] = -1

        # Update the total amount of Balance owned by the agent, as well as the return generated
        self.data['Balance'][t] = self.data['Holdings'][t] + self.data['Cash'][t]
        self.data['Returns'][t] = (self.data['Balance'][t] - self.data['Balance'][t-1])/self.data['Balance'][t-1]

        # Set the RL reward returned to the trading agent
        if not customReward:
            self.reward = self.data['Returns'][t]
        else:
            self.reward = (self.data['close'][t-1] - self.data['close'][t])/self.data['close'][t-1]
        # Transition to the next trading time step
        self.t = self.t + 1

        self.nextObservation(self.t - self.stateLength, self.t, self.data['Position'][self.t - 1])
        if(self.t == self.data.shape[0]):
            self.done = 1 
        
        self.reward = self.clip(self.reward)

        self.info = {'Step':self.t, 'Reward': self.reward, 'Done' : self.done, 'Shares': self.numberOfShares, 'Balance':self.data['Balance'][t]}
 
        return self.state, self.reward, self.done, self.info

    def render(self):
        self.data.drop(self.data[self.data['Balance'] == self.balance].index, inplace = True)
        # self.data['Action'] == 1.0
        path = "Results/",self.agent_type.name, "/resultfile.csv"
        self.data.to_csv(path)
        self.plot()
        self.plot(lable="Balance", legend="Balance")
       
    def plot(self, lable = "close", legend="Price"):
        fig, ax = plt.subplots(figsize=(20, 8))

        half_year_locator = mdates.MonthLocator(interval=10)
        ax.xaxis.set_major_locator(half_year_locator)

        ax.plot(self.data[lable].index, self.data[lable])
        ax.plot(self.data.loc[self.data['Action'] == 1.0].index, 
                    self.data[lable][self.data['Action'] == 1.0],
                    '^', markersize=5, color='green')   
        ax.plot(self.data.loc[self.data['Action'] == -1.0].index, 
                    self.data[lable][self.data['Action'] == -1.0],
                    'v', markersize=5, color='red')
        plt.legend([legend, "Buy",  "Sell"])
        plt.xlabel('Timestamp')
        plt.ylabel(legend)

        plt.savefig(''.join(['results/', str(self.marketSymbol),'_',legend, '_Rendering', '.png']))
    
    def clip(self, value:float): 
        return np.clip(value, -self.clipingValue, self.clipingValue)

    def get_data(self,):
        self.data.drop(self.data[self.data['Balance'] == self.balance].index, inplace = True)
        return self.data

