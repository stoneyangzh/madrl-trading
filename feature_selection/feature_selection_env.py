
from copy import deepcopy
from constants.agent_type import AgentType
from trading_agents.A2C_agent import train_A2C
from trading_env.trading_env_discret import CryptoTradingDiscretEnv
from constants.market_symbol import MarketSymbol
from evaluation.trading_evaluation import Evaluator
from centralized_controller import *

####################################################################
######################### Class:FeatureSelectionEnv ################
####################################################################

class FeatureSelectionEnv():
    def __init__(self, Feature_COLUMNS, agent_type: AgentType.MULTI_AGENT, total_time_steps=5000, learning_rate=7e-4, search_depth=22):
        self.Feature_COLUMNS = Feature_COLUMNS
        self.feature_subset = []
        self.search_depth = search_depth
        self.agent_type = agent_type
        self.learning_rate = learning_rate
        self.total_time_steps = total_time_steps

    def getPossibleActions(self,children):
        possibleActions = []
        for i in self.Feature_COLUMNS:
            if i not in self.feature_subset and i not in children:
                possibleActions.append(i)
        return possibleActions

    def takeAction(self,action):
        newState = deepcopy(self)
        newState.feature_subset.append(str(action))
        return newState

    def isTerminal(self):
        if len(self.feature_subset) > self.search_depth:
            return True
        else:
            return False

    def getReward(self,X, marketSymbol=MarketSymbol):
        if self.agent_type is AgentType.DQN:
            agent, sharpRatio,data = trainig(marketSymbol, X, X.columns, self.agent_type,self.learning_rate)
            return sharpRatio
        else:
            agent, sharpRatio,data = trainig(marketSymbol, X, X.columns, AgentType.A2C, self.learning_rate)
            return sharpRatio

    def __str__(self):
        return str(self.feature_subset)
    def __repr__(self):
        return str(self.feature_subset)