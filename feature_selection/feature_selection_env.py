
from copy import deepcopy
from trading_agents.A2C_agent import train_A2C
from trading_env.trading_env_discret import CryptoTradingDiscretEnv
from constants.market_symbol import MarketSymbol
from evaluation.trading_evaluation import Evaluator

####################################################################
######################### Class:FeatureSelectionEnv ################
####################################################################

class FeatureSelectionEnv():
    def __init__(self, Feature_COLUMNS, ):
        self.Feature_COLUMNS = Feature_COLUMNS
        self.feature_subset = []
        self.search_depth = 20

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
        env_train = CryptoTradingDiscretEnv(X, marketSymbol.name)
        agent = train_A2C(env_train, timesteps=100)
        evaluator = Evaluator(env_train.data)
        return evaluator.calculateSharpeRatio()

    def __str__(self):
        return str(self.feature_subset)
    def __repr__(self):
        return str(self.feature_subset)