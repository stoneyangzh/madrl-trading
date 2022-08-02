# coding:utf-8

from __future__ import division
import numpy as np
import time
import math
import random
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist
from copy import deepcopy 
from feature_selection.tree_node import TreeNode
from constants.market_symbol import *


####################################################################
######################### Class: MCTS ##############################
####################################################################

class MCTS():
    def __init__(self, timeLimit=None, iterationLimit=None, Cp = 0.9, explorationConstant=0.1,
                 X=None, marketSymbol=MarketSymbol.BTC):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.Cp = Cp
        self.marketSymbol = marketSymbol
        self.explorationConstant = explorationConstant
        self.X = X
        self.level = {}
        self.best_reward = 0.00000000000001
        self.best_sub_features = None

    def search(self, initialState):
        self.root = TreeNode(initialState, None)
        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()
            self.bestChild = self.getBestChild(self.root, 0)
        return self.getAction(self.bestChild)

    def executeRound(self):
        node = self.treePolicy(self.root)
        m = node.state.feature_subset
        
        if len(m) not in self.level.keys():
            self.level[len(m)] = [node]
        else: self.level[len(m)].append(node)

        rolloutReward = self.defaultPolicy(node.state)
        mReward = rolloutReward[:len(m)]

        self.backpropogate(node, mReward)

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        if explorationValue:
            for child in node.children.values():
                nodeValue = child.mReward + explorationValue * math.sqrt(
                    2 * math.log(node.numVisits) / child.numVisits)
                if nodeValue > bestValue:
                    bestValue = nodeValue
                    bestNodes = [child]
                elif nodeValue == bestValue:
                    bestNodes.append(child)
            return random.choice(bestNodes)
        else:
            for k,v in self.level.items():
                for i in v:
                    nodeValue = i.mReward
                    if nodeValue > bestValue:
                        bestValue = nodeValue
                        bestNodes = [i]
                    elif nodeValue == bestValue:
                        bestNodes.append(i)                    
            return np.random.choice(bestNodes)

    def defaultPolicy(self,state):
        while len(state.feature_subset) < state.search_depth and len(state.feature_subset) < len(state.Feature_COLUMNS):
            try:
                action = random.choice(state.getPossibleActions(state.feature_subset))
            except IndexError:
                raise Exception("Non-terminal state has no possible actions: " + str(state))
            state = state.takeAction(action)
        print("=====features===:", state.feature_subset)
        return state.getReward(self.X[state.feature_subset], self.marketSymbol)

    def treePolicy(self, node):
        while not node.isTerminal:
            if not node.isFullyExpanded():
                return self.expand(node)
            else:
                node = self.getBestChild(node, self.explorationConstant)
                print('bestChild is',node.state.feature_subset)
        return node

    def expand(self, node):
        filtered_actions = []
        exclude_list = []
        if node.parent in self.root.children:
            exclude_list = list(self.root.children.keys())
            exclude_list.append(node.children.keys())
        else:
            exclude_list=list(node.children.keys())
        actions = node.state.getPossibleActions(exclude_list)
        if node is not self.root:
            for action in actions:
                filtered_actions.append(self.pearson_coef_filter(self.X.loc[:,node.state.feature_subset],self.X.loc[:,action] ))
            actions = np.asarray(actions)
            action = np.random.choice(actions[filtered_actions])
        else: action  = np.random.choice(actions[:node.expand_width])
        newNode = TreeNode(node.state.takeAction(action), node)
        node.children[action] = newNode
        newNode.mReward = 0.0
        newNode.numVisits = 0
        return newNode

    def backpropogate(self, node, reward):
        t = 0
        while node is not self.root:
            t += 1
            node.numVisits += 1
            if node.parent is self.root:
                node.parent.numVisits += 1
            node.mReward += (reward[-t] - node.mReward ) / node.numVisits
            node = node.parent

    def getAction(self, bestChild):
        return bestChild.state.feature_subset
    
    def pearson_coef_filter(a,b):
        Cp = 0.9
        x = []
        y = []
        for i in range(a.shape[1]):
            x.append(np.corrcoef(a.iloc[:,i],b,rowvar = False)[0,1])
        x = np.asarray(x)
        y = np.where((x>Cp)|(x<-Cp))
        if len(y[0]):
            return False
        else:
            return True


