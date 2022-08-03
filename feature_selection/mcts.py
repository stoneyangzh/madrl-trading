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
    def __init__(self, iterationLimit=None,
                 X=None, marketSymbol=MarketSymbol.BTC):
        if iterationLimit == None:
            raise ValueError("Must have either a time limit or an iteration limit")
        # number of iterations of the search
        if iterationLimit < 1:
            raise ValueError("Iteration limit must be greater than one")
        self.searchLimit = iterationLimit
        #exploration
        self.explorationConstant=0.1
        self.limitType = 'iterations'
        self.marketSymbol = marketSymbol
        self.X = X
        self.level = {}
        self.best_reward = 0.00000000000001
        self.best_sub_features = None
        self.childs = {}

    def search(self, initialState):
        self.root = TreeNode(initialState, None)
        for i in range(self.searchLimit):
            self.executeRound()
        self.bestChild = self.getBestFeatureSubset()
        # return self.getAction(self.bestChild)
        return self.bestChild

    def executeRound(self):
        node = self.treePolicy(self.root)
        m = node.state.feature_subset
        
        if len(m) not in self.level.keys():
            self.level[len(m)] = [node]
        else: self.level[len(m)].append(node)

        rolloutReward = self.defaultPolicy(node.state)
        print("====rolloutReward:",rolloutReward)
        mReward = rolloutReward

        self.backpropogate(node, mReward)

    def getBestFeatureSubset(self):
        bestScore = -10000
        for reward in self.childs:
            if(reward > bestScore):
                bestScore = reward
        return self.childs[bestScore]

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
        if "close" not in state.feature_subset:
            state.feature_subset.append("close")

        print("=====features===:", state.feature_subset)
        reward = state.getReward(self.X[state.feature_subset], self.marketSymbol)
        self.childs[reward]=state.feature_subset
        return reward

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
            node.mReward += node.mReward / node.numVisits
            node = node.parent
    
    def pearson_coef_filter(self, a,b):
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


