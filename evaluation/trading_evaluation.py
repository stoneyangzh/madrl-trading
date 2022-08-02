# coding=utf-8
import numpy as np

from tabulate import tabulate
from matplotlib import pyplot as plt
import pandas as pd

####################################################################
######################### Class Evaluator ##########################
####################################################################

class Evaluator:
    """
    GOAL: To evaluate agent's performance.
    """
    def __init__(self, performanceData):
        self.data = performanceData

    def calculatePnL(self):
        # calculate the PnL
        self.PnL = self.data["Balance"][-1] - self.data["Balance"][0]
        return self.PnL
    
    def calculateAnnualizedReturn(self):
        # calculate the cumulative return over the entire trading horizon
        cumulativeReturn = self.data['Returns'].cumsum()
        cumulativeReturn = cumulativeReturn[-1]
        
        # calculate the time elapsed (in days)
        start = self.data.index[0].to_pydatetime()
        end = self.data.index[-1].to_pydatetime()     
        timeElapsed = end - start
        timeElapsed = timeElapsed.days

        # calculate the Annualized Return
        if(cumulativeReturn > -1):
            self.annualizedReturn = 100 * (((1 + cumulativeReturn) ** (365/timeElapsed)) - 1)
        else:
            self.annualizedReturn = -100
        return self.annualizedReturn
    
    
    def calculateAnnualizedVolatility(self):
        # calculate the Annualized Volatility (252 trading days in 1 trading year)
        self.annualizedVolatily = 100 * np.sqrt(252) * self.data['Returns'].std()
        return self.annualizedVolatily
    
    
    def calculateSharpeRatio(self, riskFreeRate=0):
        # calculate the expected return
        expectedReturn = self.data['Returns'].mean()
        
        # calculate the returns volatility
        volatility = self.data['Returns'].std()
        
        # calculate the Sharpe Ratio (365 trading days in 1 year)
        if expectedReturn != 0 and volatility != 0:
            self.sharpeRatio = np.sqrt(365) * (expectedReturn - riskFreeRate)/volatility
        else:
            self.sharpeRatio = 0
        print("=====mean====",expectedReturn)
        print("=====std====",volatility)
        return self.sharpeRatio

    def calculateProfitability(self):
        # Initialization of some variables
        good = 0
        bad = 0
        profit = 0
        loss = 0
        index = next((i for i in range(len(self.data.index)) if self.data['Action'][i] != 0), None)
        if index == None:
            self.profitability = 0
            self.averageProfitLossRatio = 0
            return self.profitability, self.averageProfitLossRatio
        Balance = self.data['Balance'][index]

        # Monitor the success of each trade over the entire trading horizon
        for i in range(index+1, len(self.data.index)):
            if(self.data['Action'][i] != 0):
                delta = self.data['Balance'][i] - Balance
                Balance = self.data['Balance'][i]
                if(delta >= 0):
                    good += 1
                    profit += delta
                else:
                    bad += 1
                    loss -= delta

        # Special case of the termination trade
        delta = self.data['Balance'][-1] - Balance
        if(delta >= 0):
            good += 1
            profit += delta
        else:
            bad += 1
            loss -= delta

        # calculate the Profitability
        self.profitability = 100 * good/(good + bad)
         
        # calculate the ratio average Profit/Loss  
        if(good != 0):
            profit /= good
        if(bad != 0):
            loss /= bad
        if(loss != 0):
            self.averageProfitLossRatio = profit/loss
        else:
            self.averageProfitLossRatio = float('Inf')

        return self.profitability, self.averageProfitLossRatio
        
    def calculatePerformance(self):
        # calculate the entire set of performance indicators
        self.calculatePnL()
        self.calculateAnnualizedReturn()
        self.calculateAnnualizedVolatility()
        self.calculateProfitability()
        self.calculateSharpeRatio()

        # Generate the performance table
        self.performanceTable = [["Profit & Loss (P&L)", "{0:.0f}".format(self.PnL)], 
                                 ["Annualized Return", "{0:.2f}".format(self.annualizedReturn) + '%'],
                                 ["Annualized Volatility", "{0:.2f}".format(self.annualizedVolatily) + '%'],
                                 ["Sharpe Ratio", "{0:.3f}".format(self.sharpeRatio)],
                                 ["Profitability", "{0:.2f}".format(self.profitability) + '%'],
                                 ["Ratio Average Profit/Loss", "{0:.3f}".format(self.averageProfitLossRatio)]]
        return self.performanceTable


    def displayPerformance(self, name):
        
        # Generation of the performance table
        self.calculatePerformance()
        
        # Display the table in the console (Tabulate for the beauty of the print operation)
        headers = ["Performance Indicator", name]
        tabulation = tabulate(self.performanceTable, headers, tablefmt="fancy_grid", stralign="center")
        print(tabulation)

if __name__ == "__main__":
    df = pd.read_csv("/Users/user/Documents/workspace/madrl-trading/results/resultfile.csv")
    # calculate the expected return
    print(df['Returns'].head())
    expectedReturn = df['Returns'].mean()
    
    # calculate the returns volatility
    volatility = df['Returns'].std()

    print(expectedReturn)
    print(volatility)
    print((expectedReturn - 0)/volatility)