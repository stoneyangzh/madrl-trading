
####################################################################
######################### Class: TreeNode ##########################
####################################################################

class TreeNode():
    def __init__(self, state, parent):
        self.children = {}
        self.state = state
        self.isTerminal = state.isTerminal()
        self.parent = parent
        self.numVisits = 0
        self.rolloutReward = 0
        self.mReward = 0
        self.expand_width = 10

    def isFullyExpanded(self):
        if len(self.children) == self.expand_width:
            return True
        else:
            return False