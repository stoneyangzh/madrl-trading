#  MSc project --  madrl-trading
## Abstract
In algorithmic trading(including crypto-currency), feature selection and trading decision making are two prominent challenges in getting long-term profit as the training
data is partially observed. And many different deep learning algorithms and reinforcement learning have been applied in algorithmic trading for crypto-currency, such as
support vector machines(SVMs), convolutional neural networks(CNN), long short-term
memory(LSTM) and value-based deep Q-learning networks(DQN), the performance is
unsatisfactory and new advanced reinforcement learning algorithms applying in this field
are insufficient. In this dissertation, a new multi-agent trading approach based on deep
reinforcement learning will be proposed. To make trading decisions and gain profits automatically in the dynamic crypto-currency markets. First, a feature selection agent will
be developed using Monte Carlo tree search(MCTS) to automatically extract and select
robust market representations. Then, two trading agents which extend and combine
the best features of two actor-critic based algorithms, deep deterministic policy gradient(DDPG) and advantage actor-critic(A2C) will be implemented. Furthermore, the
two agents can be cooperating with each other through a centralized controller. Finally,
with the evaluation results, will compare this new multi-agent approach performance
with the baselines(DQN) in terms of financial return and other metrics.
Keywords: Reinforcement learning, algorithmic trading, actor-critic, Monte
Carlo tree search, multi-agent deep reinforcement learning.
## Datasets

Due to the datasets are too big, so please download from google drive. https://drive.google.com/file/d/1XbBmK5YGq5Joj227gOrnjayE8o7TpYvn/view?usp=sharing

## Implementations

<p align="center"><img src="https://github.com/stoneyangzh/madrl-trading/blob/master/figures/trading%20agents_architecture.png"></p>


## Results
<p align="center"><img src="https://github.com/stoneyangzh/madrl-trading/blob/master/figures/results.jpeg"></p>