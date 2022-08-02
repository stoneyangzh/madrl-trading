from constants import settings
import time
from stable_baselines3 import DQN
import numpy as np

####################################################################
######################### DQN model ################################
####################################################################

#DQN model
def train_DQN(env_train, timesteps=10000, learning_rate=1e-4):

    start = time.time()
    model = DQN("MlpPolicy", env_train, verbose=1, learning_rate=learning_rate)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    print('Training time (DQN): ', (end-start)/60,' minutes')
    return model