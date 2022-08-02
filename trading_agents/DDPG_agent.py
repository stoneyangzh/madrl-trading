from constants import settings
import time
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np

####################################################################
######################### DDPG model ###############################
####################################################################

#DDPG model
def train_DDPG(env_train, timesteps=1000, learning_rate=1e-3):
    # add the noise objects for DDPG
    # n_actions = env_train.action_space.shape[-1]
    # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    print("ddpg timesteps====",timesteps)
    start = time.time()
    model = DDPG('MlpPolicy', env=env_train, learning_rate=learning_rate)
    try:
        model.learn(total_timesteps=timesteps)
    except NameError:
        print("An exception occurred",NameError)
   
    end = time.time()

    print('Training time (DDPG): ', (end-start)/60,' minutes')
    return model