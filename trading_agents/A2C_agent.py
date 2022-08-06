from stable_baselines3 import A2C
import time
from constants import settings

####################################################################
######################### A2C model ################################
####################################################################

# A2C model
def train_A2C(env_train, timesteps=10000, learning_rate=7e-4, dicount_factor=0.9):
    start = time.time()
    model = A2C('MlpPolicy', env_train, learning_rate=learning_rate, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model
