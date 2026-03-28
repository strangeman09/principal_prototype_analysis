



import sys
if sys.version_info < (3, 7):
    import contextlib
    class nullcontext:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    contextlib.nullcontext = nullcontext


import gym
import numpy as np
import torch
import torch.nn as nn
import numpy as np      
#import pandas as pd
import pickle



from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser
from os.path import join

from TD3 import TD3
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN, OPTICS
import os

from random import sample
from tqdm import tqdm
from time import sleep
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error

from random import sample
from tqdm import tqdm
from time import sleep
from collections import Counter

from utils import QNetwork
SANITY_CHECK = False

NUM_ITERATIONS = 5
NUM_EPOCHS = 20
NUM_CLASSES = 4
NUM_PROTOTYPES = 8

LATENT_SIZE = 300
BATCH_SIZE = 128

DEVICE = 'cpu'
PROTOTYPE_SIZE = 50
MAX_SAMPLES = 100000
delay_ms = 0
SIMULATION_EPOCHS = 10 #30
max_timesteps = 2000
random_seed = 0
n_episodes = 30
action_scale = 6
real_action = np.linspace(-1.,1., action_scale)
data_rewards = list()
data_errors = list()

def extract_qnetwork_from_bqn(bqn_state_dict):
    qnetwork_state_dict = {}
    
    # Filter keys that start with 'q.' and remove the prefix
    for key, value in bqn_state_dict.items():
        if key.startswith('q.'):
            qnetwork_state_dict[key[2:]] = value
    
    return qnetwork_state_dict
env = gym.make("BipedalWalker-v3")
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
net = QNetwork(state_space,action_space,action_scale = 6)
bqn_state_dict = torch.load('agent_1000')
qnetwork_state_dict = extract_qnetwork_from_bqn(bqn_state_dict)
net.load_state_dict(qnetwork_state_dict)

for iter in tqdm(range(NUM_ITERATIONS)):
    
   


    total_reward = list()
    all_errors = list()
 
    for ep in tqdm(range(SIMULATION_EPOCHS)):
        ep_reward = 0
        ep_errors = 0
        state = env.reset()

        for t in range(max_timesteps):
            action_prob,x = net(torch.tensor(state))
            action =  [int(x.max(1)[1]) for x in action_prob]
            new_a = np.array([real_action[x] for x in action])
          
            # A = model( torch.tensor(x, dtype=torch.float32).view(1, -1) )
            state, reward, done, _ = env.step(new_a)
            # state, reward, done, _ = env.step(bb_action)

            ep_reward += reward
            # ep_errors += mse_loss( torch.tensor(bb_action), A[0]).detach().item()

            if done:
                break
                
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        total_reward.append( ep_reward )
        # all_errors.append( ep_errors )
        ep_reward = 0

    env.close()  

    data_rewards.append(  sum(total_reward) / SIMULATION_EPOCHS  )
    data_errors.append(  sum(all_errors) / SIMULATION_EPOCHS )
    print("Reward: ", sum(total_reward) / SIMULATION_EPOCHS)
    print("MSE: ", sum(all_errors) / SIMULATION_EPOCHS )

    # # log the reward and MAE
    # writer.add_scalar("Reward", sum(total_reward) / SIMULATION_EPOCHS, iter)
    # writer.add_scalar("MSE", sum(all_errors) / SIMULATION_EPOCHS, iter)

    # with open('results/pwnet_results.txt', 'a') as f:
    #     f.write(f"Reward: {sum(total_reward) / SIMULATION_EPOCHS}, MSE: {sum(all_errors) / SIMULATION_EPOCHS}\n")     

data_errors = np.array(data_errors)
data_rewards = np.array(data_rewards)


print(" ")
# print("===== Data MAE:")
# print("MSE:", data_errors)
# print("Mean:", data_errors.mean())
# print("Standard Error:", data_errors.std() / np.sqrt(NUM_ITERATIONS)  )
print(" ")
print("===== Data Reward:")
print("Rewards:", data_rewards)
print("Mean:", data_rewards.mean())
print("Standard Error:", data_rewards.std() / np.sqrt(NUM_ITERATIONS)  )


