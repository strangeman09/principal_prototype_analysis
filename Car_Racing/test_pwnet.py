import sys
if sys.version_info < (3, 7):
    import contextlib
    class nullcontext:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    contextlib.nullcontext = nullcontext

import gym
import torch 
import torch.nn as nn
import numpy as np      
import pickle
import toml

from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser
from os.path import join
from games.carracing import RacingNet, CarRacing
from ppo import PPO
from torch.distributions import Beta
from tqdm import tqdm


NUM_ITERATIONS = 5
MODEL_DIR = 'weights/pw_net.pth'
CONFIG_FILE = "config.toml"
NUM_CLASSES = 3
LATENT_SIZE = 256
PROTOTYPE_SIZE = 50
BATCH_SIZE = 32
NUM_EPOCHS = 100
DEVICE = 'cpu'
delay_ms = 0
NUM_PROTOTYPES = 4
SIMULATION_EPOCHS = 30
def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = toml.load(f)
    return config
cfg = load_config()
env = CarRacing(frame_skip=0, frame_stack=4,)
net = RacingNet(env.observation_space.shape, env.action_space.shape)
data_rewards = list()
data_errors = list()

for _ in range(NUM_ITERATIONS):
    ppo = PPO(
        env,
        net,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
        gae_lambda=cfg["gae_lambda"],
        clip=cfg["clip"],
        value_coef=cfg["value_coef"],
        entropy_coef=cfg["entropy_coef"],
        epochs_per_step=cfg["epochs_per_step"],
        num_steps=cfg["num_steps"],
        horizon=cfg["horizon"],
        save_dir=cfg["save_dir"],
        save_interval=cfg["save_interval"],
    )
    ppo.load("weights/agent_weights.pt")

    mse_loss = nn.MSELoss()
    reward_arr = []
    all_errors = list()

    for i in tqdm(range(SIMULATION_EPOCHS)):
        state = ppo._to_tensor(env.reset())
        count = 0
        rew = 0
        # model.eval()

        for t in range(10000):
            # Get black box action
            value, alpha, beta, latent_x = ppo.net(state)
            value, alpha, beta = value.squeeze(0), alpha.squeeze(0), beta.squeeze(0)
            policy = Beta(alpha, beta)
            input_action = policy.mean.detach()
            bb_action = ppo.env.preprocess(input_action)
            # latent  = lin_model(latent_x.unsqueeze(0))
            # action = model(latent_x)

            # all_errors.append(  mse_loss( torch.tensor(bb_action), action[0]).detach().item()  )

            state, reward, done, _, _ = ppo.env.step(bb_action, real_action=True)
            state = ppo._to_tensor(state)
            rew += reward
            count += 1
            if done:
                break

        reward_arr.append(rew)

    data_rewards.append(  sum(reward_arr) / SIMULATION_EPOCHS  )
    # data_errors.append(  sum(all_errors) / SIMULATION_EPOCHS )
    print("Iteration Reward:", sum(reward_arr) / SIMULATION_EPOCHS)

# data_errors = np.array(data_errors)
data_rewards = np.array(data_rewards)

# print(" ")
# print("===== Data MAE:")
# print("Mean:", data_errors.mean())
# print("Standard Error:", data_errors.std() / np.sqrt(NUM_ITERATIONS)  )
print(" ")
print("===== Data Reward:")
print("Rewards:", data_rewards)
print("Mean:", data_rewards.mean())
print("Standard Error:", data_rewards.std() / np.sqrt(NUM_ITERATIONS)  )