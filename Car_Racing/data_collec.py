import sys
if sys.version_info < (3, 7):
    import contextlib

    class nullcontext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    contextlib.nullcontext = nullcontext


import toml
import numpy as np
import torch
import pickle
import os
import copy

from argparse import ArgumentParser
from os.path import join
from games.carracing import RacingNet, CarRacing
from ppo import PPO
from torch.distributions import Beta
from tqdm import tqdm


CONFIG_FILE = "config.toml"
device = "cpu"
NUM_EPISODES = 80


if not os.path.exists("weights/"):
    os.mkdir("weights/")
if not os.path.exists("data/"):
    os.mkdir("data/")


def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = toml.load(f)
    return config


cfg = load_config()
env = CarRacing(frame_skip=0, frame_stack=4)
net = RacingNet(env.observation_space.shape, env.action_space.shape)

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

states = []
real_actions = []
rewards = []
X_train = []

self_state = ppo._to_tensor(env.reset())
reward_arr = []


for ep in tqdm(range(NUM_EPISODES)):
    next_state = ppo.env.reset()
    rew = 0
    done = False
    count = 0

    while not done:
        count += 1

        value, alpha, beta, x = ppo.net(self_state)
        value = value.squeeze(0)
        alpha = alpha.squeeze(0)
        beta = beta.squeeze(0)

        policy = Beta(alpha, beta)
        input_action = policy.mean.detach()

        next_state, reward, done, info, real_action = ppo.env.step(
            input_action.cpu().numpy()
        )

        next_state = ppo._to_tensor(next_state)

        real_actions.append(real_action.tolist())
        X_train.append(x.detach().cpu().numpy())
        states.append(self_state)

        self_state = next_state
        rew += reward

    reward_arr.append(rew)
    print(count)


print("average reward per episode :", sum(reward_arr) / NUM_EPISODES)


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)


def act_to_lab(action):
    new_action = copy.deepcopy(action)

    for act in new_action:
        if act[0] < 0:
            act.insert(1, 0)
            act[0] = np.abs(act[0])
        else:
            act.insert(1, act[0])
            act[0] = 0.0

    labels = []
    for action in new_action:
        softmax_probs = softmax(action)
        max_index = np.argmax(softmax_probs)
        labels.append(max_index)

    return labels


train_labels = act_to_lab(real_actions)

X_train_tuple = tuple(X_train)
real_actions_tuple = tuple(real_actions)
img_tuple = tuple(states)

x_train_path = "/export/kbodla/car_racing"
a_train_path = "/export/kbodla/car_racing"

if not os.path.exists(x_train_path):
    os.mkdir(x_train_path)
if not os.path.exists(a_train_path):
    os.mkdir(a_train_path)


with open("/export/kbodla/car_racing/X_train.pkl", "wb") as f:
    pickle.dump(X_train_tuple, f)

with open("/export/kbodla/car_racing/actions.pkl", "wb") as f:
    pickle.dump(real_actions_tuple, f)

with open("/export/kbodla/car_racing/images.pkl", "wb") as f:
    pickle.dump(img_tuple, f)
