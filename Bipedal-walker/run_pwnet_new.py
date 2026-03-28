



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

action_array = np.load('/export/kbodla/bipedal_walker/a_train_new.npy')

import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Numerical stability fix
    return e_x / np.sum(e_x, axis=0)

def act_to_lab(actions):
    new_actions = []
    
    # Apply transformation to every element in the action array
    for action in actions:
        new_action = []
        for val in action:
            if val < 0:
                new_action.extend([abs(val), 0])  # If negative, split into [abs(value), 0]
            else:
                new_action.extend([0, val])  # If positive, split into [0, value]
        
        new_actions.append(new_action)

    labels = []
    for new_action in new_actions:
        softmax_probs = softmax(np.array(new_action))  # Convert to NumPy array
        max_index = np.argmax(softmax_probs)  # Get index of max softmax probability
        labels.append(max_index)

    return labels

train_labels = act_to_lab(action_array)

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


SANITY_CHECK = False

NUM_ITERATIONS = 5
NUM_EPOCHS = 100
NUM_CLASSES = 4
NUM_PROTOTYPES = 8

LATENT_SIZE = 240
BATCH_SIZE = 128

DEVICE = 'cpu'
PROTOTYPE_SIZE = 50
MAX_SAMPLES = 100000
delay_ms = 0
SIMULATION_EPOCHS = 10 #30


env_name = "BipedalWalker-v3"
random_seed = 0
n_episodes = 30
lr = 0.002
max_timesteps = 2000
render = True
save_gif = False
#filename = "TD3_{}_{}".format(env_name, random_seed)
#filename += '_solved'
filename = "TD3_BipedalWalker-v2_0_solved"
#directory = "./preTrained/{}".format(env_name)
directory = "TD3-PyTorch-BipedalWalker-v2/preTrained/BipedalWalker-v2/ONE"
env = gym.make(env_name, hardcore=False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
action_scale = 6
real_action = np.linspace(-1.,1., action_scale)
from utils import QNetwork

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


class ListModule(object):
    #Should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class PWNet(nn.Module):

    def __init__(self):
        super(PWNet, self).__init__()
        self.ts = ListModule(self, 'ts_')
        for i in range(NUM_PROTOTYPES):
            transformation = nn.Sequential(
                nn.Linear(LATENT_SIZE, PROTOTYPE_SIZE),
                nn.LayerNorm(PROTOTYPE_SIZE),
                nn.ReLU(),
                nn.Linear(PROTOTYPE_SIZE, PROTOTYPE_SIZE),
            )
            self.ts.append(transformation)  
        self.prototypes = None
        self.epsilon = 1e-5
        self.linear = nn.Linear(NUM_PROTOTYPES, NUM_CLASSES, bias=False) 
        self.__make_linear_weights()
        self.tanh = nn.Tanh()
        self.nn_human_x = nn.Parameter( torch.randn(NUM_PROTOTYPES, LATENT_SIZE), requires_grad=False)
        
    def __make_linear_weights(self):
        custom_weight_matrix = torch.tensor([
                                             [ 1., 0., 0., 0.], 
                                             [ -1., 0., 0., 0.], 
                                             [ 0., 1., 0., 0.], 
                                             [ 0., -1., 0., 0.], 
                                             [ 0., 0., 1., 0.],
                                             [ 0., 0., -1., 0.], 
                                             [ 0., 0., 0., 1.], 
                                             [ 0., 0., 0., -1.], 
                                             ])
        self.linear.weight.data.copy_(custom_weight_matrix.T)   
        
    def __proto_layer_l2(self, x, p):
        output = list()
        b_size = x.shape[0]
        p = p.view(1, PROTOTYPE_SIZE).tile(b_size, 1).to(DEVICE) 
        c = x.view(b_size, PROTOTYPE_SIZE).to(DEVICE)      
        l2s = ( (c - p)**2 ).sum(axis=1).to(DEVICE) 
        act = torch.log( (l2s + 1. ) / (l2s + self.epsilon) ).to(DEVICE)  
        return act
    
    def __output_act_func(self, p_acts):        
        return self.tanh(p_acts)
    
    def forward(self, x):
        
        latent_protos = None
        if self.prototypes is None:
            trans_nn_human_x = list()
            for i, t in enumerate(self.ts):
                trans_nn_human_x.append( t( torch.tensor(self.nn_human_x[i], dtype=torch.float32).view(1, -1)) )
            latent_protos = torch.cat(trans_nn_human_x, dim=0)   
        else:
            latent_protos = self.prototypes
            
        p_acts = list()
        for i, t in enumerate(self.ts):
            action_prototype = latent_protos[i]
            p_acts.append( self.__proto_layer_l2( t(x), action_prototype).view(-1, 1) )
        p_acts = torch.cat(p_acts, axis=1)
        
        logits = self.linear(p_acts)                     
        final_outputs = self.__output_act_func(logits)   
        
        return final_outputs





def evaluate_loader(model, loader, mse_loss):
    model.eval()
    total_loss = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            
            loss = mse_loss(outputs, labels)
            total += len(outputs)
            total_loss += loss.item()
    model.train()
    return total_loss / len(loader)



def proto_loss(model, nn_human_x, criterion):
    model.eval()
    target_x = trans_human_concepts(model, nn_human_x)
    loss = criterion(model.prototypes, target_x) 
    model.train()
    return loss



def trans_human_concepts(model, nn_human_x):
    
    model.eval()
    
    with torch.no_grad():
        trans_nn_human_x = list()
        for i, t in enumerate(model.ts):
            trans_nn_human_x.append( t( torch.tensor(nn_human_x[i], dtype=torch.float32).view(1, -1)) )
        
    model.train()

    return torch.cat(trans_nn_human_x, dim=0)

MODEL_DIR = 'weights/pwnet'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    
#### Start Collecting Data To Form Final Mean and Standard Error Results
data_rewards = list()
data_errors = list()

for iter in tqdm(range(NUM_ITERATIONS)):
    
    # with open('results/pwnet_results.txt', 'a') as f:
    #     f.write(f"ITERATION {iter}: \n")

    MODEL_DIR_ITER = f'weights/pwnet/iter_{iter}.pth'
    
    # writer = SummaryWriter(f"runs/pwnet/Iteration_{iter}")
    
    X_train = np.load('/export/kbodla/bipedal_walker/X_train_new.npy')
    a_train = np.load('/export/kbodla/bipedal_walker/a_train_new.npy')
    tensor_x = torch.Tensor(X_train)
    tensor_y = torch.tensor(a_train, dtype=torch.float32)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    # Get prototypes
    human_concepts = {'Hip1_Forward':  [1., 0., 0., 0.], 'Hip1_Back' :     [-1., 0., 0., 0.],
                      'Knee1_Forward': [0., 1., 0., 0.], 'Knee1_Back' :    [0., -1., 0., 0.],
                      'Hip2_Forward':  [0., 0., 1., 0.], 'Hip2_Back' :     [0., 0., -1., 0.],
                      'Knee2_Forward': [0., 0., 0., 1.], 'Knee2_Back' :    [0., 0., 0., -1.],
                     }
    human_concepts_list = np.array([l for l in human_concepts.values()])
    p_idxs = list()
    nn_human_x = list()
    X_train_new = X_train.squeeze(1)
    # print(X_train_new.shape)
    for i in range(8):
        # idxs = a_train == i
        idxs = np.array([label == i for label in train_labels])
        # print(idxs.shape)
        temp_x = X_train_new[idxs]
        mean = temp_x.mean(axis=0)
        knn = KNeighborsClassifier().fit(temp_x, list(range(len(temp_x))))
        idx = knn.kneighbors(X=mean.reshape(1,-1), n_neighbors=1, return_distance=False)
        p_idxs.append(idx.item())
        nn_human_x.append( temp_x[idx.item()].tolist() )
    nn_human_x = np.array(nn_human_x)
    n_neighbours = 1
    knn = KNeighborsClassifier(algorithm='brute')
    knn.fit(a_train, list(range(len(a_train))))
    p_idxs = knn.kneighbors(X=human_concepts_list, n_neighbors=n_neighbours, return_distance=False)
    # nn_human_images = observations[p_idxs.flatten()]

    if SANITY_CHECK:
        p_idxs = np.random.randint(0, len(X_train), NUM_PROTOTYPES)

    # nn_human_x = X_train_new[p_idxs.flatten()]
    # nn_human_x = np.load("pic_lin_closest.npy")
    # nn_human_x.squeeze(1)
    # nn_human_actions = a_train[p_idxs.flatten()]

    print(X_train.shape)
    print(a_train.shape)

    #### Training
    model = PWNet().eval()
    model.nn_human_x.data.copy_( torch.tensor(nn_human_x) )

    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    best_error = float('inf')
    model.train()

    loss_data = list()

    # Freeze Linear Layer to make more interpretable
    model.linear.weight.requires_grad = False

    running_loss = 0
    for epoch in range(NUM_EPOCHS):
            
        model.eval()
        train_error = evaluate_loader(model, train_loader, mse_loss)
        model.train()
        
        if train_error < best_error:
            torch.save(  model.state_dict(), MODEL_DIR_ITER  )
            best_error = train_error
        
        for instances, labels in train_loader:
            
            optimizer.zero_grad()
                    
            instances, labels = instances.to(DEVICE), labels.to(DEVICE)
                            
            logits = model(instances)    
            loss = mse_loss(logits, labels)
            loss_data.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
                    
        print("Epoch:", epoch, "Running Loss:", running_loss / len(train_loader), "Train error:", train_error)
        # with open('results/pwnet_results.txt', 'a') as f:
        #     f.write(f"Epoch: {epoch}, Running Loss: {running_loss / len(train_loader)}, Train error: {train_error}\n")
            
        # writer.add_scalar("Running_loss", running_loss/len(train_loader), epoch)
        # writer.add_scalar("Train_error", train_error, epoch)
        running_loss = 0
        
        scheduler.step()

    states, actions, rewards, log_probs, values, dones, X_train = [], [], [], [], [], [], []



    # Wapper model with learned weights
    model = PWNet().eval()
    model.load_state_dict(torch.load(MODEL_DIR_ITER))

    # Projection
    print("Checking for the error...", evaluate_loader(model, train_loader, mse_loss))
    


    total_reward = list()
    all_errors = list()
    model.eval()
    for ep in tqdm(range(SIMULATION_EPOCHS)):
        ep_reward = 0
        ep_errors = 0
        state = env.reset()

        for t in range(max_timesteps):
            action_prob,x = net(torch.tensor(state))
            action =  [int(x.max(1)[1]) for x in action_prob]
            new_a = np.array([real_action[x] for x in action])
            A = model( torch.tensor(x, dtype=torch.float32).view(1, -1) )
            state, reward, done, _ = env.step(A.detach().numpy()[0])
            # state, reward, done, _ = env.step(bb_action)

            ep_reward += reward
            ep_errors += mse_loss( torch.tensor(new_a), A[0]).detach().item()

            if done:
                break
                
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        total_reward.append( ep_reward )
        all_errors.append( ep_errors )
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
print("===== Data MAE:")
print("MSE:", data_errors)
print("Mean:", data_errors.mean())
print("Standard Error:", data_errors.std() / np.sqrt(NUM_ITERATIONS)  )
print(" ")
print("===== Data Reward:")
print("Rewards:", data_rewards)
print("Mean:", data_rewards.mean())
print("Standard Error:", data_rewards.std() / np.sqrt(NUM_ITERATIONS)  )


