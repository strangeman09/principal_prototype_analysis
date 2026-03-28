import gym
import torch 
import torch.nn as nn
import numpy as np      
import pickle
import toml
import cv2
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import json
import random
from tqdm import tqdm
from utils import DuelCNN, Agent
from collections import Counter
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser
from os.path import join
from torch.distributions import Beta

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans, DBSCAN, OPTICS

from random import sample
from tqdm import tqdm
from time import sleep

from collections import deque


MODEL_DIR = 'weights/pw_net.pth'
NUM_CLASSES = 6
LATENT_SIZE = 1536
PROTOTYPE_SIZE = 50
BATCH_SIZE = 32
NUM_EPOCHS = 10
DEVICE = 'cuda'
delay_ms = 0
NUM_PROTOTYPES = 6
SIMULATION_EPOCHS = 30
NUM_ITERATIONS = 3


np.bool8 = bool 
ENVIRONMENT = "PongDeterministic-v4"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODELS = False  # Save models to file so you can test later
MODEL_PATH = "./models/pong-cnn-"  # Models path for saving or loading
SAVE_MODEL_INTERVAL = 10  # Save models at every X epoch
TRAIN_MODEL = False  # Train model while playing (Make it False when testing a model)
LOAD_MODEL_FROM_FILE = True  # Load model from file
LOAD_FILE_EPISODE = 900  # Load Xth episode from file
BATCH_SIZE = 64  # Minibatch size that select randomly from mem for train nets
MAX_EPISODE = 100000  # Max episode
MAX_STEP = 100000  # Max step size for one episode
NUM_EPISODES = 3
MAX_MEMORY_LEN = 50000  # Max memory len
MIN_MEMORY_LEN = 40000  # Min memory len before start train
GAMMA = 0.97  # Discount rate
ALPHA = 0.00025  # Learning rate
EPSILON_DECAY = 0.99  # Epsilon decay rate by step

net =  DuelCNN(h=80, w=64, output_size=6)

net.load_state_dict(torch.load('models/pong-cnn-900.pkl'))
environment = gym.make(ENVIRONMENT) # , render_mode='human')  # Get env
environment.seed(0)
agent = Agent(environment)  # Create Agent
if LOAD_MODEL_FROM_FILE:
    agent.online_model.load_state_dict(torch.load(MODEL_PATH+str(LOAD_FILE_EPISODE)+".pkl", map_location=torch.device('cuda')))
    with open(MODEL_PATH+str(LOAD_FILE_EPISODE)+'.json') as outfile:
        param = json.load(outfile)
        agent.epsilon = param.get('epsilon')
    startEpisode = LOAD_FILE_EPISODE + 1
else:
    startEpisode = 1
        

import torch.nn as nn
import torch.nn.functional as F

class DuelCNNWrapper(nn.Module):
    def __init__(self):
        super(DuelCNNWrapper, self).__init__()
        
        self.additional_layer = nn.Sequential(
            nn.Linear(1536, 1536),
            nn.InstanceNorm1d(1536),
            nn.ReLU(),
        
        )

    def forward(self, x):
         
        x = self.additional_layer(x) 
        return x

 



lin_model = DuelCNNWrapper()  
lin_model.load_state_dict(torch.load('/export/kbodla/metric_model_proxyanchormoment_new.pth'))
lin_model.to(DEVICE)

class PWNet(nn.Module):

    def __init__(self):
        super(PWNet, self).__init__()
        self.ts = ListModule(self, 'ts_')
        for i in range(NUM_PROTOTYPES):
            transformation = nn.Sequential(
                nn.Linear(LATENT_SIZE, PROTOTYPE_SIZE),
                nn.InstanceNorm1d(PROTOTYPE_SIZE),
                nn.ReLU(),
                nn.Linear(PROTOTYPE_SIZE, PROTOTYPE_SIZE),
            )
            self.ts.append(transformation)  
        self.prototypes = None
        self.epsilon = 1e-5
        self.linear = nn.Linear(NUM_PROTOTYPES, NUM_CLASSES, bias=False) 
        self.__make_linear_weights()
        self.softmax = nn.Softmax(dim=1)
        self.nn_human_x = nn.Parameter( torch.randn(NUM_PROTOTYPES, LATENT_SIZE), requires_grad=False)
        
    def __make_linear_weights(self):
        prototype_class_identity = torch.zeros(NUM_PROTOTYPES, NUM_CLASSES)
        num_prototypes_per_class = NUM_PROTOTYPES // NUM_CLASSES
        for j in range(NUM_PROTOTYPES):
            prototype_class_identity[j, j // num_prototypes_per_class] = 1
        positive_one_weights_locations = torch.t(prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations
        incorrect_strength = 0.0
        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.linear.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)
        
    def __proto_layer_l2(self, x, p):
        output = list()
        b_size = x.shape[0]
        p = p.view(1, PROTOTYPE_SIZE).tile(b_size, 1).to(DEVICE) 
        c = x.view(b_size, PROTOTYPE_SIZE).to(DEVICE)      
        l2s = ( (c - p)**2 ).sum(axis=1).to(DEVICE) 
        act = torch.log( (l2s + 1. ) / (l2s + self.epsilon) ).to(DEVICE)  
        return act
    
    def __output_act_func(self, p_acts):        
        return self.softmax(p_acts)
    
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


def evaluate_loader(model, loader, cce_loss):
    model.eval()
    total_correct = 0
    total_loss = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)            
            logits = model(imgs)
            loss = cce_loss(logits, labels)
            preds = torch.argmax(logits, dim=1)
            total_correct += sum(preds == labels).item()
            total += len(preds)
            total_loss += loss.item()
    return (total_correct / total) * 100


def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = toml.load(f)
    return config


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

with open('/export/kbodla/data/X_train_feat.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('/export/kbodla/data/a_train.pkl', 'rb') as f:
    a_train = pickle.load(f)
nn_human_x = list()
p_idxs = list()

# for i in range(NUM_CLASSES):
#     idxs = a_train == i
#     temp_x = X_train[idxs]
#     mean = temp_x.mean(axis=0)
#     knn = KNeighborsClassifier().fit(temp_x, list(range(len(temp_x))))
#     idx = knn.kneighbors(X=mean.reshape(1,-1), n_neighbors=1, return_distance=False)
#     p_idxs.append(idx.item())
#     nn_human_x.append( temp_x[idx.item()].tolist() )
# nn_human_x = np.array(nn_human_x)
nn_human_x = np.load("prot_replace_proxyanchormoment_new_intrain.npy")

#### Start Collecting Data To Form Final Mean and Standard Error Results
data_rewards = list()
data_errors = list()
model = PWNet().eval()
model.to(DEVICE)
model.nn_human_x.data.copy_( torch.tensor(nn_human_x) )
model.load_state_dict(torch.load('weights/pw_net.pth' ))
# agent.online_model.to('cuda')
counts = []

count_ep = []
for episode in tqdm(range(SIMULATION_EPOCHS)):
    
    startTime = time.time()  # Keep time
    state,_ = environment.reset()  # Reset env
    # state = state.to('cuda')
    state = agent.preProcess(state)  # Process image
    
    # Stack state . Every state contains 4 time contionusly frames
    # We stack frames like 4 channel image
    # state = np.array(state)
    state = np.stack((state, state, state, state))

    total_max_q_val = 0  # Total max q vals
    total_reward = 0     # Total reward for each episode
    total_loss = 0       # Total loss for each episode
    total_error = list()
    count = 0
    # state = torch.tensor(state)
    # state.to('cuda')
    for step in range(MAX_STEP):
        count+=1
        # Select and perform an action
        agent_action, latent_x = agent.act(state)
        # test_state = torch.tensor(state)
        # test_state = test_state.to('cuda').float()
        
        latent = lin_model(latent_x.unsqueeze(0))# Act
        action = torch.argmax(model(latent)).item()

        # print(agent_action, action)

        # Normally the randomness is the number on the right (.049...)
        # But as PW-Net is trained on the data from the original model which was already random
        # we lower the randomness here for a fairer comparison.
        # PW-Net here is trained on ~5% random data, plus 0.025 randomness
        # if np.random.random_sample() < .025:   #  .04953625663766238:
        #     action = np.random.randint(0, 5)

        next_state, reward, done,_, info = environment.step(action) 
        # next_state, reward, done,_, info = environment.step(agent_action)

        next_state = agent.preProcess(next_state)  # Process image

        # Stack state . Every state contains 4 time contionusly frames
        # We stack frames like 4 channel image
        next_state = np.stack((next_state, state[0], state[1], state[2]))

        # Store the transition in memory
        # agent.storeResults(state, action, reward, next_state, done)  # Store to mem

        # Move to the next state
        state = next_state  # Update state

        # total_reward += reward
        # total_error.append( agent_action == action )

        if done:
            # all_rewards.append(total_reward)
            # all_errors.append( sum(total_error) / len(total_error ) )
            counts.append(count)
            break


print(" ")
print("-----")
print(counts)


counts = np.array(counts)


print(" ")
print("===== Data Accuracy:")
print("Mean:", counts.mean())
print("Standard Error:", counts.std() / np.sqrt(SIMULATION_EPOCHS)  )
