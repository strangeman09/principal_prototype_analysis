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
import toml
import imageio
import os

from copy import deepcopy
from torch.distributions import Beta
from tqdm import tqdm
from games.carracing import RacingNet, CarRacing
from ppo import PPO

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


class PWNet(nn.Module):

    def __init__(self):
        super(PWNet, self).__init__()
        self.ts = ListModule(self, 'ts_')
        for i in range(NUM_PROTOTYPES):
            transformation = nn.Sequential(
                nn.Linear(LATENT_SIZE, PROTOTYPE_SIZE),
                # nn.BatchNorm1d(PROTOTYPE_SIZE),
                # nn.InstanceNorm1d(PROTOTYPE_SIZE, affine=True, track_running_stats=True),
                nn.ReLU(),
                nn.Linear(PROTOTYPE_SIZE, PROTOTYPE_SIZE),
            )
            self.ts.append(transformation)  
        self.epsilon = 1e-5
        self.linear = nn.Linear(NUM_PROTOTYPES, NUM_CLASSES, bias=False) 
        self.__make_linear_weights()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU() 
        self.nn_human_x = nn.Parameter( torch.randn(NUM_PROTOTYPES, LATENT_SIZE), requires_grad=False)
        
    def __make_linear_weights(self):
        """
        Must be manually defined to connect prototypes to human-friendly concepts
        For example, -1 here corresponds to steering left, whilst the 1 below it to steering right
        Together, they can encapsulate the overall concept of steering
        More could be connected, but we just use 2 here for simplicity.
        """

        custom_weight_matrix = torch.tensor([
                                             [-1., 0., 0.], 
                                             [ 1., 0., 0.],
                                             [ 0., 1., 0.], 
                                             [ 0., 0., 1.],
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
        """
        Use appropriate activation functions for the problem at hand
        Here, tanh and relu make the most sense as they bin the possible output
        ranges to be what the car is capable of doing.
        """

        p_acts.T[0] = self.tanh(p_acts.T[0])  # steering between -1 -> +1
        p_acts.T[1] = self.relu(p_acts.T[1])  # acc > 0
        p_acts.T[2] = self.relu(p_acts.T[2])  # brake > 0
        return p_acts
    
    def forward(self, x):
        
        # Get the latent prototypes by putting them through the individual transformations
        trans_nn_human_x = list()
        for i, t in enumerate(self.ts):
            trans_nn_human_x.append( t( torch.tensor(self.nn_human_x[i], dtype=torch.float32).view(1, -1)) )
        latent_protos = torch.cat(trans_nn_human_x, dim=0)   
            
        # Do similarity of inputs to prototypes
        p_acts = list()
        for i, t in enumerate(self.ts):
            action_prototype = latent_protos[i]
            p_acts.append( self.__proto_layer_l2( t(x), action_prototype).view(-1, 1) )
        p_acts = torch.cat(p_acts, axis=1)
        
        # Put though activation function method
        logits = self.linear(p_acts)                     
        final_outputs = self.__output_act_func(logits)   
        
        return final_outputs, p_acts


def evaluate_loader(model, loader, loss):
    model.eval()
    total_error = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            current_loss = loss(logits, labels)
            total_error += current_loss.item()
            total += len(imgs)
    model.train()
    return total_error / total


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


def proto_loss(model, nn_human_x, criterion):
    model.eval()
    target_x = trans_human_concepts(model, nn_human_x)
    loss = criterion(model.prototypes, target_x) 
    model.train()
    return loss
    

def trans_human_concepts(model, nn_human_x):
    model.eval()
    trans_nn_human_x = list()
    for i, t in enumerate(model.ts):
        trans_nn_human_x.append( t( torch.tensor(nn_human_x[i], dtype=torch.float32).view(1, -1)) )
    model.train()
    return torch.cat(trans_nn_human_x, dim=0)


#### Start Collecting Data To Form Final Mean and Standard Error Results



cfg = load_config()
env = CarRacing(frame_skip=0, frame_stack=4,)
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

model = PWNet().eval()
model.load_state_dict(torch.load(MODEL_DIR))
nn_human_x = np.load('original_prototypes.npy')
# nn_human_x = np.load('piecewise_prot_indata.npy')
self_state = ppo._to_tensor(env.reset())
reward_arr = list()
states, real_actions, rewards, model_actions, activations = [], [], [], [], []


NUM_EPISODES= 1

############################################old demoe######################
# for ep in tqdm(range(NUM_EPISODES)):

#     next_state = ppo.env.reset()
#     rew = 0
#     done = False
#     count = 0

#     ep_actions = list()
#     ep_x = list()

#     while not done:

#         count += 1

#         # Run one step of the environment based on the current policy
#         value, alpha, beta, x = ppo.net(self_state)
#         value, alpha, beta = value.squeeze(0), alpha.squeeze(0), beta.squeeze(0)

#         policy = Beta(alpha, beta)
    
#         # Choose how to get actions (sample or take mean)
#         action, activation = model(x)
#         input_action = policy.mean.detach()
#         # input_action = policy.sample()

#         next_state, reward, done, info, real_action = ppo.env.step(input_action.cpu().numpy())
#         next_state = ppo._to_tensor(next_state)
#         # x = model(self_state)
#         # Store the transition
#         # ep_actions.append(real_action.tolist())
      
#         # ep_x.append(x.detach().cpu().numpy())

#         real_actions.append(real_action.tolist())
#         model_actions.append(action.tolist())
      
#         # X_train.append(x.detach().cpu().numpy())
#         states.append(self_state)
#         activations.append(activation.detach().cpu().numpy())
        

#         self_state = next_state
#         rew += reward
        
#     reward_arr.append(rew)
#     print(count)
#     rew += reward

#######################################old demoe ######################

os.makedirs("videos", exist_ok=True)

frames = []
frame_activations = []


reward_arr = []

for ep in tqdm(range(NUM_EPISODES)):
    state = ppo.env.reset()
    state = ppo._to_tensor(state)
    done = False
    ep_reward = 0

    while not done:
        value, alpha, beta, latents = ppo.net(state)
        policy = Beta(alpha.squeeze(0), beta.squeeze(0))
        proto_action, activation = model(latents)

        action = policy.mean.detach().cpu().numpy()
        next_state, reward, done, info, _ = ppo.env.step(action)
        state = ppo._to_tensor(next_state)

        # Save frame
        frame = ppo.env.render(mode="rgb_array")
        frames.append(frame)

        # Save model activations
        frame_activations.append(activation.detach().cpu().numpy())

        ep_reward += reward

    print(f"Episode {ep+1} completed: Reward = {ep_reward}")
    reward_arr.append(ep_reward)


# ==== SAVE OUTPUTS ====
imageio.mimsave("videos/episode.mp4", frames, fps=30)
np.save("videos/frame_activations.npy", np.array(frame_activations))

print("\n✔ Saved:")
print("  🎥 videos/episode.mp4")
print("  📊 videos/frame_activations.npy")
print(f"\nAverage Reward = {np.mean(reward_arr):.2f}")
env.close()







# print("average reward per episode :", sum(reward_arr) / NUM_EPISODES)






# np.save('data/real_actions.npy', real_actions)
# np.save('data/images.npy', states)
# np.save('data/activations.npy', model_actions)
# np.save('data/model_actions.npy', activations)



