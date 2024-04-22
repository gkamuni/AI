
# A3C for Kung Fu

## Part 0 - Installing the required packages and importing the libraries

### Installing Gymnasium
"""

!pip install gymnasium
!pip install "gymnasium[atari, accept-rom-license]"
!apt-get install -y swig
!pip install gymnasium[box2d]

"""### Importing the libraries"""

import cv2
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributions as distributions
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

"""## Part 1 - Building the AI

### Creating the architecture of the Neural Network
"""

class Network(nn.Module):
  def __init__(self, action_size):
    super(Network, self).__init__()
    self.conv1 = nn.Conv2d(in_channels = 4, out_channels = 32, kernel_size = (3,3), stride = 2)
    self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), stride = 2)
    self.conv3 = nn.Conv2d(in_channels= 32, out_channels = 32, kernel_size = (3,3), stride = 2)
    self.flatten = torch.nn.Flatten()
    self.fc1 = torch.nn.Linear(512, 128)
    # we will be having two output layers. The Q values(action values) AND THE CRITIC(state values)
    # we will use fc2a and fc2s
    self.fc2a = torch.nn.Linear(128, action_size)         #action or the Qvalues
    self.fc2s = torch.nn.Linear(128, 1)              #state values, 1 because it will only give out one value


  def forward(self, state):
    x = self.conv1(state)                   # the state will be passed through the first convolutional layer
    x = F.relu(x)                           # relu function to actvate the signal
    x = self.conv2(x)                       # similar forward propogation to the rest of the layers
    x = F.relu(x)
    x = self.conv3(x)
    x = F.relu(x)
    x = self.flatten(x)                      # from the flattening layer, we have to forward propogate the signal to the first input layer
    x = self.fc1(x)
    x = F.relu(x)
    # now we have 2 layers, fc2a and fc2s
    action_values = self.fc2a(x)
    state_value = self.fc2s(x)[0]          # the [0] is to access one value(not in vector) in the future
    return action_values, state_value

"""## Part 2 - Training the AI

### Setting up the environment
"""

# setting up the environment is a little complicated, but got it. Yaay.
# go through the defined function, you will understand
class PreprocessAtari(ObservationWrapper):

  def __init__(self, env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4):
    super(PreprocessAtari, self).__init__(env)
    self.img_size = (height, width)
    self.crop = crop
    self.dim_order = dim_order
    self.color = color
    self.frame_stack = n_frames
    n_channels = 3 * n_frames if color else n_frames
    obs_shape = {'tensorflow': (height, width, n_channels), 'pytorch': (n_channels, height, width)}[dim_order]
    self.observation_space = Box(0.0, 1.0, obs_shape)
    self.frames = np.zeros(obs_shape, dtype = np.float32)

  def reset(self):
    self.frames = np.zeros_like(self.frames)
    obs, info = self.env.reset()
    self.update_buffer(obs)
    return self.frames, info

  def observation(self, img):
    img = self.crop(img)
    img = cv2.resize(img, self.img_size)
    if not self.color:
      if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.
    if self.color:
      self.frames = np.roll(self.frames, shift = -3, axis = 0)
    else:
      self.frames = np.roll(self.frames, shift = -1, axis = 0)
    if self.color:
      self.frames[-3:] = img
    else:
      self.frames[-1] = img
    return self.frames

  def update_buffer(self, obs):
    self.frames = self.observation(obs)

def make_env():
  env = gym.make("KungFuMasterDeterministic-v0", render_mode = 'rgb_array')
  env = PreprocessAtari(env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4)
  return env

env = make_env()

state_shape = env.observation_space.shape
number_actions = env.action_space.n
print("Observation shape:", state_shape)
print("Number actions:", number_actions)
print("Action names:", env.env.env.get_action_meanings())

"""### Initializing the hyperparameters"""

# we have three hyperparameters
# nthey are 1. Learning rate, 2. discount factor, 3. number of environments (specific to the A3C model because we train multiple agents in multiple environments)
# we will be using 10 environments to train the multiple agents
# more environmets, more experience, more information. Hunger games


learning_rate = 1e-4
discount_factor = 0.99
number_environments = 10

"""### Implementing the A3C class"""

# we will be using a softmax stratagy
# this class will contain 3 methods
# the init method, act method, step method,

class Agent():
  def __init__(self, action_size):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")          # this step is only if the code is being executed in a different computer or compiler
    self.action_size = action_size
    self.network = Network(action_size).to(self.device)                                               #creating the brain by calling the Network Class
    self.optimizer = torch.optim.Adam(self.network.parameters(), lr = learning_rate)                          # Adam is an optimizer used when updating the weights and back propogating

# we will be implementing the softmax startagy
  def act(self, state):
    if state.ndim == 3:                                                            # we need the extra dimension i.e the state need to be in a batch
      state = [state]                                                              # this helps it add the extra dimension to the condition

    state = torch.tensor(state, dtype = torch.float32, device = self.device)                        # we convert numpy arrays to pytorch tensors because tensors are multidimension arrays with more functionalities and can accept ifferent datatypes
      # we call the network function on the state to get the action values
    action_values, _ = self.network(state)                                         # _ is to disregard the state_value which is also an output
      # we use the softmax policy, we used epsilon greedy in our previous codes
    policy = F.softmax(action_values, dim = -1)                                    #changes the action values to probabilities
    return np.array([np.random.choice(len(p), p = p) for p in policy.detach().cpu().numpy()])




    # do not get overwhelmed by the return" line of code. We have mulitple environments returning multiple policies
    # the list of arrays is selected randomly from that environments. That is the reason we use the for loop.

  # finally we will implement the step method. It is long so Focus, you got this
  def step(self, state, action, reward, next_state, done):                      # here this method will take batches of states, sctions, rewards etc
    batch_size = state.shape[0]
       # we convert the parameters to pytorch tensor
    state = torch.tensor(state, dtype = torch.float32, device = self.device)
    next_state = torch.tensor(next_state, dtype = torch.float32, device = self.device)
    reward = torch.tensor(reward, dtype = torch.float32, device = self.device)
    done = torch.tensor(done, dtype = torch.bool, device = self.device).to(dtype = torch.float32)
    action_values, state_value = self.network(state)
      # we also need the next state value from the network and also the target state value
    _, next_state_value = self.network(next_state)
    target_state_value = reward + discount_factor * next_state_value * (1-done)

    # we need advantage feature of the A3C
    advantage = target_state_value - state_value

    # we need the critic feature of the A3C. Actors loss and critics loss
    probs = F.softmax(action_values, dim = -1)
    logprobs = F.log_softmax(action_values, dim = -1)
    entropy = -torch.sum(probs * logprobs, axis = -1)
    batch_idx = np.arange(batch_size)
    logp_actions = logprobs[batch_idx, action]

    actor_loss = -(logp_actions * advantage.detach()).mean() - 0.001 * entropy.mean()
    critic_loss = F.mse_loss(target_state_value.detach(), state_value)

    total_loss = actor_loss + critic_loss

    self.optimizer.zero_grad()
    #back propogation
    total_loss.backward()

    #update the weights
    self.optimizer.step()

"""### Initializing the A3C agent"""

agent = Agent(number_actions)

"""### Evaluating our A3C agent on a Certain number of episodes

"""

def evaluate(agent, env, n_episodes = 1):         #we can chage the environments
  episodes_rewards = []
  for _ in range(n_episodes):
    state, _ = env.reset()
    total_reward = 0
    while True:
      action = agent.act(state)
      state, reward, done, info, _ = env.step(action[0])
      total_reward += reward
      if done:
        break
      episodes_rewards.append(total_reward)
    return episodes_rewards

"""### Testing multiple agents on multiple environments at the same time"""

class EnvBatch:

  def __init__(self, n_envs = 10):
    self.envs = [make_env() for _ in range(n_envs)]

  def reset(self):
    _states = []
    for env in self.envs:
      _states.append(env.reset()[0])
    return np.array(_states)

  def step(self, actions):
    next_states, rewards, dones, infos, _ = map(np.array, zip(*[env.step(a) for env, a in zip(self.envs, actions)]))
    for i in range(len(self.envs)):
      if dones[i]:
        next_states[i] = self.envs[i].reset()[0]
    return next_states, rewards, dones, infos

"""### Training the A3C agent"""

import tqdm

env_batch = EnvBatch(number_environments)
batch_states = env_batch.reset()

with tqdm.trange(0, 3001) as progress_bar:
  for i in progress_bar:
    batch_actions = agent.act(batch_states)
    batch_next_states, batch_rewards, batch_dones, _ = env_batch.step(batch_actions)
    batch_rewards *= 0.01
    agent.step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
    batch_states = batch_next_states
    if i % 1000 == 0:
      print("Average agent reward: ", np.mean(evaluate(agent, env, n_episodes = 10)))

"""## Part 3 - Visualizing the results"""

import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

def show_video_of_model(agent, env):
  state, _ = env.reset()
  done = False
  frames = []
  while not done:
    frame = env.render()
    frames.append(frame)
    action = agent.act(state)
    state, reward, done, _, _ = env.step(action[0])
  env.close()
  imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, env)

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()