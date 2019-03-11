import gym
import gym_sip
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


# credit to https://github.com/apaszke
# adapted to my own gym environment


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(output_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, input_size)

    def forward(self, x):
        x = F.relu(self.l1(x.float()))
        x = F.relu(self.l2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x.double()

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    s = torch.tensor(state, device=device)
    s = s.reshape(1, 14)
    # print(s)
    if sample > eps_threshold:
        with torch.no_grad():

            return policy_net(s).max(1)[1].view(1, 1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# main_init()
env = gym.make('Sip-v0').unwrapped
env.reset()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

EPOCHS = 500
EPISODES = 50
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# num_cols = tmp_df.shape[1]

input_size = 14
hidden_size = 50
output_size = 2

policy_net = Net(input_size, hidden_size, output_size).to(device)
target_net = Net(input_size, hidden_size, output_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

reward_sum = 0
for ep in range(EPISODES):
    # init
    prev_state = env.cur_state
    cur_state = env.cur_state
    s = (cur_state - prev_state)
    # s = cur_state
    # print(s)
    # TODO can't train on derivative because datetime adding does not work

    for i in range(EPOCHS):

        action = select_action(s)  # selecting action using deep q network
        print(action)
        ret_state, r, done, info = env.step(action)
        r_tensor = torch.tensor([r], device=device)

        # Observe new state TODO
        prev_state = cur_state
        cur_state = ret_state

        if not done:
            next_state = cur_state - prev_state
            # next_state = cur_state
        else:
            next_state = None

        # Store the transition in memory
        memory.push(s, action, next_state, r_tensor)

        s = next_state

        optimize_model()
    # Update the target network, copying all weights and biases in DQN
    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
