import gym
import gym_sip
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import helpers as h
import random
import numpy as np

EPOCHS = 50000

# main_init()
env = gym.make('Sip-v0').unwrapped
# env.reset()

BATCH_SIZE = 1
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000

N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES + 4))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # print(x)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

steps_done = 0
reward_sum = 0
reward_list = []

# init
prev_state = env.game.cur_state
cur_state = env.game.cur_state
s = (cur_state - prev_state)

dqn = DQN()
num_games = 500

x_axis = []
y_axis = [] 

for game_num in range(num_games):  # run on set number of games
    if game_num % 100 == 0:
        print("GAME: ", end='')
        print(game_num)
        print('\n')
    try:
        cur_state, d = env.next()
    except IndexError:
        break
    for i in range(EPOCHS):
        a = dqn.choose_action(s)  # give deep q network state and return action
        ns, r, d, odds = env.step(a)  # next state, reward, done, 
        print('reward: ', end='')
        print(r)
        print('\n')
        if ns is not None:
            if env.init_a_bet.a_odds != 0 and env.init_a_bet.h_odds != 0:
                awaysale_price = h.awaysale_price(env.init_a_bet, odds)
                homesale_price = h.homesale_price(env.init_h_bet, odds)

                points_sum = env.cur_state[3] + env.cur_state[4]

                if homesale_price > 1000: 
                    print("homesale_price high")
                    print(homesale_price)
                    print(awaysale_price)
                    env.init_h_bet.__repr__()
                    print(odds)
                    # homesale_price = -99

                # print(str(homesale_price) + " hs_price")

                x_axis.append(i)
                # y_axis.append(homesale_price)
                y_axis.append(r)

            dqn.store_transition(s, a, r, odds)
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
            if not d:
                prev_state = cur_state
                cur_state = ns
                s = cur_state - prev_state
            else:
                break
        else:
            break

print(env.money)
print(len(x_axis))
np_x_axis = np.array(x_axis)
np_y_axis = np.array(y_axis)

# np_rl = np.array(reward_list)
# np_rl = np_rl.astype(float)

plt.scatter(np_x_axis, np_y_axis, s=2.5, alpha=.5)
plt.show()
