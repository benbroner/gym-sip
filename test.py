import gym
import random

import gym_sip
import matplotlib.pyplot as plt

env = gym.make('Sip-v0')
env.reset()

reward_sum = 0

epochs = 1000
plot_count = 100
div = epochs/plot_count
ecount = 500
div2 = epochs / ecount

for i in range(epochs):
    metadata = {'render.modes': ['rgb_array']}
    s, r, d, m = env.step(random.randrange(0, env.action_space.n))
    reward_sum += r

    if i % (epochs / div) == 0:
        print(str(reward_sum))

    if i % 20 == 0:
        print('EPOCH: ' + str(i))

    if d == 1:
        env.next()

    plt.scatter(i, r, color='r', s=10, marker='o')
    plt.scatter(i, reward_sum, color='b', s=10, marker='o')


print(str(reward_sum))
plt.show()
