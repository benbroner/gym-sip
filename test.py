import gym
import random

import gym_sip
import matplotlib.pyplot as plt

env = gym.make('Sip-v0')
env.reset()

reward_sum = 0
for ep in range(num_eps):

    for i in count():

        s, r, d, m = env.step(random.randrange(0, env.action_space.n))
        reward_sum += r

        if env.money <= 0:
            print('out of money')
            env.reset()

        if d == 1:
            env.next()

plt.show()
