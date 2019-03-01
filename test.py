import gym
import random

import gym_sip
import matplotlib.pyplot as plt
env = gym.make('Sip-v0')
env.reset()

reward_sum = 0

for i in range(10000):
    s, r, d, m = env.step(random.randrange(0, env.action_space.n))
    print(str(r))
    plt.scatter(i, reward_sum, color='r', s=10, marker='o')
    reward_sum += r
    if i % 100 == 0:
        print(str(reward_sum))
    if d == 1:
        env.next()
plt.show()