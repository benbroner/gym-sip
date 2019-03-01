import gym
import random

import gym_sip
import matplotlib.pyplot as plt
env = gym.make('Sip-v0')
env.reset()

reward_sum = 0

for i in range(20000):
    s, r, d, m = env.step(random.randrange(0, env.action_space.n))
    print(str(r))
    plt.scatter(i, reward_sum, color='r', s=10, marker='o')
    reward_sum += r
    if env.money <= 0:
        print('reset')
        env.reset()
    if i % 100 == 0:
        print("tot: " + str(reward_sum))
        print("MONEY: " + str(env.money))
    if d == 1:
        env.next()
plt.show()
