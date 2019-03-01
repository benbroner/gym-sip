import gym
import random

import gym_sip
import matplotlib.pyplot as plt
env = gym.make('Sip-v0')
env.reset()

reward_sum = 0

for i in range(10000):
    s, r, d, m = env.step(random.randrange(0, env.action_space.n))
    print('reward: ' + str(r))
    plt.scatter(i, reward_sum, color='r', s=2, marker='o')
    plt.scatter(i, env.money, color='b', s=2, marker='o')
    if r != -25:
        plt.scatter(i, r, color='g', s=2, marker='o')
    reward_sum += r
    if env.money <= 0:
        print('reset')
        env.reset()
    print("tot: " + str(reward_sum))
    print("money: " + str(env.money))
    if i % 100 == 0:
        print("i: " + str(i))
    if d == 1:
        env.next()

plt.show()
