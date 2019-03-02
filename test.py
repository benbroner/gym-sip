import gym
import random

import gym_sip
import matplotlib.pyplot as plt
env = gym.make('Sip-v0')
env.reset()

reward_sum = 0

for i in range(30000):
    if i % 100 == 0:
        print("i: " + str(i))
    s, r, d, m = env.step(random.randrange(0, env.action_space.n))
    reward_sum += r

    print('reward: ' + str(r))
    print("money: " + str(env.money) + '\n\n')

    plt.scatter(i, reward_sum, color='r', s=2, marker='o')
    plt.scatter(i, env.money, color='b', s=2, marker='o')

    if r != -25:
        plt.scatter(i, r, color='g', s=2, marker='o')

    if env.money <= 0:
        print('reset')
        env.reset()

    if d == 1:
        env.next()

plt.show()
