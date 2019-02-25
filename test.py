import gym
import random

import gym_sip
env = gym.make('Sip-v0')
env.reset()

reward_sum = 0

for i in range(1000):
    metadata = {'render.modes': ['rgb_array']}
    s, r, d, m = env.step(random.randrange(0, env.action_space.n))
    reward_sum += r
    # print(m)
    if i % 100 == 0:
        print(str(reward_sum))
    if d == 1:
        env.next()

print(str(reward_sum))
