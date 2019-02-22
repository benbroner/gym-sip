import gym
import random

import gym_sip
env = gym.make('Sip-v0')
env.reset()

reward_sum = 0

for i in range(1):
    s, r, d, m = env.step(random.randrange(0, env.action_space.n))
    # print(s['secs', 'a_pts', 'h_pts', ])
    # print(s)
    # reward_sum += r
    # print(str(reward_sum))
