import gym
import random

import gym_sip
env = gym.make('Sip-v0')
print(env.reset())

for i in range(1000):
    env.step(random.randrange(0, env.action_space.n))
