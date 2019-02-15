import gym
import gym_sip
import random

env = gym.make('Sipper-v0')
env.reset()

for i in range(10):
    env.step(random.randrange(0, env.action_space.n))
