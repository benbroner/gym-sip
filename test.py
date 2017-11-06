import gym
import gym_stocks
import random

env = gym.make('Stocks-v0')
print env.reset()

for i in range(10):
    print env.step(random.randrange(0, env.action_space.n))
    #print env.step(0)
