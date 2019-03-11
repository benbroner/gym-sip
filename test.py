import gym
import gym_sip


import random
import numpy as np


# main_init()
env = gym.make('Sip-v0').unwrapped
env.reset()

steps_done = 0
reward_sum = 0

# init
prev_state = env.cur_state
cur_state = env.cur_state
s = (cur_state - prev_state)

for ep in range(EPISODES):


    for i in range(EPOCHS):
        
        s, r, d, info = env.step(random.randrange(0, env.action_space.n))

        if not done:
            next_state = cur_state - prev_state
            # next_state = cur_state
        else:
            next_state = None

        s = next_state

        optimize_model()
    # Update the target network, copying all weights and biases in DQN
    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
