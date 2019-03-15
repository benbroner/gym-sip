import gym
import gym_sip


import random
import numpy as np

EPOCHS = 50000

# main_init()
env = gym.make('Sip-v0').unwrapped
env.reset()

steps_done = 0
reward_sum = 0

# init
prev_state = env.game.cur_state
cur_state = env.game.cur_state
s = (cur_state - prev_state)

for ep in range(5):

    for i in range(EPOCHS):
        # num of games
        s, r, d, odds = env.step(random.randrange(0, env.action_space.n))
        # print(odds)

        if not d:
            prev_data = cur_state
            cur_state = s
            next_state = cur_state - prev_state
            # next_state = cur_state
        else:
            cur_state, d = env.next()
            next_state = None

        s = next_state

        # optimize_model()
    # Update the target network, copying all weights and biases in DQN
    # if ep % TARGET_UPDATE == 0:
    #     target_net.load_state_dict(policy_net.state_dict())

for hedge in env.hedges:
    hedge.__repr__()
print(env.money)