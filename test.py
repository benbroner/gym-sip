import gym
import random
import deepq as dq
import gym_sip
import matplotlib.pyplot as plt

EPOCHS = 500
EPISODES = 50

# make
env = gym.make('Sip-v0')
env.reset()

reward_sum = 0
for ep in range(EPISODES):
	# init
	prev_state, prev_done = env.game.next()
	cur_state, cur_done = env.game.next()
	s = cur_state - prev_state
    for i in range(EPOCHS):

    	action = dq.select_action(s)  # selecting action using deep q network
        s, r, d, info = env.step()
        r_tensor = torch.tensor([r])

        # Observe new state TODO
        prev_line = cur_line
        cur_line = s 

        if not done:
            next_state = cur_line - prev_line
        else:
            next_state = None

        # Store the transition in memory
        memory.push(s, action, next_state, r_tensor)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        dq.optimize_model()
        if done:
            episode_durations.append(t + 1)
            dq.plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
