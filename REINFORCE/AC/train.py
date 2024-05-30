"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym
import matplotlib.pyplot as plt

from env.custom_hopper import *
#from agentAC import Agent, Policy
from agent import AgentAC, PolicyAC

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100, type=int, help='Number of training episodes')  #Default=10000 
    parser.add_argument('--print-every', default=10, type=int, help='Print info every <> episodes')   #default=500,
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = PolicyAC(observation_space_dim, action_space_dim)
	agent = AgentAC(policy, device=args.device)

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

	returns=[]

	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over

			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done)

			train_reward += reward

		returns.append(train_reward)
		agent.update_policy()	
		
		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)


	torch.save(agent.policy.state_dict(), "model.mdl")

	plt.plot(returns)
	plt.xlabel('Episode')
	plt.ylabel('Return')
	plt.title('Episode returns over time')
	plt.show()
	

if __name__ == '__main__':
	main()