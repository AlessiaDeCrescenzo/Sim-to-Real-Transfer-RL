"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym
import matplotlib.pyplot as plt

from env.custom_hopper import *
from agent2 import Agent, ActorCritic
import pickle
import numpy as np
from env.Wrapper import TrackRewardWrapper
from utils_tuning import set_seed,save_rewards


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=35000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=500, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--fine-tuning-params', default='ActorCritic/result_AC.pkl', type=str, help='Path to fine-tuning parameters')
    return parser.parse_args()

args = parse_args()


def main():
	seed=3153
	set_seed(seed)
	
	env = gym.make('CustomHopper-source-v0')
	env=TrackRewardWrapper(env)
	
	env.seed(seed)
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""

	# Load fine-tuning parameters
	with open(args.fine_tuning_params, 'rb') as infile:
		fine_tuning_params = pickle.load(infile)[1]  # [1] because you only need the config, not the score
	

	#print(fine_tuning_params)

	observation_space_dim=env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]
    
	actorcritic = ActorCritic(observation_space_dim, action_space_dim,hidden=fine_tuning_params['hidden'])
	agent = Agent(actorcritic, device=args.device, lr=fine_tuning_params['lr'], gamma=fine_tuning_params['gamma']) 

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #

	returns=[]
	avg_returns = []  # List to store average returns

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
		mean_rewards = np.mean(returns[-1000:])     #-100
		avg_returns.append(mean_rewards)# Calculate average return
		agent.update_policy()	
		
		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)
			print(f'Mean return (last 1000 episodes): {mean_rewards}')

	torch.save(agent.actorcritic.state_dict(), "ActorCritic/actor_model.mdl")

	plt.plot(returns)
	plt.plot(avg_returns, label='Average Return (last 1000 episodes)', linestyle='--') 
	plt.xlabel('Episode')
	plt.ylabel('Return')
	plt.title('Episode returns over time')
	plt.legend()
	plt.show()

	save_rewards('Basic_AC.txt',"Actor Critic", avg_returns)
	

if __name__ == '__main__':
	main()