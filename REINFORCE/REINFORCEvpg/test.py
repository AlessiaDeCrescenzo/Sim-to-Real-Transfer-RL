"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import AgentREINFORCE, PolicyREINFORCE
import pickle
from env.Wrapper import TrackRewardWrapper
import matplotlib.pyplot as plt
from utils_tuning import set_seed

def plot_rewards(reward_buffer, num_episodes):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_episodes + 1), reward_buffer, marker='o', linestyle='-')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward over Episodes')
    plt.grid(True)
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='REINFORCE/REINFORCEvpg/model_reinforce.mdl', type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')
    parser.add_argument('--fine_tuning_params', default='REINFORCE/REINFORCEvpg/result_REINFORCE.pkl', type=str, help='Path to fine-tuning parameters')
    return parser.parse_args()

args = parse_args()


def main():
	set_seed(315304)
	
	env = gym.make('CustomHopper-source-v0')
	env=TrackRewardWrapper(env)
	env.seed(315304)

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())

	# Load fine-tuning parameters
	with open(args.fine_tuning_params, 'rb') as infile:
		fine_tuning_params = pickle.load(infile)[1]  # [1] because you only need the config, not the score

	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]
    
	
	policy = PolicyREINFORCE(observation_space_dim, action_space_dim, hidden=fine_tuning_params['hidden'])
	policy.load_state_dict(torch.load(args.model), strict=True)
	agent = AgentREINFORCE(policy, device=args.device, lr=fine_tuning_params['lr'], gamma=fine_tuning_params['gamma'])

	for episode in range(args.episodes):
		done = False
		test_reward = 0
		state = env.reset()

		while not done:

			action, _ = agent.get_action(state, evaluation=True)

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			if args.render:
				env.render()

			test_reward += reward

		print(f"Episode: {episode} | Return: {test_reward}")
	
	plot_rewards(env.buffer, len(env.buffer))
	

if __name__ == '__main__':
	main()