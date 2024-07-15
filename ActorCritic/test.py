"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import Agent, ActorCritic
from env.Wrapper import TrackRewardWrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actorcriticmodel', default='ActorCritic/actor_model.mdl', type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=True, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()


def main():

	seed=316619
	set_seed(seed)

	env = gym.make('CustomHopper-source-v0')
	
	env= TrackRewardWrapper(env)

	env.seed(seed)

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	actorcritic = ActorCritic(observation_space_dim, action_space_dim)
	actor.load_state_dict(torch.load(args.actorcriticmodel), strict=True)


	agent = Agent(actor,critic, device=args.device)

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
	

if __name__ == '__main__':
	main()