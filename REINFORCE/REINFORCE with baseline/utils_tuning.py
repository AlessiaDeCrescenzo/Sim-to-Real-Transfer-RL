# utils_tuning.py

import torch
import gym
from env.custom_hopper import *
from agent import AgentREINFORCE, PolicyREINFORCE

def train(train_env='CustomHopper-source-v0', device='cpu', episodes=20, lr=1e-3, gamma=0.99, baseline=0):    #solo per ora numero davvero basso di episodi
    env = gym.make(train_env)

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = PolicyREINFORCE(observation_space_dim, action_space_dim)
    agent = AgentREINFORCE(policy, device=device, baseline=baseline)
    agent.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    agent.gamma = gamma

    #print('Action space:', env.action_space)
    #print('State space:', env.observation_space)
    #print('Dynamics parameters:', env.get_parameters())

    for episode in range(episodes):
        done = False
        train_reward = 0
        state = env.reset()  # Reset the environment and observe the initial state

        while not done:  # Loop until the episode is over
            action, action_probabilities = agent.get_action(state)
            previous_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(previous_state, state, action_probabilities, reward, done)

            train_reward += reward

        agent.update_policy()  # Update policy at the end of the episode

    return agent

def test(agent, episodes=100, test_env='CustomHopper-source-v0'):
    env = gym.make(test_env)

    test_return = 0
    for episode in range(episodes):
        done = False
        state = env.reset()

        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, done, info = env.step(action.detach().cpu().numpy())
            test_return += reward

    return test_return / episodes
