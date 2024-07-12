# utils_tuning.py

import torch
import gym
from env.custom_hopper import *
from agent import AgentREINFORCE, PolicyREINFORCE
import random
from env.Wrapper import TrackRewardWrapper


# Function to set random seeds
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_rewards(filename, algo_name, rewards):
    
    with open(filename, 'a') as file:
        file.write(f"Name of algo: {algo_name}\n")
        file.write(f"{rewards}\n")
        file.write("\n")  # Add a newline for readability


def train(train_env='CustomHopper-source-v0', device='cpu', episodes=10000, lr=1e-3, gamma=0.99, hidden=64,seed=None):    #solo per ora numero davvero basso di episodi
    
    seed1=np.random.randint(1)

    env = gym.make(train_env)
    env=TrackRewardWrapper(env)
    env.seed(seed1)

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = PolicyREINFORCE(observation_space_dim, action_space_dim)
    agent = AgentREINFORCE(policy, device=device)
    agent.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    agent.gamma = gamma
    policy.hidden = hidden

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

def test(agent, episodes=200, test_env='CustomHopper-source-v0',seed=None):
    # Set the seed
    seed1=np.random.randint(1)
    env = gym.make(test_env)
    env=TrackRewardWrapper(env)
    env.seed(seed1)

    test_return = 0
    for episode in range(episodes):
        done = False
        state = env.reset()

        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, done, info = env.step(action.detach().cpu().numpy())
            test_return += reward

    return test_return / episodes
