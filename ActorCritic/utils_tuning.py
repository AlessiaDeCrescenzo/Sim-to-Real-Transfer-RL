import torch
import gym
from env.custom_hopper import *
from agent2 import Agent, ActorCritic
from env.Wrapper import TrackRewardWrapper
import random

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

# Define the training function
def train(train_env='CustomHopper-source-v0', device='cpu', episodes=5000, lr=1e-3, gamma=0.99, hidden=64):
    # Create the environment using the specified environment name
    seed1=np.random.randint(1)

    env = gym.make(train_env)
    env=TrackRewardWrapper(env)
    env.seed(seed1)

    # Get the dimensions of the observation space and action space
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    # Initialize the actor, critic, and agent
    actorcritic = ActorCritic(observation_space_dim, action_space_dim)
    
    # Adjust the hidden layer size if necessary
    actorcritic.hidden = hidden

    agent = Agent(actorcritic, device=device, lr=lr)

    # Set the discount factor (gamma) for the agent
    agent.gamma = gamma

    # Loop over the number of episodes
    for episode in range(episodes):
        done = False
        train_reward = 0

        # Reset the environment at the start of each episode
        state = env.reset()

        # Loop until the episode ends
        while not done:
            # Get the action and the action probabilities from the agent
            action, action_probabilities = agent.get_action(state)
            previous_state = state

            # Take the action in the environment and observe the next state and reward
            state, reward, done, info = env.step(action.detach().cpu().numpy())

            # Store the outcome of the step in the agent's memory
            agent.store_outcome(previous_state, state, action_probabilities, reward, done)

            # Accumulate the reward
            train_reward += reward

        # Update the policy at the end of the episode
        agent.update_policy()

    return agent

# Define the testing function
def test(agent, episodes=100, test_env='CustomHopper-source-v0'):
    # Create the environment for testing
    seed1=np.random.randint(1)

    env = gym.make(test_env)
    env=TrackRewardWrapper(env)
    env.seed(seed1)

    test_return = 0

    # Loop over the number of testing episodes
    for episode in range(episodes):
        done = False

        # Reset the environment at the start of each episode
        state = env.reset()

        # Loop until the episode ends
        while not done:
            # Get the action from the agent (in evaluation mode)
            action, _ = agent.get_action(state, evaluation=True)

            # Take the action in the environment and observe the next state and reward
            state, reward, done, info = env.step(action.detach().cpu().numpy())

            # Accumulate the reward
            test_return += reward

    # Return the average return over all testing episodes
    return test_return / episodes
