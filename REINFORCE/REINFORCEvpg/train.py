"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse
import matplotlib.pyplot as plt  # Add this import
import torch
import gym
from env.custom_hopper import *
from agent import AgentREINFORCE, PolicyREINFORCE
import pickle #dovrebbe servire per leggere il file
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')    #default=100000
    parser.add_argument('--print-every', default=5000, type=int, help='Print info every <> episodes')  #default=20000
    parser.add_argument('--device', default='cpu', type=str, help='Network device [cpu, cuda]')
    parser.add_argument('--fine-tuning-params', default='best_fine_tuning_result.pkl', type=str, help='Path to fine-tuning parameters')  #per leggere iperparameter del fine tunig
    return parser.parse_args()

args = parse_args()

def seed_everything(seed,env):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    env.seed(seed)

def main():
    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    seed_everything(316619,env)
    #env.seed(315304)

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())



    # Load fine-tuning parameters
    with open(args.fine_tuning_params, 'rb') as infile:
        fine_tuning_params = pickle.load(infile)[1]  # [1] because you only need the config, not the score

    # Training

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = PolicyREINFORCE(observation_space_dim, action_space_dim, hidden=fine_tuning_params['hidden'])  #METTERE HIDDEN#in questo modo dovrebbe cambiare i parametri di default in quegli migliori
    agent = AgentREINFORCE(policy, device=args.device, lr=fine_tuning_params['lr'], gamma=fine_tuning_params['gamma'])

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
        agent.update_policy()  # Update policy at the end of the episode

        if (episode + 1) % args.print_every == 0:
            print('Training episode:', episode)
            print('Episode return:', train_reward)
            print(f'Mean return (last 1000 episodes): {mean_rewards}')


    torch.save(agent.policy.state_dict(), "model_reinforce_2.mdl")

    plt.plot(returns, label='Episode Return')
    plt.plot(avg_returns, label='Average Return (last 1000 episodes)', linestyle='--') 
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Episode returns over time')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

