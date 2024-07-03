"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from udr_env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--source-log-path', type=str)
    parser.add_argument('--target-log-path', type=str)
    parser.add_argument('--episodes', default=100_000, type=int)
    parser.add_argument('--evalepisodes',default=100,type=int)
    return parser.parse_args()

args = parse_args()

if args.train is None or args.source_log_path is None or args.target_log_path is None:
    exit('Arguments required')


#N_ENVS = os.cpu_count()
MAX_EPS = args.episodes
#ENV_EPS = int(np.ceil(MAX_EPS / N_ENVS))

def compute_bounds(params):
    bounds = list((m-hw,m+hw) for m,hw in [(params['thigh_mean'],params['thigh_hw']),(params['leg_mean'],params['leg_hw']),(params['foot_mean'],params['foot_hw'])])
    return bounds


def plot_results(log_paths, timesteps, xaxis, task_name):
    dfs = []
    for log_path in log_paths:
        if os.path.isdir(log_path):  # Check if the path is a directory
            # If the path is a directory, assume it contains monitor files and get the list of files inside
            monitor_files = [os.path.join(log_path, file) for file in os.listdir(log_path) if file.endswith('.csv')]
            dfs.extend([pd.read_csv(file, skiprows=1) for file in monitor_files])
        else:
            dfs.append(pd.read_csv(log_path, skiprows=1))  # Skip the header row
    
    plt.figure(figsize=(8, 6))
    for i, df in enumerate(dfs):
        plt.plot(df[xaxis], df['r'])

    plt.xlabel(xaxis)
    plt.ylabel('Reward')
    plt.title(f'{task_name} - Training Progress')
    plt.legend()
    plt.grid()
    plt.show()

def main():

    params = {
			'thigh_mean': 3.92699082,
			'leg_mean': 2.71433605,
			'foot_mean': 5.0893801,
			'hw': 0.5,
			'thigh_hw': 0.5,
			'leg_hw': 0.5,
			'foot_hw': 1.5
			}
	
    bounds=compute_bounds(params)

    source_env = gym.make('CustomHopperUDR-source-v0',bounds=bounds)
    target_env=gym.make('CustomHopperUDR-target-v0',bounds=bounds)

    #source_env = make_vec_env('CustomHopper-source-v0', n_envs=N_ENVS, vec_env_cls=DummyVecEnv, monitor_dir=args.source_log_path)
    #target_env = make_vec_env('CustomHopper-target-v0', n_envs=N_ENVS, vec_env_cls=DummyVecEnv, monitor_dir=args.target_log_path)

    #print('State space:', train_env.observation_space)  # state-space
    #print('Action space:', train_env.action_space)  # action-space
    #print('Dynamics parameters:', target_env.get_parameters())  # masses of each link of the Hopper

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    
    stop_callback = StopTrainingOnMaxEpisodes(max_episodes=MAX_EPS, verbose=1) # callback for stopping at 100_000 episodes
    # When using multiple training environments, agent will be evaluated every
    # eval_freq calls to train_env.step(), thus it will be evaluated every
    # (eval_freq * n_envs) training steps. See EvalCallback doc for more information.
    callback_list = [stop_callback]

    if args.train == 'source':
        source_env=Monitor(source_env,args.source_log_path,allow_early_resets=True)
        train_env = source_env # sets the train to source
        source_eval_callback = EvalCallback(eval_env=source_env, n_eval_episodes=50, eval_freq=5000, log_path=args.source_log_path) # Create callback that also evaluates agent for 50 episodes every 15000 source environment steps.
        callback_list.append(source_eval_callback)
    else:
        target_env=Monitor(target_env,args.target_log_path)
        train_env = target_env
        target_eval_callback = EvalCallback(eval_env=target_env, n_eval_episodes=50, eval_freq=5000, log_path=args.target_log_path) # Create callback that evaluates agent for 50 episodes every 15000 training environment steps.
        callback_list.append(target_eval_callback)

    callback = CallbackList(callback_list)

    train_env.dr=True
    model = SAC('MlpPolicy', batch_size=128, learning_rate=0.00025, env=train_env, verbose=1, device='cpu')
    model.learn(total_timesteps=int(5e3), callback=callback, tb_log_name=args.train)
    model.save("SAC_model_UDR"+args.train)
    train_env.dr=False

    # Plot the results
    log_path = args.target_log_path if args.train == 'target' else args.source_log_path
    plot_results([log_path], 5e3, 't', "SAC CustomHopper")

    model = SAC.load("SAC_model_"+args.train)

    if args.test == 'source':
        mean_reward, std_reward = evaluate_policy(model, source_env, n_eval_episodes=args.evalepisodes, render=True)
        print(mean_reward,std_reward)
    else: 
        mean_reward, std_reward = evaluate_policy(model, target_env, n_eval_episodes=args.evalepisodes, render=True)
        print(mean_reward,std_reward)


if __name__ == '__main__':
    main()