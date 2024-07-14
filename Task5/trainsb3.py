"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
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
import pickle
from env.Wrapper import TrackRewardWrapper
from utils_sac import save_rewards


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--episodes', default=100_000, type=int)
    parser.add_argument('--evalepisodes',default=250,type=int)
    parser.add_argument('--fine_tuning_parameters',default='Task5/result_SAC.pkl',type=str,help='Hyperparameters file path')
    parser.add_argument('--seed',default=0,type=int,help='Seed')
    return parser.parse_args()

args = parse_args()

if args.train is None or args.test is None:
    exit('Arguments required')


MAX_EPS = args.episodes

def main():
    source_env = gym.make('CustomHopper-source-v0')
    target_env=gym.make('CustomHopper-target-v0')

    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    
    stop_callback = StopTrainingOnMaxEpisodes(max_episodes=MAX_EPS, verbose=1) # callback for stopping at 100_000 episodes
    # When using multiple training environments, agent will be evaluated every
    # eval_freq calls to train_env.step(), thus it will be evaluated every
    # (eval_freq * n_envs) training steps. See EvalCallback doc for more information.
    callback_list = [stop_callback]

    if args.train == 'source':
        source_env = TrackRewardWrapper(source_env)
        #source_env=Monitor(source_env,args.source_log_path,allow_early_resets=True)
        train_env = source_env # sets the train to source
        source_eval_callback = EvalCallback(eval_env=source_env, n_eval_episodes=50, eval_freq=10000) # Create callback that also evaluates agent for 50 episodes every 15000 source environment steps.
        callback_list.append(source_eval_callback)
    else:
        target_env=TrackRewardWrapper(target_env)
        train_env = target_env
        target_eval_callback = EvalCallback(eval_env=target_env, n_eval_episodes=50, eval_freq=10000) # Create callback that evaluates agent for 50 episodes every 15000 training environment steps.
        callback_list.append(target_eval_callback)

    callback = CallbackList(callback_list)

    with open(args.fine_tuning_parameters, 'rb') as infile:
        fine_tuning_params = pickle.load(infile)[1]  # [1] because you only need the config, not the score

    model = SAC('MlpPolicy', batch_size=fine_tuning_params['batch_size'], learning_rate=fine_tuning_params['learning_rate'], env=train_env, verbose=1, device='cpu',seed=args.seed)
    model.learn(total_timesteps=int(3e5), callback=callback, tb_log_name=args.train)
    model.save("SAC_model_"+args.train)

    # Plot the results
    #log_path = args.target_log_path if args.train == 'target' else args.source_log_path

    model = SAC.load("SAC_model_"+args.train)

    if args.test == 'source':
        test_env= gym.make('CustomHopper-source-v0')
        test_env=TrackRewardWrapper(test_env)
    else: 
        test_env= gym.make('CustomHopper-target-v0')
        test_env=TrackRewardWrapper(test_env)

    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=args.evalepisodes, render=False)
    print(mean_reward,std_reward)

    if args.train == 'source':
        save_rewards('SAC.txt','SAC_'+args.train, train_env.succ_metric_buffer)
        save_rewards('SAC.txt','SAC_rewards'+args.train, train_env.buffer)
    else:
        save_rewards('SAC_target.txt','SAC_'+args.train, train_env.succ_metric_buffer)
        save_rewards('SAC_target.txt','SAC_rewards'+args.train, train_env.buffer)
    
    save_rewards('SAC_test_noUDR.txt','SAC_'+args.train+'_'+args.test,test_env.succ_metric_buffer)

if __name__ == '__main__':
    main()
