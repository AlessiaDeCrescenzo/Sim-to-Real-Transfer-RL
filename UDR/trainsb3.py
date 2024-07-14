"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
import pickle
from udr_env.custom_hopper import *
from udr_env.Wrapper import TrackRewardWrapper
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
from utils_sac import save_rewards

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--episodes', default=100_000, type=int)
    parser.add_argument('--evalepisodes',default=250,type=int)
    parser.add_argument('--fine_tuning_parameters', default='UDR/result_SAC.pkl', type=str, help='Path to fine-tuning parameters')
    parser.add_argument('--seed',default=0,type=int, help='Seed')
    return parser.parse_args()
    

args = parse_args()

if args.train is None or args.test is None:
    exit('Arguments required')

MAX_EPS = args.episodes

def compute_bounds(params):
    bounds = list((m-hw,m+hw) for m,hw in [(params['thigh_mean'],params['thigh_hw']),(params['leg_mean'],params['leg_hw']),(params['foot_mean'],params['foot_hw'])])
    return bounds

def main():

    with open('UDR/best_udr_tuning_result.pkl', 'rb') as infile:
        params = pickle.load(infile)[0] 
	
    print(params)

    bounds=compute_bounds(params)

    source_env = gym.make('CustomHopperUDR-source-v0',bounds=bounds)
    target_env=gym.make('CustomHopperUDR-target-v0',bounds=bounds)


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
        source_env=TrackRewardWrapper(source_env)
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

    train_env.set_dr_training(True)
    model = SAC('MlpPolicy', batch_size=fine_tuning_params['batch_size'], learning_rate=fine_tuning_params['learning_rate'], env=train_env, verbose=1, device='cpu',seed=args.seed)
    model.learn(total_timesteps=int(3e5), callback=callback, tb_log_name=args.train)
    model.save("SAC_model_UDR"+args.train)
    train_env.set_dr_training(False)

    model = SAC.load("SAC_model_UDR"+args.train)

    source_env = gym.make('CustomHopperUDR-source-v0',bounds=bounds)
    target_env = gym.make('CustomHopperUDR-target-v0',bounds=bounds)
    if args.test == 'source':
        eval_env= TrackRewardWrapper(source_env)
        eval_env.set_dr_training(False)
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=args.evalepisodes, render=True)
        print(mean_reward,std_reward)
    else: 
        eval_env= TrackRewardWrapper(target_env)
        eval_env.set_dr_training(False)
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=args.evalepisodes, render=True)
        print(mean_reward,std_reward)

    if args.train == 'source':
        save_rewards('SAC_UDR.txt','SAC_'+args.train, train_env.succ_metric_buffer)
        save_rewards('SAC_UDR.txt','SAC_rewards'+args.train, train_env.buffer)
    else:
        save_rewards('SAC_UDR_target.txt','SAC_'+args.train, train_env.succ_metric_buffer)
        save_rewards('SAC_UDR_target.txt','SAC_rewards'+args.train, train_env.buffer)
    
    save_rewards('SAC_test_UDR.txt','SAC_UDR_'+args.train+'_'+args.test,eval_env.succ_metric_buffer)


if __name__ == '__main__':
    main()