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
import pickle
from udr_env.Wrapper import TrackRewardWrapper
from utils_sac import save_rewards

model = SAC.load("SAC_model_UDRsource")

test_env= gym.make('CustomHopper-target-v0')
test_env=TrackRewardWrapper(test_env)

mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=250, render=False)
print(mean_reward,std_reward)

save_rewards('SAC_test_UDR.txt','SAC_source_target',test_env.succ_metric_buffer)