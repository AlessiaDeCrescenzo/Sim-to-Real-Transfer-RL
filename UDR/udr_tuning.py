# Pretty prints
from pprint import pprint

# Policy
from udr_env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
# Standard import
import numpy as np
import gym
import argparse
import os
from tqdm import tqdm
import time

# Hyperparameters optimization import
from sklearn.model_selection import ParameterGrid
import pickle

import itertools


# Save info
history = []
dump_counter = 0

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env-train', default='CustomHopperUDR-source-v0', type=str, help='Train environment')
	parser.add_argument('--env-test', default='CustomHopperUDR-source-v0', type=str, help='Test environment')
	parser.add_argument('--device', default='cpu', type=str, help='Device [cpu, cuda]')
	parser.add_argument('--train-steps', default=5e4, type=int, help='Train timesteps for a single policy')
	parser.add_argument('--verbose', default='False', type=bool, help='Print hyperparameters and mean return at each iteration')
	return parser.parse_args()

args = parse_args()
args.verbose = (args.verbose == 'True')

DEVICE = args.device
VERBOSE = args.verbose  # print hyperparameters and mean return at each iteration
TRAIN_STEPS = args.train_steps  
OUT_OF_BOUNDS_RETURN = "OUT of BOUNDS"
ENV_TRAINING = args.env_train  
ENV_TESTING = args.env_test

LEARNING_RATE = 1e-3
BATCH_SIZE = 128


def gridify(parameters_dict: dict) -> list:
	return list(ParameterGrid(parameters_dict))

def compute_bounds(params):
	bounds = list((m-hw, m+hw) for m, hw in [(params['thigh_mean'], params['thigh_hw']), (params['leg_mean'], params['leg_hw']), (params['foot_mean'], params['foot_hw'])])
	if VERBOSE:
		print(f"Masses bounds: {bounds}")
	return bounds

def _create_source_env(bounds):
	source_env = gym.make(ENV_TRAINING)
	source_env.bounds=bounds
	source_env.set_dr_training()
	return source_env

def _train(params: dict, env) -> None:
	"""
	params:         hyperparameters for training
	env:            train environment"""

	model = SAC("MlpPolicy", env, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, device=DEVICE,seed=315304)
	start_time = time.time()
	model.learn(total_timesteps=TRAIN_STEPS)
	
	finish_time = time.time()
	return model

def train_and_test(params: dict) -> dict:
	global dump_counter
	if VERBOSE:
		pprint(params)
	
	dumpv = {
		'history': history
	}
	#with open(f'opt_{dump_counter}.log', 'wb') as outf:
	#	pickle.dump(obj=dumpv, file=outf)
	#print(dump_counter)
	dump_counter += 1

	print('\n start')
	# Check hyperparameters out of bounds
	if any(x < 0 for x in params.values()):# or params['learning_rate'] > 1 or params['ent_coef'] > 1:
		return OUT_OF_BOUNDS_RETURN

	bounds = compute_bounds(params) # Compute bounds
	
	if any(x[0] < 0 for x in bounds): # Check masses not to be less than zero
		return OUT_OF_BOUNDS_RETURN

	source_env = _create_source_env(bounds)
	target_env = gym.make(ENV_TESTING)

	model = _train(params, source_env) 
	mean, std_dev = evaluate_policy(model, target_env, n_eval_episodes=50, deterministic=True)
	print('end test')
	if VERBOSE:
		pprint(mean)

	source_env.close()
	target_env.close()
	history.append((params, - mean))  # negative because the method is made for minimizing
	
	return {'loss': -mean, 'status': True}


##################################################################
def main():
	b0 = 3.92699082
	b1 = 2.71433605
	b2 = 5.08938010

	default_params = {
		'thigh_mean': b0,
		'leg_mean': b1,
		'foot_mean': b2,
	}

	space = {
		'thigh_hw': [0.5, 1],
        'leg_hw': [0.5, 1],
        'foot_hw':[0.5, 1]
	}

	keys = list(space.keys())
	for p in tqdm(itertools.product(*space.values())):
		kw = dict(zip(keys, p))
		for k, v in default_params.items():
			kw[k] = v 
		for bp in ['thigh', 'leg', 'foot']:
			kw[f'{bp}_hw'] = kw[f'{bp}_hw']
		train_and_test(kw)


	print('len(history):', len(history))
	best = min(history, key=lambda tpl: tpl[1])

	print(f"Best results: (the score is negated if the value has to be maximized): \n {best}")

	dumpv = {
		'history': history
	}

	# Save the best fine-tuning result
	with open('UDR/best_udr_tuning_result.pkl', 'wb') as outfile:
		pickle.dump(obj=best, file=outfile)

if __name__ == '__main__':
	main()