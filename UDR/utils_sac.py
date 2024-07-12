from udr_env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
import pickle

def save_rewards(filename, algo_name, rewards):
    
    with open(filename, 'a') as file:
        file.write(f"Name of algo: {algo_name}\n")
        file.write(f"{rewards}\n")
        file.write("\n")  # Add a newline for readability

def compute_bounds(params):
    bounds = list((m-hw,m+hw) for m,hw in [(params['thigh_mean'],params['thigh_hw']),(params['leg_mean'],params['leg_hw']),(params['foot_mean'],params['foot_hw'])])
    return bounds

with open('UDR/best_udr_tuning_result.pkl', 'rb') as infile:
    params = pickle.load(infile)[0] 
    
bounds=compute_bounds(params)
train_env = gym.make('CustomHopperUDR-source-v0',bounds=bounds)
train_env.set_dr_training(True)
test_env=gym.make('CustomHopperUDR-source-v0',bounds=bounds)

def train(args):
    model = SAC('MlpPolicy', env=train_env, device='cpu', **args, verbose=0) 
    model.learn(total_timesteps=5e4)
    return model

def test(model):
    mean, _ = evaluate_policy(model, test_env, n_eval_episodes=100)
    return mean
