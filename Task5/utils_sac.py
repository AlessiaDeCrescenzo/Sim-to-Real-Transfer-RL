from env.custom_hopper import *
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

def save_rewards(filename, algo_name, rewards):
    
    with open(filename, 'a') as file:
        file.write(f"Name of algo: {algo_name}\n")
        file.write(f"{rewards}\n")
        file.write("\n")  # Add a newline for readability

train_env = make_vec_env('CustomHopper-source-v0', n_envs=12, vec_env_cls=DummyVecEnv)
test_env = gym.make('CustomHopper-source-v0')

def train(args):
    model = SAC('MlpPolicy', env=train_env, device='cpu', **args, verbose=0,seed=315304) 
    model.learn(total_timesteps=5e4)
    return model

def test(model):
    mean, _ = evaluate_policy(model, test_env, n_eval_episodes=250)
    return mean
