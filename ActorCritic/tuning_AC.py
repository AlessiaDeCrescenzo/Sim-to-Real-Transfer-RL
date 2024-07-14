# main_tuning.py

import itertools
import pickle
from multiprocessing import Pool
from tqdm import tqdm
from utils_tuning import train, test, set_seed
import random

# Define the parameters for hyperparameter tuning
params = {
    'lr': [1e-3, 1e-4],
    'gamma': [0.99,0.999],
    'hidden': [64,128]              
}

MULTIPLE_STARTS = 4  # Number of runs for each parameter combination to average out randomness

# Function to train and test the agent with given arguments
def pool_tt(args):
    agent = train(**args) # Train the agent with the given parameters
    return test(agent) # Test the trained agent and return the test score

seed_env=315304

results = []  # List to store the results of all parameter combinations
set_seed(seed_env)

keys = list(params.keys())  # Extract the parameter names
# Iterate over all combinations of parameter values
for p in tqdm(itertools.product(*params.values())):
    kw = dict(zip(keys, p))
    pool = Pool(processes=MULTIPLE_STARTS)
    scores = pool.map(pool_tt, [kw]*MULTIPLE_STARTS)
    score = sum(scores) / len(scores)
    results.append([score, kw])

# Sort and get the best fine-tuning result
results.sort(reverse=True, key=lambda x: x[0])
best_result = results[0]

# Print the top 6 configurations
print("Top 6 configurations:")
for i in range(min(6, len(results))):
   print(f"Rank {i+1}: Score = {results[i][0]}, Config = {results[i][1]}")

# Print the best fine-tuning result
print("Best fine-tuning result:")
print(f"Score: {best_result[0]}, Config: {best_result[1]}")

# Save the best fine-tuning result
with open('ActorCritic/result_AC.pkl', 'wb') as outfile:
    pickle.dump(obj=best_result, file=outfile)
