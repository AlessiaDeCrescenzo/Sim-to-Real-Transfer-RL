# main_tuning.py

# main_tuning.py

import itertools
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from utils_tuning import train, test, set_seed
import torch

params = {
    'lr': [5e-2, 1e-3, 5e-4],
    'gamma': [0.99, 0.999],
    'hidden': [32, 64, 128]
}

MULTIPLE_STARTS = 4
seed_env= 316683  # Fixed seed for reproducibility
seed_test=316683


def pool_tt(args):
    agent = train(seed=seed_env,**args)
    return test(agent,seed=seed_test)

results = []
set_seed(seed_env)

keys = list(params.keys())
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
with open('REINFORCE/REINFORCEvpg/result_REINFORCE.pkl', 'wb') as outfile:
    pickle.dump(obj=best_result, file=outfile)