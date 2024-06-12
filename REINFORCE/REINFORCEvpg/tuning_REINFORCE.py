# main_tuning.py

import itertools
import pickle
from multiprocessing import Pool
from tqdm import tqdm
from utils_tuning import train, test

params = {
    'lr': [5e-2, 1e-3, 5e-4],
    'gamma': [0.99, 0.999],
    'hidden': [32, 64, 128]
}

MULTIPLE_STARTS = 4

def pool_tt(args):
    agent = train(**args)
    return test(agent)

results = []

keys = list(params.keys())
for p in tqdm(itertools.product(*params.values())):
    kw = dict(zip(keys, p))
    pool = Pool(processes=MULTIPLE_STARTS)
    scores = pool.map(pool_tt, [kw]*MULTIPLE_STARTS)
    score = sum(scores) / len(scores)
    results.append([score, kw])

# Sort the results by score in descending order
results.sort(reverse=True, key=lambda x: x[0])

# Print the top 6 configurations
print("Top 6 configurations:")
for i in range(min(6, len(results))):
    print(f"Rank {i+1}: Score = {results[i][0]}, Config = {results[i][1]}")

#save results
with open('top_6_configs.pkl', 'wb') as outfile:
    pickle.dump(obj=results, file=outfile)
