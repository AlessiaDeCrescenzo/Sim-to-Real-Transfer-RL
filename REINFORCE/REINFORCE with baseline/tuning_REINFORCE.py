# main_tuning.py

import itertools
import pickle
from multiprocessing import Pool
from tqdm import tqdm
from utils_tuning import train, test

params = {
    'lr': [1e-3, 1e-4, 1e-5],
    'gamma': [0.95, 0.99, 0.999],
    'baseline': [0, 10, 20]
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

print(max(results))

with open('log_pickle', 'wb') as outfile:
    pickle.dump(obj=results, file=outfile)
