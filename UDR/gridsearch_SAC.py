import itertools
import os
import pickle
from multiprocessing import Pool
from tqdm import tqdm
from pprint import pprint
import pickle

# Choose one:
from utils_sac import train, test


# SAC:
params = {
	'learning_rate': [3e-4, 2.5e-4, 1e-3],
	'batch_size': [64, 128]
}


MULTIPLE_STARTS = 4

def pool_tt(args: dict):
	model = train(args)
	return test(model)

	
results = []

keys = list(params.keys())
for p in tqdm(itertools.product(*params.values())):
	kw = dict(zip(keys, p))
	pprint(kw)
	pool = Pool(processes=MULTIPLE_STARTS)
	scores = pool.map(pool_tt, [kw]*MULTIPLE_STARTS)
	score = sum(scores)/len(scores)
	results.append([score, kw])

    # alternative log
	# np.savez(f'outputs/log_{counter}', results=results)

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
with open('UDR/result_SAC.pkl', 'wb') as outfile:
    pickle.dump(obj=best_result, file=outfile)