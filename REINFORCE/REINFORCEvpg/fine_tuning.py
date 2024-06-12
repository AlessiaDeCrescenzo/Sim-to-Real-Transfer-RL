# fine_tuning.py

import pickle
from multiprocessing import Pool
from tqdm import tqdm
from utils_tuning import train, test

# Set the number of episodes for fine-tuning
FINE_TUNING_TRAIN_EPISODES = 300
FINE_TUNING_TEST_EPISODES = 50


MULTIPLE_STARTS = 4

def pool_tt(args):
    train_episodes = args.pop('train_episodes', 1000)
    test_episodes = args.pop('test_episodes', 50)
    agent = train(**args)
    return test(agent)

# Load top 6 configurations
with open('top_6_configs.pkl', 'rb') as infile:
    top_6_results = pickle.load(infile)

fine_tuning_results = []

# Perform fine-tuning on top 6 configurations
for score, config in tqdm(top_6_results):
    config['train_episodes'] = FINE_TUNING_TRAIN_EPISODES
    config['test_episodes'] = FINE_TUNING_TEST_EPISODES
    pool = Pool(processes=MULTIPLE_STARTS)
    scores = pool.map(pool_tt, [config]*MULTIPLE_STARTS)
    score = sum(scores) / len(scores)
    fine_tuning_results.append([score, config])

# Sort and get the best fine-tuning result
fine_tuning_results.sort(reverse=True, key=lambda x: x[0])
best_result = fine_tuning_results[0]

# Print the best fine-tuning result
print("Best fine-tuning result:")
print(f"Score: {best_result[0]}, Config: {best_result[1]}")

# Save the best fine-tuning result
with open('best_fine_tuning_result.pkl', 'wb') as outfile:
    pickle.dump(obj=best_result, file=outfile)