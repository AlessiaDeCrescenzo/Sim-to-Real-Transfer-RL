import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default=None, type=str, help='Path file')    #default=100000
    return parser.parse_args()


def plot_rewards(file_path):

    algo_names = []
    rewards_lists = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        current_algo_name = None
        current_rewards = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("Name of algo:"):
                if current_algo_name is not None:
                    algo_names.append(current_algo_name)
                    rewards_lists.append(current_rewards)
                current_algo_name = line.replace("Name of algo:", "").strip()
                current_rewards = []
            elif line.startswith("[") and line.endswith("]"):
                current_rewards = eval(line)
        
        if current_algo_name is not None:
            algo_names.append(current_algo_name)
            rewards_lists.append(current_rewards)
    
    # Plotting
    for i, rewards in enumerate(rewards_lists):
        plt.plot(rewards, label=algo_names[i])
    
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('Moving average of test results')
    plt.legend()
    plt.show()

def extract_list_from_file(file_path, algo_name):

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            # Check if the line contains the algorithm name
            if algo_name in lines[i]:
                # The next line contains the list
                values_list = eval(lines[i + 1].strip())
                return values_list
    return []

def plot_results(file1, algo_name1, file2, algo_name2):

    list1 = extract_list_from_file(file1, algo_name1)
    list2 = extract_list_from_file(file2, algo_name2)
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(list1, label='Rewards')
    plt.plot(list2, label='500-episodes moving average')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Results  training with UDR')
    plt.legend()
    plt.show()


#plot_results('SAC_UDR.txt', 'SAC_rewardssource','SAC_UDR.txt', 'SAC_source')

plot_rewards('SAC_test_UDR.txt')