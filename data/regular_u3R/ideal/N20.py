import pickle

from imaginary_time_optimization.maxcut import MaxCut

with open('graphu3R', 'rb') as file:
    graph = pickle.load(file)

node = '20'
graph20 = graph[node]
maxcut_problems = []

for i in range(len(graph20)):
    maxcut = MaxCut(graph20[i])
    print('----------------------------------------------------------', flush=True)
    print(f'Solving MaxCut for {i}-th graph with {maxcut.num_qubits} nodes with brute force...', flush=True)
    _ = maxcut.brute_force()
    print(f'Brute force solution: {maxcut.brute_force_solution}', flush=True)
    print('----------------------------------------------------------', flush=True)
    maxcut_problems.append(maxcut)
    with open('maxcut_N20', 'wb') as file:
        pickle.dump(maxcut_problems, file)

