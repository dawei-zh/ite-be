import pickle

from imaginary_time_optimization.maxcut import MaxCut

with open('graphu3R', 'rb') as file:
    graph = pickle.load(file)

node = '6'
graph6 = graph[node]
maxcut_problems = []

for i in range(len(graph6)):
    maxcut = MaxCut(graph6[i])
    print('----------------------------------------------------------', flush=True)
    print(f'Solving MaxCut for {i}-th graph with {maxcut.num_qubits} nodes with brute force...', flush=True)
    _ = maxcut.brute_force()
    print(f'Brute force solution: {maxcut.brute_force_solution}', flush=True)
    print('----------------------------------------------------------', flush=True)
    maxcut_problems.append(maxcut)
    with open('maxcut_N6', 'wb') as file:
        pickle.dump(maxcut_problems, file)

