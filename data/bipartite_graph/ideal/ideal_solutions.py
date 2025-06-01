import pickle
from imaginary_time_optimization.maxcut import MaxCut

with open('u3R_bipartite', 'rb') as file:
    graph_all = pickle.load(file)

nodes = ['6', '8', '10', '12', '14', '16', '18', '20']
for node in nodes:
    graph = graph_all[node]
    maxcut_problems = []
    maxcut_name = 'bipartite_N' + node

    for i in range(len(graph)):
        maxcut = MaxCut(graph[i])
        print('----------------------------------------------------------', flush=True)
        print(f'Solving MaxCut for {i}-th graph with {maxcut.num_qubits} nodes with brute force...', flush=True)
        _ = maxcut.brute_force()
        print(f'Brute force solution: {maxcut.brute_force_solution}', flush=True)
        print('----------------------------------------------------------', flush=True)
        maxcut_problems.append(maxcut)
        with open(maxcut_name, 'wb') as file:
            pickle.dump(maxcut_problems, file)

