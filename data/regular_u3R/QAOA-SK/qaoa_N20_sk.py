import pickle

node = '20'
maxcut_name = 'maxcut_N' + node
approx_name = 'qaoa_sk_appro_N' + node
overlap_name = 'qaoa_sk_overlap_N' + node

with open(maxcut_name, 'rb') as file:
    graph = pickle.load(file)

approx_data = {}
overlap_data = {}
levels = [3, 4, 5, 6, 7, 8, 9, 10]
start_from = 0
assert int(node) == graph[start_from].num_qubits, "Assigned node number does not match the graph."

print(f'Solving MaxCut via QAOA SK model for N = {node} starting from {start_from}-th graph', flush=True)

for i in range(start_from, len(graph)):
    assert i >= start_from, "Starts from a wrong graph index."
    tag = f'graph_{i}'
    tmp_approx = []
    tmp_overlap = []
    for p in levels:
        print(f'Solving MaxCut via QAOA for {i}-th graph with level {p}...', flush=True)
        maxcut = graph[i]
        qaoa_res = maxcut.qaoa(level = p, 
                               initial = 'sk', 
                               repeat = 10, 
                               pretrain = True)
        tmp_approx.append(qaoa_res.approx_ratio)
        tmp_overlap.append(qaoa_res.overlap)

        approx_data[tag] = tmp_approx
        overlap_data[tag] = tmp_overlap
        with open(approx_name, 'wb') as file:
            pickle.dump(approx_data, file)
        with open(overlap_name, 'wb') as file:
            pickle.dump(overlap_data, file)