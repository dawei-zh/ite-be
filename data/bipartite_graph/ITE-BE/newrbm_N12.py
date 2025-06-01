import numpy as np
import pickle

node = '12'
start_from = 0
maxcut_name = 'bipartite_N' + node
approx_name = 'newrbm_appro_N' + node + '_' + str(start_from)
overlap_name = 'newrbm_overlap_N' + node + '_' + str(start_from)
postselect_name = 'newrbm_postselect_N' + node + '_' + str(start_from)
result_name = 'newrbm_result_N' + node + '_' + str(start_from)

with open(maxcut_name, 'rb') as file:
    graph = pickle.load(file)

approx_data = {}
overlap_data = {}
postselect_data = {}
result_data = {}
times = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

assert int(node) == graph[start_from].num_qubits, "Assigned node number does not match the graph."

print(f'Solving MaxCut via newRBM for N = {node} starting from {start_from}-th graph', flush=True)

for i in range(start_from, len(graph)):
    assert i >= start_from, "Starts from a wrong graph index."
    tag = f'graph_{i}'
    tmp_approx = []
    tmp_overlap = []
    tmp_postselect = []
    tmp_result = []
    for tau in times:
        print(f'Solving MaxCut via newRBM for {i}-th graph with time {tau}...', flush=True)
        maxcut = graph[i]
        newrbm_res = maxcut.new_rbm(tau = tau, repeat = 10, verbose = 1)
        tmp_approx.append(newrbm_res.approx_ratio)
        tmp_overlap.append(newrbm_res.overlap)
        tmp_postselect.append(newrbm_res.postselect_rate)
        tmp_result.append(newrbm_res._results)

        approx_data[tag] = tmp_approx
        overlap_data[tag] = tmp_overlap
        postselect_data[tag] = tmp_postselect
        result_data[tag] = tmp_result
        with open(approx_name, 'wb') as file:
            pickle.dump(approx_data, file)
        with open(overlap_name, 'wb') as file:
            pickle.dump(overlap_data, file)
        with open(postselect_name, 'wb') as file:
            pickle.dump(postselect_data, file)
        with open(result_name, 'wb') as file:
            pickle.dump(result_data, file)
