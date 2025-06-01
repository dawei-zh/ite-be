import numpy as np
import pickle
import sys
# Add the parent directory to sys.path
sys.path.insert(0, '/home1/daweiz/newrbm')

node = '8'
start_from = 0
maxcut_name = 'bipartite_N' + node
approx_name = 'qaoarbm_appro_N' + node + '_p4681012' + '_' + str(start_from)
overlap_name = 'qaoarbm_overlap_N' + node + '_p4681012' + '_' + str(start_from)
postselect_name = 'qaoarbm_postselect_N' + node + '_p4681012' + '_' + str(start_from)
result_name = 'qaoarbm_result_N' + node + '_p4681012' + '_' + str(start_from)

with open(maxcut_name, 'rb') as file:
    graph = pickle.load(file)

approx_data = {}
overlap_data = {}
postselect_data = {}
result_data = {}
times = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
levels = [4, 6, 8, 10, 12]

assert int(node) == graph[start_from].num_qubits, "Assigned node number does not match the graph."

print(f'Solving MaxCut via QAOA+RBM for N = {node} starting from {start_from}-th graph', flush=True)

for i in range(start_from, len(graph)):
    assert i >= start_from, "Starts from a wrong graph index."
    tag = f'graph_{i}'
    tmp_approx = []
    tmp_overlap = []
    tmp_postselect = []
    tmp_result = []
    for p in levels:
        for tau in times:
            print(f'Solving MaxCut via QAOA+RBM for {i}-th graph with time step {tau} and level {p} with sk parameters...', flush=True)
            maxcut = graph[i]
            qaoarbm_res = maxcut.qaoa_rbm(tau = tau, 
                                          qaoa_level = p, 
                                          qaoa_initial = 'sk', 
                                          qaoa_pretrain = False, 
                                          repeat = 10, 
                                          verbose = 1)
            tmp_approx.append((p, tau, qaoarbm_res.approx_ratio))
            tmp_overlap.append((p, tau, qaoarbm_res.overlap))
            tmp_postselect.append((p, tau, qaoarbm_res.postselect_rate))
            tmp_result.append((p, tau, qaoarbm_res._results))

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