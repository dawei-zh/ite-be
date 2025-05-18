import networkx as nx
import numpy as np
import imaginary_time_optimization.utils.rbm as rbm
import imaginary_time_optimization.utils.newrbm as newrbm
import imaginary_time_optimization.utils.qaoa as qaoa
import qiskit.quantum_info as qi

class MaxCut:
    def __init__(self, 
                 graph: nx.Graph):
        """
        MaxCut problem for undirect graph
        """
        self.num_qubits = len(graph.nodes)
        self.graph = graph
        self._get_hamiltonian()

        self.brute_force_solution = None
        self.wavefunction_solution = None

    def _get_hamiltonian(self):

        ham = []
        ham_2 = []
        ham_str = ''

        for i in range(len(self.graph.edges)):
            
            pos1, pos2, weight = list(self.graph.edges(data="weight"))[i]

            if weight == None:
                weight = 1
            assert weight != 0, "Weight cannot be zero!"

            # Get Hamiltonian used for future methods
            # Numbering follows an ascending order of first qubit
            if pos1 < pos2:
                ham.append((pos1, pos2, weight))
            else:
                ham.append((pos2, pos1, weight))

            # Get Hamiltonian string
            _weight = str(np.round(abs(weight), decimals = 2))
            if _weight == '1':
                tmp_str = 'Z' + str(pos1) + 'Z' + str(pos2) 
            else:
                tmp_str = _weight + 'Z' + str(pos1) + 'Z' + str(pos2)
            if weight < 0:
                ham_str += '-' + tmp_str
            else:
                ham_str += '+' + tmp_str

            # Get Hamiltonian in qiskit form
            tmp = ['I'] * self.num_qubits
            tmp[pos1] = 'Z'
            tmp[pos2] = 'Z'
            ham_2.append((''.join(tmp), weight))
        
        self._hamiltonian = ham
        self.hamiltonian = ham_str
        self._hamiltonian_qiskit = qi.SparsePauliOp.from_list(ham_2)

        return 0

    def brute_force(self) -> list:
        """
        Obtain ideal solutions by brute force searching for strings with maximum cost
        """
        # Obtain all possible binary strings for the given number of qubits
        strings = [format(i, f'0{self.num_qubits}b') for i in range(2 ** self.num_qubits)] 
        solutions = {}

        # Compute cost for each binary string
        for solution in strings:
            cost = 0
            for i, j, _ in self._hamiltonian:
                if solution[i] != solution[j]:
                    cost += 1
            
            solutions[solution] = cost

        # Search for the string with the maximum cost
        max_cost = max(solutions.values())
        self.brute_force_solution = [key for key, value in solutions.items() if value == max_cost]
        return self.brute_force_solution
    
    def wavefunction(self) -> list:
        """
        Obtain ideal solution by directly solving the imaginary time evolution of the Hamiltonian
        """
        eigenvalues, eigenvectors = np.linalg.eig(self._hamiltonian_qiskit)
        min_eigenvalue = np.min(eigenvalues)

        # Find all eigenvectors with the minimum eigenvalue
        min_indices = np.where(eigenvalues == min_eigenvalue)[0]
        min_eigenvectors = []
        for min_ind in min_indices:
            min_eigenvectors.append(eigenvectors[:, min_ind])
        
        # Convert column vectors to binary strings
        # All elements of eigenvector are either 0 or 1
        # The binary string is the binary representation of 
        # position of the only "1" in the eigenvector
        result = []
        for vector in min_eigenvectors:
            # Not sure whether the around is required
            tmp = ''.join(['0' if int(np.around(i.real)) == 0 else '1' for i in vector]) 
            
            result.append(bin(tmp.find('1'))[2:].zfill(self.num_qubits))
        
        self.wavefunction_solution = result
        return result

    def rbm(self, 
            tau: float, 
            nshot: int = 100000, 
            repeat: int = 1,
            verbose: int = 0):
        """
        The solution of MaxCut problem using RBM method
        
        For MaxCut problem, each term in the Hamiltonian commutes any other terms. 
        Hence, we have no Trotter error so the step is 1 and time can be arbitrary. 
        """
        assert self.brute_force_solution != None, "Please run brute_force() first to get the ideal solution or assign the ideal solution!"
        return rbm.RBMResult(tau = tau, 
                             num_qubits = self.num_qubits,
                             hamiltonian = self._hamiltonian, 
                             ideal_solution = self.brute_force_solution, 
                             nshot = nshot, 
                             repeat = repeat, 
                             verbose = verbose)
        
    def new_rbm(self, 
                tau: float, 
                nshot: int = 100000, 
                repeat: int = 1,
                verbose: int = 0):
        
        assert self.brute_force_solution != None, "Please run brute_force() first to get the ideal solution or assign the ideal solution!"
        return newrbm.NewRBMResult(tau = tau, 
                                   num_qubits = self.num_qubits,
                                   hamiltonian = self._hamiltonian, 
                                   ideal_solution = self.brute_force_solution, 
                                   nshot = nshot, 
                                   repeat = repeat, 
                                   verbose = verbose)
        
    def qaoa(self, 
             level: int, 
             initial: str = None, 
             training_attempt: int = 10,
             train_iter: int = 1000,
             optimizer: str = 'COBYLA',
             repeat: int = 1,
             nshot: int = 20000,
             pretrain: bool = False, 
             verbose: int = 0):
        
        assert self.brute_force_solution != None, "Please run brute_force() first to get the ideal solution or assign the ideal solution!"
        return qaoa.QAOAResult(level = level, 
                               initial = initial,
                               num_qubits = self.num_qubits,
                               hamiltonian = self._hamiltonian,
                               ideal_solution = self.brute_force_solution,
                               nshot = nshot, 
                               repeat = repeat,
                               training_attempt = training_attempt,
                               train_iter = train_iter, 
                               optimizer = optimizer,
                               pretrain = pretrain,
                               verbose = verbose)

    def qaoa_rbm(self, 
                tau: float,
                qaoa_level: int,
                qaoa_pretrain: bool,
                qaoa_optimizer: str = 'COBYLA',
                qaoa_train_iter: int = 0, 
                qaoa_initial: str = None,
                qaoa_pretrain_repeat: int = 1,
                nshot: int = 100000, 
                repeat: int = 1,
                verbose: int = 0):
        
        assert self.brute_force_solution != None, "Please run brute_force() first to get the ideal solution or assign the ideal solution!"
        return rbm.RBMResult(tau = tau, 
                             num_qubits = self.num_qubits,
                             hamiltonian = self._hamiltonian, 
                             ideal_solution = self.brute_force_solution, 
                             nshot = nshot, 
                             repeat = repeat, 
                             qaoa = True, 
                             qaoa_level = qaoa_level, 
                             qaoa_initial = qaoa_initial, 
                             qaoa_pretrain = qaoa_pretrain, 
                             qaoa_pretrain_repeat = qaoa_pretrain_repeat,
                             qaoa_optimizer = qaoa_optimizer,
                             qaoa_train_iter = qaoa_train_iter, 
                             verbose = verbose)