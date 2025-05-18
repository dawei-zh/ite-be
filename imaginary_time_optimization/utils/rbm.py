from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np
import imaginary_time_optimization.utils.general as general_utils
import imaginary_time_optimization.utils.qaoa as qaoa_utils

# def _one_body(circ, W_1, s_1, measure):
#     meas1, meas2, meas3 = measure
#     # X_1 interaction
#     circ.reset(3)
#     circ.cx(0, 3)
#     circ.rx(2 * W_1, 0)
#     circ.cx(0, 3)
#     circ.rx(2 * s_1 * W_1, 3)
#     circ.measure(3, meas1)
#     circ.barrier()
#     # X_2 interaction
#     circ.reset(3)
#     circ.cx(1, 3)
#     circ.rx(2 * W_1, 1)
#     circ.cx(1, 3)
#     circ.rx(2 * s_1 * W_1, 3)
#     circ.measure(3, meas2)
#     circ.barrier()
#     # X_3 interaction
#     circ.reset(3)
#     circ.cx(2, 3)
#     circ.rx(2 * W_1, 2)
#     circ.cx(2, 3)
#     circ.rx(2 * s_1 * W_1, 3)
#     circ.measure(3, meas3)
#     circ.barrier()

#     return circ

def _two_body(circ: QuantumCircuit, 
              pos1: int, 
              pos2: int, 
              W: float, 
              s: int,
              meas_bit: int) -> QuantumCircuit:
    
    """Create two body circuits for certain two-qubits Z_{pos1}Z Pauli string"""

    num_qubits = circ.num_qubits
    ancilla = num_qubits - 1

    circ.reset(ancilla) 
    circ.h(ancilla)

    circ.cx(pos1, ancilla)
    circ.rz(2 * W, ancilla)
    circ.cx(pos1, ancilla)
        
    circ.cx(pos2, ancilla)
    circ.rz(2 * s * W, ancilla)
    circ.cx(pos2, ancilla)
    circ.h(ancilla)
    circ.measure(ancilla, meas_bit)
    circ.barrier()

    return circ

def add_rbm_circuit(circ: QuantumCircuit, 
                    hamiltonian: list, 
                    tau: float) -> QuantumCircuit:
    """Only include two body interaction for now."""
    for j in range(len(hamiltonian)):
        pos1, pos2, weight = hamiltonian[j]
        K = weight * tau
        if K > 0:
            s = 1
        else:
            s = -1

        W = np.arccos(np.exp(-2 * K)) / 2
        circ = _two_body(circ, pos1, pos2, W, s, j)

    return circ

class RBMResult:
    def __init__(self, 
                 tau: float,
                 num_qubits: int,
                 hamiltonian: list, 
                 ideal_solution: list,
                 nshot: int, 
                 repeat: int,
                 qaoa = False,
                 qaoa_level: int = None,
                 qaoa_initial: str = None,
                 qaoa_pretrain: bool = True,
                 qaoa_pretrain_repeat: int = 1,
                 qaoa_optimizer: str = None,
                 qaoa_train_iter: int = 0,
                 verbose = 0):
        """
        Customized class for solution of RBM method. 

        Verbose 0: combined and post-selected result
        Verbose 1: not combined but post-selected result
        Verbose 2: not combined and not post-selected result
        """
        
        self.tau = tau
        self.nqbits = num_qubits
        self.hamiltonian = hamiltonian
        self.nshot = nshot
        self.ideal = ideal_solution
        self.ncbits = len(hamiltonian)
        self.qaoa = qaoa
        if self.qaoa == True:
            self.qaoa_level = qaoa_level
            self.initial = qaoa_initial
            self.qaoa_pretrain = qaoa_pretrain
            self.training_attempt = qaoa_pretrain_repeat
            self.optimizer = qaoa_optimizer
            self.train_iter = qaoa_train_iter

        self.circ = self._circuit()

        tmp = []
        _post = []
        _approx = []
        _overlap = []
        for _ in range(repeat):
            if verbose == 2:
                tmp.append(self._execute())
            else:
                _res = general_utils.invert_counts(general_utils.post_select(self._execute(), 
                                                 self.nqbits,
                                                 self.ncbits))
                tmp.append(_res)
                _post.append(sum(_res.values()) / nshot)
                _approx.append(self._approx_ratio(_res))
                _overlap.append(self._overlap(_res))

        if verbose == 0: 
            self._results = general_utils.combine_results(tmp, self.nqbits)
            self.postselect_rate = np.mean(_post)
            self.approx_ratio = np.mean(_approx)
            self.overlap = np.mean(_overlap)
        elif verbose == 1:
            self._results = tmp
            self.postselect_rate = _post
            self.approx_ratio = _approx
            self.overlap = _overlap
        else:
            self._results = tmp
            self.postselect_rate = np.nan
            self.approx_ratio = np.nan
            self.overlap = np.nan

    def __repr__(self) -> str:
        attrs = ', '.join(f'{key}={value!r}' for key, value in self.__dict__.items())
        return f"RBMResult({attrs})"
    
    def __str__(self) -> str:
        if self.qaoa == False:
            return f"RBMResult with {self.nqbits} node, time step {self.tau}, post-select rate {self.postselect_rate}, approximation ratio {self.approx_ratio}, overlap with ideal solution {self.overlap}"
        else: 
            return f"QAOA+RBMResult with {self.nqbits} node, QAOA level {self.qaoa_level}, pre-training is {self.qaoa_pretrain} and training iteration as {self.train_iter} for each training, time step {self.tau}, post-select rate {self.postselect_rate}, approximation ratio {self.approx_ratio}, overlap with ideal solution {self.overlap}"
    

    def _circuit(self) -> QuantumCircuit:
        """
        Create a quantum circuit for the RBM method.
        """
        qr = QuantumRegister(self.nqbits, 'qubits')
        anc = QuantumRegister(1, 'ancilla') # Use last qubit as ancilla 
        cbits = ClassicalRegister(self.ncbits, 'mid-meas')
        meas = ClassicalRegister(self.nqbits, 'final-meas')
        circ = QuantumCircuit(qr, anc, cbits, meas)

        # Prepare equally super-position state
        for i in range(self.nqbits):
            circ.h(i)
        
        circ.barrier()
        if self.qaoa == True:
            if self.qaoa_pretrain == True:
                if self.train_iter == 0:
                    self.train_iter = 50
                training_shot = 10000
                theta = qaoa_utils.qaoa_training(num_qubit = self.nqbits, 
                                level = self.qaoa_level, 
                                hamiltonian = self.hamiltonian, 
                                training_attempt = self.training_attempt, 
                                initial = self.initial, 
                                optimizer = self.optimizer, 
                                train_iter = self.train_iter, 
                                nshot = training_shot)
            elif self.initial == 'sk':
                theta = np.hstack(qaoa_utils.get_sk_gamma_beta(self.qaoa_level))
            elif self.initial == 'd3':
                theta = np.hstack(qaoa_utils.get_gamma_beta_d3graph(self.level))
            else:
                theta = np.random.rand(2 * self.qaoa_level)
                Warning("QAOA parameters are randomly initialized!!")
            circ = qaoa_utils.qaoa_circ(circ, self.nqbits, self.hamiltonian, theta)
            circ.barrier()
        circ = add_rbm_circuit(circ, self.hamiltonian, self.tau)
        circ.measure([i for i in range(self.nqbits)], [self.ncbits + i for i in range(self.nqbits)])

        return circ
    
    def _execute(self) -> dict:
        """Execute the quantum circuit"""
        sim = AerSimulator()
        return sim.run(self.circ, 
                       shots = self.nshot).result().get_counts()
    
    def _approx_ratio(self, result: dict, decimals = 2):
        """
        Calculate the approximation ratio of the result. 
        """
        ideal_result = {}
        ideal_result[self.ideal[0]] = 100

        ideal_exp = qaoa_utils.maxcut_cost_function(ideal_result, self.hamiltonian)
        sim_exp = qaoa_utils.maxcut_cost_function(result, self.hamiltonian)
        
        return np.round(sim_exp / ideal_exp, decimals = decimals)
   
    def _overlap(self, result: dict, decimals = 2):
        """
        Overlapping with ideal solution. 
        If measuring several times, how many percentage of measurement gives ideal solution
        """
        tmp = 0
        for key in self.ideal:
            tmp += result[key]
        try:
            return np.round(tmp / sum(result.values()), decimals = decimals)
        except ZeroDivisionError:
            return np.nan
        
    def _new_approx_ratio(self, result: dict, decimals = 2):
        """
        Calculate the approximation ratio of the result. 
        """
        ideal_result = {}
        ideal_result[self.ideal[0]] = 100

        ideal_exp = qaoa_utils.maxcut_cost_function(ideal_result, self.hamiltonian)
        sim_exp = qaoa_utils.maxcut_cost_function(result, self.hamiltonian)
        
        return np.round(sim_exp / ideal_exp, decimals = decimals)