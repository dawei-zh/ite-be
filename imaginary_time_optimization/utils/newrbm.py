import numpy as np
import imaginary_time_optimization.utils.general as general_utils
import imaginary_time_optimization.utils.rbm as rbm
import imaginary_time_optimization.utils.qaoa as qaoa_utils
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

def _one_body(circ: QuantumCircuit, 
              pos: int, 
              ancilla: int, 
              W: float):
    
    b = np.pi / 4
    # Implement ZX rotation
    circ.h(ancilla)
    circ.cx(pos, ancilla)
    circ.rz(2 * W, ancilla)
    circ.cx(pos, ancilla)
    circ.h(ancilla)
    # Implement X rotation
    circ.rx(2 * b, ancilla)

    return circ

def _two_body(circ, pos1, pos2, W, meas_bit, correction = False):
    """Create two body circuits for certain two-qubits Z_{pos1}Z_{pos2} Pauli string"""
    num_qubits = circ.num_qubits
    ancilla = num_qubits - 1
    
    circ.reset(ancilla) # Use last qubit as ancilla
    circ.cx(pos1, pos2)

    circ = _one_body(circ, pos2, ancilla, W)
    
    if correction == True:
        circ.cx(ancilla, pos2)
    
    circ.cx(pos1, pos2)
    #circ.measure(ancilla, meas_bit)
    circ.barrier()

    return circ

def add_newrbm_circuit(circ: QuantumCircuit, 
                       hamiltonian: list, 
                       tau: float, correction = True) -> QuantumCircuit:
    """Only include two body interaction for now."""

    if correction == True:
        part1 = []
        part2 = []
        tmp = []
        for j in range(len(hamiltonian)):
            pos1, pos2, weight = hamiltonian[j]
            if pos1 not in tmp and pos2 not in tmp:
                part1.append((pos1, pos2, weight, True))
                tmp.append(pos1)
                tmp.append(pos2)
            else:
                part2.append(hamiltonian[j])

        hamiltonian = part1 + part2

    
    # for j in range(len(hamiltonian)):
    #     if len(hamiltonian[j]) == 4:
    #         pos1, pos2, weight, flag = hamiltonian[j]
    #     else:
    #         pos1, pos2, weight = hamiltonian[j]

    #     K = weight * tau

    #     A = np.sqrt(np.exp(4 * K) + 1) / np.sqrt(4 * np.exp(2 * K))
    #     W = np.arctan(np.exp(2 * A)) - np.pi / 4
    #     W = np.arccos(np.exp(-2 * K)) / 2
    #     circ = _two_body(circ, pos1, pos2, W, j, flag)
    #     flag = False
    for j in range(len(hamiltonian)):
        if len(hamiltonian[j]) == 4:
            pos1, pos2, weight, flag = hamiltonian[j]
            K = weight * tau

            A = np.sqrt(np.exp(4 * K) + 1) / np.sqrt(4 * np.exp(2 * K))
            W = np.arctan(np.exp(2 * A)) - np.pi / 4
            W = np.arccos(np.exp(-2 * K)) / 2
            circ = _two_body(circ, pos1, pos2, W, j, flag)
        else:
            pos1, pos2, weight = hamiltonian[j]
            K = weight * tau
            if K > 0:
                s = 1
            else:
                s = -1

            W = np.arccos(np.exp(-2 * K)) / 2
            circ = rbm._two_body(circ, pos1, pos2, W, s, j)

    return circ


class NewRBMResult:
    def __init__(self, 
                 tau: float,
                 num_qubits: int,
                 hamiltonian: list, 
                 ideal_solution: list,
                 nshot: int, 
                 repeat: int,
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
        return f"RBMResult with {self.nqbits} node, time step {self.tau}, post-select rate {self.postselect_rate}, approximation ratio {self.approx_ratio}, overlap with ideal solution {self.overlap}"
    

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

        circ = add_newrbm_circuit(circ, self.hamiltonian, self.tau)
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