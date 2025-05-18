from qiskit import QuantumCircuit
import numpy as np
from scipy.optimize import minimize
from qiskit_aer import AerSimulator
import imaginary_time_optimization.utils.general as general_utils

def qaoa_circ(circ: QuantumCircuit, 
              num_qubits: int,
              hamiltonian: list, 
              theta: list) -> QuantumCircuit:
    """
    Create p-level QAOA circuit
    """
    level = len(theta) // 2
    gamma = theta[:level] # parameters for problem unitary
    beta = theta[level:] # parameters for mixer unitary

    for i in range(level):
        for j in range(len(hamiltonian)):
            pos1, pos2, weight = hamiltonian[j]

            circ.cx(pos1, pos2)
            circ.rz(-1 * gamma[i] * weight, pos2) # be careful here about the value with weight
            circ.cx(pos1, pos2)

        circ.barrier()

        for j in range(0, num_qubits):
            circ.rx(2 * beta[i], j)
                
        circ.barrier()

    return circ

def objective_function(theta, num_qubits, hamiltonian, verbose = 0):
    """
    QAOA is going to minimize sum_{i,j}[(<Z_jZ_k> - 1) * weight_{jk}/2]
    """
    tmp_circ = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        tmp_circ.h(i)
    tmp_circ.barrier()

    tmp_circ = qaoa_circ(tmp_circ, num_qubits, hamiltonian, theta)
    tmp_circ.measure_all()
    nshot = 10000
    aer_sim = AerSimulator()
    result = general_utils.invert_counts(aer_sim.run(tmp_circ, shots = nshot).result().get_counts())

    if verbose == 1:
        return (maxcut_cost_function(result, hamiltonian), tmp_circ)

    return maxcut_cost_function(result, hamiltonian)

def maxcut_cost_function(result, hamiltonian):
    """
    Minimize cost function sum_{i,j}[weight_{jk}/2 * (<Z_jZ_k> - 1)] for MaxCut
    """
    cost = 0
    nshot = sum(result.values())
    for string, count in result.items():
        tmp_cost = 0
        for i in range(len(hamiltonian)):
            pos1, pos2, weight = hamiltonian[i]
            if string[pos1] != string[pos2]: 
                # The j-th bit and k-th bit are either 01 or 10
                # Which indicates <Z_jZ_k> = -1
                exp_zz = -1 
            else:
                # The j-th bit and k-th bit are either 00 or 11
                # Which indicates <Z_jZ_k> = 1
                exp_zz = 1

            #tmp_cost += (count / nshot) * weight * 0.5 * (exp_zz - 1)
            tmp_cost += count * weight * 0.5 * (exp_zz - 1)
        cost += tmp_cost
        
    try:
        return cost / nshot
    except ZeroDivisionError:
        return np.nan

def _training(num_qubits: int,
                  hamiltonian: list, 
                  theta: list, 
                  optimizer: str = 'COBYLA',
                  maxiter: int = 1000) -> np.ndarray:
    """
    Return parameter optimization result for QAOA
    """
    
    options = {
        'maxiter': maxiter,    # Number of iterations
        'disp': False       
    }

    train_result = minimize(objective_function, 
                            theta, 
                            args = (num_qubits, hamiltonian), 
                            method = optimizer, 
                            options = options)

    return train_result.x

def qaoa_training(num_qubit: int,
                  level: int, 
                  hamiltonian: list, 
                  training_attempt: int, 
                  initial: str, 
                  optimizer: str, 
                  train_iter: int, 
                  nshot: int) -> np.ndarray:
    aer_sim = AerSimulator()
    tmp_cost = []
    tmp_theta = []
    for _ in range(training_attempt):
        if initial == 'sk':
            theta = np.hstack(get_sk_gamma_beta(level))
        elif initial == 'd3':
            theta = np.hstack(get_gamma_beta_d3graph(level))
        else:
            theta = 2 * np.pi * np.random.random(2 * level)
        
        _theta = _training(num_qubit, 
                           hamiltonian, 
                           theta, 
                           optimizer, 
                           train_iter)
        tmp_circ = QuantumCircuit(num_qubit)
        for i in range(num_qubit):
            tmp_circ.h(i)
        tmp_circ.barrier()

        tmp_circ = qaoa_circ(tmp_circ, num_qubit, hamiltonian, _theta)
        tmp_circ.measure_all()

        tmp_res = general_utils.invert_counts(aer_sim.run(tmp_circ, shots = nshot).result().get_counts())
        tmp_cost.append(maxcut_cost_function(tmp_res, hamiltonian))
        tmp_theta.append(_theta)

    return tmp_theta[np.argmin(tmp_cost)]


def get_sk_gamma_beta(level: int) -> np.ndarray:
    """
    Load the look-up table for initial points from
    https://arxiv.org/pdf/2110.14206.pdf
    """
    assert level <= 17, "Level should be less than or equal to 17"
    sk_gamma = [np.array([0.5]), 
                np.array([0.3817, 0.6655]), 
                np.array([0.3297, 0.5688, 0.6406]), 
                np.array([0.2949, 0.5144, 0.5586, 0.6429]), 
                np.array([0.2705, 0.4804, 0.5074, 0.5646, 0.6397]), 
                np.array([0.2528, 0.4531, 0.4750, 0.5146, 0.5650, 0.6392]), 
                np.array([0.2383, 0.4327, 0.4516, 0.4830, 0.5147, 0.5686, 0.6393]), 
                np.array([0.2268, 0.4162, 0.4332, 0.4608, 0.4818, 0.5179, 0.5717, 0.6393]), 
                np.array([0.2172, 0.4020, 0.4187, 0.4438, 0.4592, 0.4838, 0.5212, 0.5754, 0.6398]), 
                np.array([0.2089, 0.3902, 0.4066, 0.4305, 0.4423, 0.4604, 0.4858, 0.5256, 0.5789, 0.6402]), 
                np.array([0.2019, 0.3799, 0.3963, 0.4196, 0.4291, 0.4431, 0.4611, 0.4895, 0.5299, 0.5821, 0.6406]), 
                np.array([0.1958, 0.3708, 0.3875, 0.4103, 0.4185, 0.4297, 0.4430, 0.4639, 0.4933, 0.5343, 0.5851, 0.6410]), 
                np.array([0.1903, 0.3627, 0.3797, 0.4024, 0.4096, 0.4191, 0.4290, 0.4450, 0.4668, 0.4975, 0.5385, 0.5878, 0.6414]), 
                np.array([0.1855, 0.3555, 0.3728, 0.3954, 0.4020, 0.4103, 0.4179, 0.4304, 0.4471, 0.4703, 0.5017, 0.5425, 0.5902, 0.6418]), 
                np.array([0.1811, 0.3489, 0.3667, 0.3893, 0.3954, 0.4028, 0.4088, 0.4189, 0.4318, 0.4501, 0.4740, 0.5058, 0.5462, 0.5924, 0.6422]), 
                np.array([0.1771, 0.3430, 0.3612, 0.3838, 0.3896, 0.3964, 0.4011, 0.4095, 0.4197, 0.4343, 0.4532, 0.4778, 0.5099, 0.5497, 0.5944, 0.6425]), 
                np.array([0.1735, 0.3376, 0.3562, 0.3789, 0.3844, 0.3907, 0.3946, 0.4016, 0.4099, 0.4217, 0.4370, 0.4565, 0.4816, 0.5138, 0.5530, 0.5962, 0.6429])]
    sk_beta = [np.array([np.pi / 8]), 
               np.array([0.4960, 0.2690]), 
               np.array([0.5500, 0.3675, 0.2109]), 
               np.array([0.5710, 0.4176, 0.3028, 0.1729]), 
               np.array([0.5899, 0.4492, 0.3559, 0.2643, 0.1486]), 
               np.array([0.6004, 0.4670, 0.3880, 0.3176, 0.2325, 0.1291]), 
               np.array([0.6085, 0.4810, 0.4090, 0.3534, 0.2857, 0.2080, 0.1146]), 
               np.array([0.6151, 0.4906, 0.4244, 0.3780, 0.3224, 0.2606, 0.1884, 0.1030]), 
               np.array([0.6196, 0.4973, 0.4354, 0.3956, 0.3481, 0.2973, 0.2390, 0.1717, 0.0934]), 
               np.array([0.6235, 0.5029, 0.4437, 0.4092, 0.3673, 0.3246, 0.2758, 0.2208, 0.1578, 0.0855]), 
               np.array([0.6268, 0.5070, 0.4502, 0.4195, 0.3822, 0.3451, 0.3036, 0.2571, 0.2051, 0.1459, 0.0788]), 
               np.array([0.6293, 0.5103, 0.4553, 0.4275, 0.3937, 0.3612, 0.3248, 0.2849, 0.2406, 0.1913, 0.1356, 0.0731]), 
               np.array([0.6315, 0.5130, 0.4593, 0.4340, 0.4028, 0.3740, 0.3417, 0.3068, 0.2684, 0.2260, 0.1792, 0.1266, 0.0681]), 
               np.array([0.6334, 0.5152, 0.4627, 0.4392, 0.4103, 0.3843, 0.3554, 0.3243, 0.2906, 0.2535, 0.2131, 0.1685, 0.1188, 0.0638]), 
               np.array([0.6349, 0.5169, 0.4655, 0.4434, 0.4163, 0.3927, 0.3664, 0.3387, 0.3086, 0.2758, 0.2402, 0.2015, 0.1589, 0.1118, 0.0600]), 
               np.array([0.6363, 0.5184, 0.4678, 0.4469, 0.4213, 0.3996, 0.3756, 0.3505, 0.3234, 0.2940, 0.2624, 0.2281, 0.1910, 0.1504, 0.1056, 0.0566]), 
               np.array([0.6375, 0.5197, 0.4697, 0.4499, 0.4255, 0.4054, 0.3832, 0.3603, 0.3358, 0.3092, 0.2807, 0.2501, 0.2171, 0.1816, 0.1426, 0.1001, 0.0536])]
    
    return (sk_gamma[level - 1], sk_beta[level - 1])

def get_gamma_beta_d3graph(level: int) -> np.ndarray:
    """
    Returns the parameters for QAOA for MaxCut on regular graphs from arXiv:2107.00677
    """
    assert level <= 11, "Level should be less than or equal to 11"
    fix_gamma = [np.array([0.616]), 
                 np.array([0.488, 0.898]), 
                 np.array([0.422, 0.798, 0.937]), 
                 np.array([0.409, 0.781, 0.988, 1.156]), 
                 np.array([0.360, 0.707, 0.823, 1.005, 1.154]), 
                 np.array([0.331, 0.645, 0.731, 0.837, 1.009, 1.126]), 
                 np.array([0.310, 0.618, 0.690, 0.751, 0.859, 1.020, 1.122]), 
                 np.array([0.295, 0.587, 0.654, 0.708, 0.765, 0.864, 1.026, 1.116]), 
                 np.array([0.279, 0.566, 0.631, 0.679, 0.726, 0.768, 0.875, 1.037, 1.118]), 
                 np.array([0.267, 0.545, 0.610, 0.656, 0.696, 0.729, 0.774, 0.882, 1.044, 1.115]), 
                 np.array([0.257, 0.528, 0.592, 0.640, 0.677, 0.702, 0.737, 0.775, 0.884, 1.047, 1.115])]
    
    fix_beta = [np.array([0.393]),
                np.array([0.555, 0.293]), 
                np.array([0.609, 0.459, 0.235]), 
                np.array([0.600, 0.434, 0.297, 0.159]), 
                np.array([0.632, 0.523, 0.390, 0.275, 0.149]), 
                np.array([0.636, 0.534, 0.463, 0.360, 0.259, 0.139]), 
                np.array([0.648, 0.554, 0.490, 0.445, 0.341, 0.244, 0.131]), 
                np.array([0.649, 0.555, 0.500, 0.469, 0.420, 0.319, 0.231, 0.123]), 
                np.array([0.654, 0.562, 0.509, 0.487, 0.451, 0.403, 0.305, 0.220, 0.117]), 
                np.array([0.656, 0.563, 0.514, 0.496, 0.469, 0.436, 0.388, 0.291, 0.211, 0.112]), 
                np.array([0.656, 0.563, 0.516, 0.504, 0.482, 0.456, 0.421, 0.371, 0.276, 0.201, 0.107])]
    
    return (fix_gamma[level - 1], fix_beta[level - 1])

class QAOAResult:
    def __init__(self, 
                 level: int,
                 initial: str,
                 num_qubits: int,
                 hamiltonian: list, 
                 ideal_solution: dict,
                 nshot: int, 
                 repeat: int,
                 training_attempt: int,
                 train_iter: int,
                 optimizer: str,
                 pretrain: bool,
                 verbose = 0):
        """
        Customized class for solution of QAOA method. 

        Verbose 0: averaged result
        Verbose 1: best result
        Verbose 2: all results
        """
        self.level = level
        self.initial = initial
        self.nqbits = num_qubits
        self.hamiltonian = hamiltonian
        self.ideal = ideal_solution
        self.nshot = nshot
        self.training_attempt = training_attempt
        self.train_iter = train_iter
        self.optimizer = optimizer
        self.pretrain = pretrain

        tmp = []
        _approx = []
        _overlap = []
        for _ in range(repeat):
            if self.pretrain == True:
                if initial == 'sk':
                    self.optimized_theta = np.hstack(get_sk_gamma_beta(level))
                if initial == 'd3':
                    self.optimized_theta = np.hstack(get_gamma_beta_d3graph(level))
            else:
                self.optimized_theta = self._training()
            self.circ = self._circuit()
            _res = general_utils.invert_counts(self._execute())
            tmp.append(_res)
            _approx.append(self._approx_ratio(_res))
            _overlap.append(self._overlap(_res))

        if verbose == 0: 
            self._results = general_utils.combine_results(tmp, self.nqbits)
            self.approx_ratio = np.mean(_approx)
            self.overlap = np.mean(_overlap)
        elif verbose == 1:
            self.approx_ratio = np.max(_approx)
            self.overlap = np.max(_overlap)
            if np.argmax(_approx) == np.argmax(_overlap):
                self._results = tmp[np.argmax(_approx)]
            else:
                self._results = tmp[np.argmax(_approx)]
                print("Approximation ratio and overlap gives different best result")
        else:
            self._results = tmp
            self.approx_ratio = _approx
            self.overlap = _overlap

    def __repr__(self) -> str:
        attrs = ', '.join(f'{key}={value!r}' for key, value in self.__dict__.items())
        return f"QAOAResult({attrs})"
    
    def __str__(self) -> str:
        return f"QAOAResult with {self.nqbits} node, level {self.level}, approximation ratio {self.approx_ratio}, overlap with ideal solution {self.overlap}"
    
    def _training(self):
        return qaoa_training(num_qubit = self.nqbits, 
                             level = self.level, 
                             hamiltonian = self.hamiltonian, 
                             training_attempt = self.training_attempt, 
                             initial = self.initial, 
                             optimizer = self.optimizer, 
                             train_iter = self.train_iter, 
                             nshot = self.nshot) 


    def _circuit(self) -> QuantumCircuit:
        """
        Create a quantum circuit for the QAOA method.
        """
        circ = QuantumCircuit(self.nqbits)
        for i in range(self.nqbits):
            circ.h(i)
        circ.barrier()
        circ = qaoa_circ(circ, self.nqbits, self.hamiltonian, self.optimized_theta)
        circ.measure_all()

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

        ideal_exp = maxcut_cost_function(ideal_result, self.hamiltonian)
        sim_exp = maxcut_cost_function(result, self.hamiltonian)
        
        return np.round(sim_exp / ideal_exp, decimals = decimals)
   
    def _overlap(self, result: dict, decimals = 2):
        """
        Overlapping with ideal solution. 
        If measuring several times, how many percentage of measurement gives ideal solution
        """
        tmp = 0
        for key in self.ideal:
            if key in result:
                tmp += result[key]
            
        try:
            return np.round(tmp / sum(result.values()), decimals = decimals)
        except ZeroDivisionError:
            return np.nan