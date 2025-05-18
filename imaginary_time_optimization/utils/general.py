from typing import List
import numpy as np

def get_basis_state(num_qubit: int) -> List[str]:
    """Get all computational basis binary string for n-qubits system"""
    if num_qubit <= 0:
        print("nqubit should be greater than 0")
        return 0
    
    if num_qubit > 1:
        tmp1 = ['0' + i for i in get_basis_state(num_qubit-1)]
        tmp2 = ['1' + i for i in get_basis_state(num_qubit-1)]
        return tmp1 + tmp2
    if num_qubit == 1:
        return ['0', '1']
    
def empty_standard_dict(num_qubit: int) -> dict:
    """Create an empty dictionary for each n-qubit state with value 0. """
    basis_states = get_basis_state(num_qubit)
    empty_standard_dict1 = {}
    for state in basis_states:
        empty_standard_dict1[state] = 0
        
    return empty_standard_dict1

def post_select(count: dict, 
                num_qubits: int, 
                num_cbits: int) -> dict:
    
    final = empty_standard_dict(num_qubits)

    for key in count.keys():
        meas, ancilla = key.split(' ')
        assert len(ancilla) == num_cbits
        # Choose data with mid-meas as all zero
        if ancilla == '0' * num_cbits:
            #meas = meas[::-1]
            # Add data to standard dictionary
            # Change the output string order
            final[meas] += count[key]

    return final

def invert_counts(counts):
    return {k[::-1]:v for k, v in counts.items()}

def combine_results(result: list, num_qubits: int) -> dict:
    """
    Combine results from repeated experiments
    """
    final = empty_standard_dict(num_qubits)
    for res in result:
        for key in res.keys():
            final[key] += res[key]

    return final

def two_body_expectation(result: dict, hamiltonian: list):
    """
    Calculate the expectation value of a two-body Z terms Hamiltonian
    """
    cost = 0
    for key, value in result.items():
        tmp_cost = 0
        for i in range(len(hamiltonian)):
            pos1, pos2, weight = hamiltonian[i]
            if key[pos1] == key[pos2]:
                tmp_cost += value * weight
            else:
                tmp_cost -= value * weight

        cost += tmp_cost
    try:
        return cost / sum(result.values())
    except ZeroDivisionError:
        return np.nan