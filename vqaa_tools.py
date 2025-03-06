'''
Tools to solve QUBO using an analog device and VQAA
'''

import numpy as np
import matplotlib.pyplot as plt
from pulser import Pulse, Sequence, Register
from pulser_simulation import QutipBackend, SimConfig, QutipEmulator
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform, ConstantWaveform
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import itertools


def evaluate_mapping(new_coords, Q):
    """Cost function to minimize. Ideally, the pairwise distances are conserved."""
    new_coords = np.reshape(new_coords, (len(Q), 2))
    # computing the matrix of the distances between all coordinate pairs
    new_Q = squareform(
        DigitalAnalogDevice.interaction_coeff / pdist(new_coords) ** 6
    )
    return np.linalg.norm(new_Q - Q)


def biggest_difference(new_Q, Q):
    """Computes the biggest elementwise relative difference between two matrices"""
    
    # Considers only non-zero elements
    mask = Q != 0  
    
    relative_errors = 100*np.abs((new_Q[mask] - Q[mask]) / Q[mask])
 
    return np.max(relative_errors)


def compute_element_U(coords):
    """Computes the value of an element in the interaction matrix U corresponding to coordinates coords"""
    return (DigitalAnalogDevice.interaction_coeff / (pdist(coords)[0]) ** 6)


def minimization_embedding(Q, show=False):
    """Embedding of QUBO matrix Q using a minimization strategy"""
    costs = []
    np.random.seed(0)
    x0 = np.random.random(len(Q) * 2)

    def callback(xk):
        cost = evaluate_mapping(xk, Q)  
        costs.append(cost)  


    res = minimize(
        evaluate_mapping,
        x0, 
        args=(Q,),  
        method="Nelder-Mead",
        tol=1e-6,
        options={"maxiter": 2000000, "maxfev": None},
        callback=callback
    )
    coords = np.reshape(res.x, (len(Q), 2))

    new_Q = (DigitalAnalogDevice.interaction_coeff / pdist(coords) ** 6)

    if show:
        plt.plot(costs)

    return coords


def compute_U(coords):
    """Computes the whole interaction matrix for a set of coordinates"""
    return  squareform(DigitalAnalogDevice.interaction_coeff / pdist(coords) ** 6)


def frobenius_similarity(new_Q, Q):
    """Frobenius similarity between two matrices"""
    Q_ = Q / np.linalg.norm(Q)
    U_ = new_Q / np.linalg.norm(new_Q)
    norm = np.linalg.norm(Q_ - U_)

    return norm


def atoms_register(coords, show=True):
    """Shows the embedded register for a set of coordinates"""
    qubits = {f"q{i}": coord for (i, coord) in enumerate(coords)}
    reg = Register(qubits)
    if show:
        reg.draw(
            blockade_radius=DigitalAnalogDevice.rydberg_blockade_radius(1.0),
            draw_graph=True,
            draw_half_radius=True,
        )
    return reg


def atoms_list(N):
    """Prepares a list of N elements from 0 to N-1"""
    return [_ for _ in range(N)]


def heuristical_embedding(V, P, Q):
    """
    Register Embedding Algorithm (arxiv.org/pdf/2402.05748)

    V : set of atoms [0, ..., N-1]

    P : set of positions (grid of positions)
    
    Q : QUBO matrix
    """
    Pa = [] # list of already assigned positions

    u = V[0]
    center = (0.0, 0.0)
    Pa.append([u, center])

    # Update the remaining atoms and positions
    V.remove(u)
    P.remove(center)
    while V:
        u = V[0]
        min_sum = float("inf")
        for p in P:
            sum = 0
            for atom_position in Pa:
                # For every already places atom, we unpack the value of the atom and its position
                v = atom_position[0] 
                pv = atom_position[1]

                sum = sum + np.abs(Q[u,v]-compute_element_U(np.vstack([p,pv])))
                
            # If we find the best position available so far
            if sum<min_sum:
                # Update best position and the actual min_sum
                min_sum = sum
                best_position = p

            # We keep looking for better positions until we have examined all of them

        # Update the remaining atoms and position sets and the placed atoms-positions list
        Pa.append([u, best_position])
        P.remove(best_position)
        V.remove(u)

        # Prepare the coordinates in array form
        coords = np.array([item[1] for item in Pa])

    return coords


def generate_grid(x_max, y_max, g):
    """Generates a grid of points in the space [-x_max, x_max] x [-y_max, y_max] separated g in each direction"""

    x_vals = np.round(np.arange(-x_max-g, x_max+g, g), 5)  
    y_vals = np.round(np.arange(-y_max-g, y_max+g, g), 5) 

    P = list(itertools.product(x_vals, y_vals))  

    return P


def define_sequence_qaa(register, Q, omega, delta, T=4000, device = DigitalAnalogDevice, show=False):
    """Defines the sequence for the register and adds the neccessary channels to apply QAA,
    including the adiabatic pulse to the Rydberg Global channel, as well as a constant pulse to the DMM"""

    sequence = Sequence(register, device=device)
    sequence.declare_channel("rydberg_global", "rydberg_global")

    node_weights = np.diag(Q)
    norm_node_weights = node_weights/np.min(node_weights) # we use the min for normalisation since the diagonal values are negative
    det_map_weights = 1 - norm_node_weights


    det_map = register.define_detuning_map(
        {f"q{i}" : det_map_weights[i] for i in range(len(det_map_weights))}
    )

    sequence.config_detuning_map(det_map, "dmm_0")

    adiabatic_pulse = Pulse(
        InterpolatedWaveform(T, [1e-9, omega, 1e-9]),
        InterpolatedWaveform(T, [-delta, 0, delta]),
        0
    )

    sequence.add(adiabatic_pulse, "rydberg_global")

    sequence.add_dmm_detuning(ConstantWaveform(T, -delta*(1-np.min(det_map_weights))), "dmm_0")
    
    if show:
        sequence.draw(
            draw_detuning_maps=True,
            draw_qubit_det=True,
            draw_qubit_amp=True
        )

    return sequence


def run_sequence(sequence, N=1000):
    """Runs the sequence and returns the count dictionary"""
    simul = QutipEmulator.from_sequence(
        sequence,
        sampling_rate=0.05,
        config=SimConfig(
            runs=10,
        ),
        evaluation_times="Minimal",
    )
    results = simul.run()
    # final = results.get_final_state()
    count_dict = results.sample_final_state(N_samples=N)

    return count_dict


def plot_distribution(C,N, best_solutions):
    """Plots the distribution for the sampled bitstrings"""

    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True)[:N])
    color_dict = {key: "r" if key in best_solutions else "g" for key in C}
    plt.figure(figsize=(12, 6))
    plt.xlabel("bitstrings")
    plt.ylabel("counts")
    plt.bar(C.keys(), C.values(), width=0.5, color=color_dict.values())
    plt.xticks(rotation="vertical")
    plt.show()


def agg_cost(count_dict, Q):
    """Computes the aggregated cost of the sampled solutions in count dict"""
    cost = 0
    for key, times in count_dict.items():
        vector_key = np.array([int(bit) for bit in key]).reshape(-1, 1)
        cost += times*vector_key.T@Q@vector_key
    
    return cost


def simple_quantum_loop(Q, register, parameters):
    """Loop to run the whole process of QAA"""
    params = np.array(parameters)
    print(params)
    parameter_omega = params[0] 
    parameter_detuning = params[1]
    seq = define_sequence_qaa(register, Q, parameter_omega, parameter_detuning)

    counts = run_sequence(seq)

    return counts

def func_simple(parameters, *args):
    """Function to get the cost of a run in QAA with certain parameters"""
    Q = args[0][0]
    register = args[0][1]
    Cs = args[0][2]
    C = simple_quantum_loop(Q, register, parameters)
    Cs[0] = C
    cost = agg_cost(C, Q)
    return cost

def run_vqaa(Q, register):
    """Run the VQAA and return the optimal solutions"""
    x0 = np.random.uniform(0, 10, 2)
    Cs = [-1]
    res = minimize(
        func_simple,
        x0,
        args=[Q, register, Cs],
        method="Nelder-Mead",
        tol=1e-3,
        options={"maxiter": 1, "maxfev": 1} 
    )

    x_opt = res.x
    return Cs[0], x_opt
