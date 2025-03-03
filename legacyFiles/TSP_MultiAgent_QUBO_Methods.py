import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from scipy.optimize import minimize
from neal import SimulatedAnnealingSampler
from dimod import BinaryQuadraticModel

"""
Methods to create the QUBO matrix for using it in Pulser with the Travelling Salesman Problem with Multi Agent representation.

- The distances are no symmetric, we use a non symmetric cost matrix.
- N is the number of stops. p_l is the number of travels in the route l. L is the number of bus lines. R = sum_{l} (N(p_l+1)) is the number of variables.
- For each line, we specify the start stop, end stop and number of travels (edges) in the route.

There are two levels of representation:

- QUBO variables: y_k, k=0,1,...,R-1 and matrices with shape (R,R). The distances are stored in an N x N matrix which is used several times.
- Stops representation: n_i, i=0,1,...,N-1. Graph representation of the stops.

"""


def create_QUBO_matrix(distances, p_list, 
                       startNodes, endNodes, L, 
                       list_of_lambdas=None, 
                       constraints_activations=None):
    
    """
    Create the QUBO matrix for the TSP with Multi Agent representation.

    Parameters
    ----------
    distances : np.array
        Matrix of distances between the stops.
    p_list : List[int]
        List of number of travels in each route.
    startNodes : List[int]
        List of start stops in each route.
    endNodes : List[int]
        List of end stops in each route.
    L : int
        Number of bus lines.
    list_of_lambdas : Optional[List[float]]
        List of weights for the constraints. The index 0 is the global constraint.
    constraints_activations : Optional[List[bool]]
        List of activations for the constraints. The index 0 is the global constraint.

    Returns
    -------
    Q : np.array
        QUBO matrix.
    """

    N = distances.shape[0]
    R = sum([N*(p+1) for p in p_list])
    Q = np.zeros((R, R))

    if list_of_lambdas is None:
        list_of_lambdas = [1.0 for _ in range(6)]

    if constraints_activations is None:
        constraints_activations = [True for _ in range(6)]
    
    offset = 0
    for l in range(L):
        p = p_list[l]
        startNode = startNodes[l] + offset
        endNode = endNodes[l] + offset + N * p
        
        # Cost function for each route
        for k in range(p):
            for i in range(N):
                for j in range(N):
                    Q[i + N*k + offset, j + N*(k+1) + offset] += distances[i, j]
        Q += Q.T  # Ensure symmetry
        
        # Local constraints for each line
        if constraints_activations[1]:  # Just one stop per position
            for t in range(p+1):
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            Q[i + N*t + offset, j + N*t + offset] += list_of_lambdas[1]
        
        if constraints_activations[2]:  # At least one stop per position
            for k in range(N * (p+1)):
                Q[k + offset, k + offset] += list_of_lambdas[2] * (1 - 2 * (p+1))
                for t in range(N * (p+1)):
                    if t != k:
                        Q[k + offset, t + offset] += list_of_lambdas[2]
        
        if constraints_activations[3]:  # Just one visit per node
            for k in range(N * (p+1)):
                for f in range(k+N, N*(p+1), N):
                    Q[k + offset, f + offset] += list_of_lambdas[3]
                    Q[f + offset, k + offset] += list_of_lambdas[3]
        
        if constraints_activations[4]:  # Start node
            Q[startNode, startNode] += -list_of_lambdas[4]
        
        if constraints_activations[5]:  # End node
            Q[endNode, endNode] += -list_of_lambdas[5]
        
        offset += N * (p + 1)
    
    """# Global constraint: Ensure all stops are visited at least once
    if constraints_activations[0]:
        global_lambda = list_of_lambdas[0]
        for i in range(N):
            indices = []
            offset = 0
            for l in range(L):
                for k in range(p_list[l] + 1):
                    indices.append(i + N*k + offset)
                offset += N * (p_list[l] + 1)
            
            # Apply the penalty: -lambda to individual variables, +lambda to quadratic terms
            for idx in indices:
                Q[idx, idx] += -global_lambda
            for idx1 in indices:
                for idx2 in indices:
                    if idx1 < idx2:
                        Q[idx1, idx2] += global_lambda
                        Q[idx2, idx1] += global_lambda  # Ensure symmetry"""
    return Q, list_of_lambdas



def is_symmetric(matrix, tol=1e-8):
    """
    Check if a matrix is symmetric.

    Parameters
    ----------
    matrix : np.array
        Matrix to check.
    tol : float
        Tolerance for the comparison.

    Returns
    -------
    bool
        True if the matrix is symmetric.
    """
    return np.allclose(matrix, matrix.T, atol=tol)


# Checks

def convertCostMatrixToQUBORepresentation(distances: np.ndarray, p_list: List[int], L: int) -> np.ndarray:
    """
    Convert the cost matrix to the QUBO representation.
    """
    N = distances.shape[0]
    R = sum([N*(p+1) for p in p_list])
    cost_matrix = np.zeros((R, R))

    offset = 0
    for l in range(L):
        p_l = p_list[l]
        for k in range(p_l):
            for i in range(N):
                for j in range(N):
                    cost_matrix[i + N*k + offset, j + N*(k+1) + offset] = distances[i, j]
        offset += N * (p_l + 1)
    
    return cost_matrix

def calculate_cost(solution_array, Q_matrix):
    """
    Calculate the cost of a solution.
    It assumes that the Q_matrix and the array are in the QUBO representation.

    If the input is the distances matrix in the QUBO representation, it will return the distances cost.
    """

    return solution_array.T @ Q_matrix @ solution_array

def check_solution(solution_array, N, p_list, startNodes, endNodes, L):
    """
    Check the constraints of the solution, including global constraint.
    """
    for l in range(L):
        offset = sum([N*(p+1) for p in p_list[:l]])
        p = p_list[l]
        startNode = startNodes[l] + offset
        endNode = endNodes[l] + offset + N * p

        if not check_constraint_1(solution_array, N, p, offset):
            print(f"Constraint 1 not fulfilled for line {l}.")
        if not check_constraint_2(solution_array, N, p, offset):
            print(f"Constraint 2 not fulfilled for line {l}.")
        if not check_constraint_3(solution_array, N, p, offset):
            print(f"Constraint 3 not fulfilled for line {l}.")
        if not check_constraint_4(solution_array, startNode):
            print(f"Constraint 4 not fulfilled for line {l}.")
        if not check_constraint_5(solution_array, endNode):
            print(f"Constraint 5 not fulfilled for line {l}.")
    
    if not check_solution_global_constraint(solution_array, N, p_list):
        print("Global constraint not fulfilled: Some stops are not visited at least once.")


def check_solution_return(solution_array, N, p_list, startNodes, endNodes, L):
    """
    Check the constraints of the solution, including global constraint.
    """
    for l in range(L):
        offset = sum([N*(p+1) for p in p_list[:l]])
        p = p_list[l]
        startNode = startNodes[l] + offset
        endNode = endNodes[l] + offset + N * p

        if not check_constraint_1(solution_array, N, p, offset):
            return False
        if not check_constraint_2(solution_array, N, p, offset):
            return False
        if not check_constraint_3(solution_array, N, p, offset):
            return False
        if not check_constraint_4(solution_array, startNode):
            return False
        if not check_constraint_5(solution_array, endNode):
            return False
    
    if not check_solution_global_constraint(solution_array, N, p_list):
        return False

    return True

def check_solution_global_constraint(solution_array, N, p_list):
    """
    Check the global constraint: all stops must be visited at least once.
    """
    visited = np.zeros(N)
    offset = 0
    for l in range(len(p_list)):
        for k in range(p_list[l] + 1):
            for i in range(N):
                if solution_array[i + N*k + offset] == 1:
                    visited[i] = 1  # Mark as visited
        offset += N * (p_list[l] + 1)
    
    return np.all(visited == 1)
    


def check_constraint_1(solution_array, N, p, offset):
    """Ensure that at most one stop is selected per position."""
    for l in range(p+1):
        if sum(solution_array[l*N + offset:(l+1)*N + offset]) > 1:
            return False
    return True

def check_constraint_2(solution_array, N, p, offset):
    """Ensure that at least one stop is selected per position."""
    for l in range(p+1):
        if sum(solution_array[l*N + offset:(l+1)*N + offset]) < 1:
            return False
    return True

def check_constraint_3(solution_array, N, p, offset):
    """Ensure that each stop is visited at most once across the route."""
    for i in range(N):
        visit_count = sum(solution_array[i + N*l + offset] for l in range(p+1))
        if visit_count > 1:
            return False
    return True

def check_constraint_4(solution_array, startNode):
    """Ensure that the route starts at the designated start node."""
    return solution_array[startNode] == 1

def check_constraint_5(solution_array, endNode):
    """Ensure that the route ends at the designated end node."""
    return solution_array[endNode] == 1


# Drawing

def draw_solution(solution_array, distances, p_list, startNodes, endNodes, L):
    """
    Draw the graph of the solution for multiple bus lines.
    """
    N = distances.shape[0]
    G = nx.Graph()
    
    # Add nodes
    for i in range(N):
        G.add_node(i)
    
    # Colors for different lines
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    edge_colors = []
    
    offset = 0
    for l in range(L):
        p = p_list[l]
        
        # Add edges based on solution
        for k in range(p):
            for i in range(N):
                for j in range(N):
                    if solution_array[i + N*k + offset] == 1 and solution_array[j + N*(k+1) + offset] == 1:
                        G.add_edge(i, j)
                        edge_colors.append(colors[l % len(colors)])
        
        offset += N * (p + 1)
    
    # Assign colors to nodes
    node_colors = ['gray' for _ in range(N)]
    for l in range(L):
        node_colors[startNodes[l]] = 'red'
        node_colors[endNodes[l]] = 'green'
    
    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=node_colors, edge_color=edge_colors, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)


# Optimization of lambdas

def generate_valid_naive_solution(N, p_list, startNodes, endNodes, L):
    """
    Generate an initial naive valid solution.
    """
    R = sum([N*(p+1) for p in p_list])
    solution = np.zeros(R)
    offset = 0
    
    for l in range(L):
        p = p_list[l]
        startNode = startNodes[l] + offset
        endNode = endNodes[l] + offset + N * p
        solution[startNode] = 1
        solution[endNode] = 1
    
        shift = 0
        for k in range(p - 1):
            if k + shift == startNodes[l] or k + shift == endNodes[l]:
                shift += 1
            solution[N * (k + 1) + (k + shift) + offset] = 1
        
        offset += N * (p + 1)
    
    return solution



def optimize_lambdas(distances, p_list, startNodes, endNodes, method="L-BFGS-B", initial_solution= None, initial_lambdas=None):
    """
    Optimize the lambdas for the QUBO matrix with adaptive balancing between cost and constraints.

    Parameters
    ----------
    distances : np.array
        Matrix with the distances between the stops. It is not symmetric. Has shape (N,N).
    p_list : List[int]
        List of number of travels in each route.
    startNodes : List[int]
        List of start stops in each route.
    endNodes : List[int]
        List of end stops in each route. 
    method : str
        Optimization method from scipy.optimize.minimize. Default is "L-BFGS-B".
    initial_solution : Optional[np.array]
        Initial solution to use for the optimization.
    initial_lambdas : Optional[np.array]
        Initial values for the lambdas

    Returns
    -------
    lambdas : np.array
        Optimized lambdas for the initial tentative solution.
    """
    N = distances.shape[0]
    L = len(p_list)

    def cost_function(lambdas):
        
        # Create the QUBO matrix
        Q, _ = create_QUBO_matrix(distances, p_list, startNodes, endNodes, L, list_of_lambdas=lambdas)
        
        # Generate a valid initial solution
        if initial_solution is not None:
            solution_guess = initial_solution
        else:
            solution_guess = generate_valid_naive_solution(N, p_list, startNodes, endNodes, L)

        cost = calculate_cost(solution_guess, Q)
        
        # Adaptive penalty based on the violation level
        penalty = 0
        if not check_solution_return(solution_guess, N, p_list, startNodes, endNodes, L):
            penalty = 10000 * abs(cost)  # Adaptive penalty proportional to cost
        
        return cost + penalty
    
    if initial_lambdas is not None:
        lambdas_init = initial_lambdas
    else:
        lambdas_init = np.ones(6) * 0.1  # Start with small lambda values
    res = minimize(cost_function, lambdas_init, method=method, bounds=[(0, 10)] * 6)
    
    return res.x


# Solvers

def solve_qubo_with_Dwave(Q):
    """
    Solve the QUBO problem using D-Wave's Simulated Annealing Sampler.
    """
    num_vars = Q.shape[0]
    Q_dict = {(i, j): Q[i, j] for i in range(num_vars) for j in range(i, num_vars)}
    
    bqm = BinaryQuadraticModel.from_qubo(Q_dict)
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=100)
    
    best_solution = np.array(list(response.first.sample.values()))
    best_cost = response.first.energy
    
    return best_solution, best_cost
        