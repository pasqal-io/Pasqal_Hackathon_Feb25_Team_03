import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, List

"""
Methods to create the QUBO matrix for using it in Pulser.

- The distances are tranformed to be symmetric and use a triangular superior matrix.
- N is the number of stops. R=N(N-1)/2 is the number of relevant edges.
- We consider just one line which has an start (stop 0), and end (stop N-1) and p+1 stops in total (p edges).

Thera are three levels of representation:

- QUBO variables: y_k, k=0,1,...,R-1 and matrices with shape (R,R). The distances are stored in the diagonal of the matrix.
- Adjacency matrix: x_ij, i,j=0,1,...,N-1. Matrices of shape (N,N). The distances are stored in the upper triangular matrix.
- Stops representation: n_i, i=0,1,...,N-1. Graph representation of the stops.

"""

# Conversion from adjacency matrix to QUBO variables and viceversa

def convert_from_N_to_R(N):
    return int(N*(N-1)/2)

def convert_from_R_to_N(R):
    return int((1 + np.sqrt(1 + 8*R)) / 2)

def convert_from_ij_to_k(i, j, N):
    if i >= N or j >= N:
        raise ValueError(f'i={i} or j={j} is out of bounds for N={N}')
    return sum(N-1-m for m in range(i)) + j - i - 1

def convert_from_k_to_ij(k, R):
    N = convert_from_R_to_N(R)
    if k >= R:
        raise ValueError(f'k={k} is out of bounds for R={R}')
    i = 0
    Si = sum(N-1-m for m in range(i))
    while Si <= k:
        i += 1
        Si = sum(N-1-m for m in range(i))

    i -= 1
    j = k + i + 1 - sum(N-1-m for m in range(i))
    return i, j

def convert_from_adjacency_to_QUBO(adjacency_matrix):
    N = len(adjacency_matrix)
    R = convert_from_N_to_R(N)
    QUBO_matrix = np.zeros((R, R))
    for i in range(N):
        for j in range(i+1, N):
            k = convert_from_ij_to_k(i, j, N)
            QUBO_matrix[k, k] = adjacency_matrix[i, j]
    return QUBO_matrix

def convert_from_QUBO_to_adjacency(QUBO_matrix):
    R = len(QUBO_matrix)
    N = convert_from_R_to_N(R)
    adjacency_matrix = np.zeros((N, N))
    for k in range(R):
        i, j = convert_from_k_to_ij(k, R)
        adjacency_matrix[i, j] = QUBO_matrix[k, k]
    return adjacency_matrix


# QUBO matrix creation

def create_distances_matrix(distances, node0: Optional[int] = None, nodeN1: Optional[int] = None):
    """
    From an asymmetric distances matrix, it creates a triangular superior matrix by summing the transpose
    and eliminating the elements under the diagonal. Allows reordering so that a specified node becomes
    the first (0) and another becomes the last (N-1). If no nodes are specified, selects the two nodes
    with the maximum distance.
    
    Parameters:
    - distances: np.array, asymmetric distances matrix.
    - node0: int, optional, node to place at position 0.
    - nodeN1: int, optional, node to place at position N-1.
    
    Returns:
    - distances_matrix: np.array, the symmetric distance matrix with upper triangular elements.
    """
    
    # Ensure distances is a numpy array
    distances = np.array(distances)
    N = distances.shape[0]
    
    # Make the matrix symmetric
    distances_matrix = distances + distances.T
    
    if node0 is None or nodeN1 is None:
        # Find the indices of the maximum nonzero element in the symmetric matrix
        max_val = np.max(distances_matrix * (distances_matrix > 0))
        k, l = np.argwhere(distances_matrix == max_val)[0]
        return create_distances_matrix(distances, node0=k, nodeN1=l)
    
    # Create the new order
    remaining_nodes = [i for i in range(N) if i not in {node0, nodeN1}]
    order = [node0] + remaining_nodes + [nodeN1]
    
    # Reorder the matrix
    distances = distances[np.ix_(order, order)]
    
    # Make the matrix symmetric again after reordering
    distances_matrix = distances + distances.T
    
    # Keep only the upper triangular part, setting lower triangular to zero
    distances_matrix = np.triu(distances_matrix, k=1)
    
    return distances_matrix

def introduce_array_in_diagonal(array):
    """
    Introduce the array in the diagonal of the matrix.
    """
    matrix = np.zeros((len(array), len(array)))
    for i in range(len(array)):
        matrix[i, i] = array[i]
    return matrix


def create_QUBO_matrix(list_of_lambdas, p, R, distances_matrix, active_constraints: Optional[List[bool]] = None):
    """
    Create the QUBO matrix for the Bus Routes Optimization problem.
    The distances_matrix is assumed to be in the QUBO representation (diagonal matrix).
    """

    if active_constraints is None:
        active_constraints = [True, True, True, True]

    Q = np.zeros((R,R))
    N = convert_from_R_to_N(R)
    if p >= N:
        raise ValueError("The number of stops connections p must be less than N")
    numberOfLambdas = len(list_of_lambdas)
    neededLambdas = 4
    if numberOfLambdas != neededLambdas:
        for i in range(neededLambdas - numberOfLambdas):
            list_of_lambdas.append(5.0)

    # Cost function
    for k in range(R):
        Q[k, k] = distances_matrix[k, k]

    # Constraints

    if active_constraints[0]:

        # Constraint 1: Total number of stops p+1 (p edges)
        for k in range(R):
            Q[k, k] += list_of_lambdas[0]*(1-2*p)
            for l in range(0, R):
                if l != k:
                    Q[k, l] +=  list_of_lambdas[0]

    if active_constraints[1]:

        # Constaint 2: Each stop has 0 or 2 edges if is not the start or end
        for i in range(1,N-1):
            for j in range(0, i):
                kji = convert_from_ij_to_k(j, i, N)
                Q[kji, kji] -= 2*list_of_lambdas[1]
                for l in range(0, i):
                    kli = convert_from_ij_to_k(l, i, N)
                    Q[kji, kli] += list_of_lambdas[1]
                for r in range(i+1, N):
                    kir = convert_from_ij_to_k(i, r, N)
                    Q[kji, kir] += list_of_lambdas[1]
            for k in range(i+1, N):
                kik = convert_from_ij_to_k(i,k,N)
                Q[kik, kik] -= 2*list_of_lambdas[1]
                for l in range(0, i):
                    kli = convert_from_ij_to_k(l, i, N)
                    Q[kik, kli] += list_of_lambdas[1]
                for r in range(i+1, N):
                    kir = convert_from_ij_to_k(i, r, N)
                    Q[kik, kir] += list_of_lambdas[1]

    if active_constraints[2]:
        # Constraint 3: Stop 0 has 1 edge
        for j in range(1,N):
            k0j = convert_from_ij_to_k(0, j, N)
            Q[k0j, k0j] -= list_of_lambdas[2]
            for l in range(1, N):
                if l != j:
                    k0l = convert_from_ij_to_k(0, l, N)
                    Q[k0j, k0l] += 2* list_of_lambdas[2]

    if active_constraints[3]:
        # Constraint 4: Stop N-1 has 1 edge
        for i in range(N-1):
            kiN1 = convert_from_ij_to_k(i, N-1, N)
            Q[kiN1, kiN1] -= list_of_lambdas[3]
            for l in range(N-1):
                if l != i:
                    klN1 = convert_from_ij_to_k(l, N-1, N)
                    Q[kiN1, klN1] += 2* list_of_lambdas[3]

    return Q, list_of_lambdas


# Checks

def calculate_cost(solution_array, Q_matrix):
    """
    Calculate the cost of a solution.
    It assumes that the Q_matrix and the array are in the QUBO representation.
    """

    return solution_array.T @ Q_matrix @ solution_array

def brute_force_finding(Q_matrix, distances_matrix, R):
    """
    Brute force method to find the optimal solution.
    It returns the bitstring, the cost with constraints, the adjacency matrix and the distances.
    """
    bitstrings = [np.binary_repr(i, R) for i in range(2 ** R)]
    costs = []
    # this takes exponential time with the dimension of the QUBO
    for b in bitstrings:
        z = np.array(list(b), dtype=int)
        cost = calculate_cost(z, Q_matrix)
        costs.append(cost)
    matrices = []
    for b in bitstrings:
        b_array = np.array(list(b), dtype=int)
        b_matrix = introduce_array_in_diagonal(b_array)
        matrices.append(convert_from_QUBO_to_adjacency(b_matrix))
    distances = []
    for b in bitstrings:
        b_array = np.array(list(b), dtype=int)
        distances.append(calculate_cost(b_array, distances_matrix))

    zipped = zip(bitstrings, costs, matrices, distances)
    sort_zipped = sorted(zipped, key=lambda x: x[1])
    return sort_zipped

def check_constraints(solution_array, p, R):
    """
    Check the constraints of the solution.
    """
    solution_matrix = introduce_array_in_diagonal(solution_array)
    adjacency_matrix = convert_from_QUBO_to_adjacency(solution_matrix)
    N = convert_from_R_to_N(R)
    constraint_1_check(solution_array, p)
    constraint_2_check(adjacency_matrix, N)
    constraint_3(adjacency_matrix, N)
    constraint_4(adjacency_matrix, N)

def constraint_1_check(solution_array, p):
    if np.sum(solution_array) != p:
        print("The lenght of the path is ", np.sum(solution_array), "but the p value is ", p)
        return False
    print("Constraint 1 passed: The lenght of the path is correct.")
    return True

def constraint_2_check(adjacency_matrix, N):
    for i in range(1,N-1):
        suma = 0
        for j in range(0, i):
            suma += adjacency_matrix[j, i]
        for j in range(i+1, N):
            suma += adjacency_matrix[i, j]
        if suma != 2 and suma != 0:
            print("The node ", i, " has ", suma, " connections")
            return False
    
    print("Constraint 2 passed: All nodes (apart from 0 and N-1) have 0 or 2 connections.")
    return True

def constraint_3(adjacency_matrix, N):
    suma = np.sum(adjacency_matrix[0])
    if suma > 1:
        print("The row 0 has more than one element")
        return False
    elif suma < 1:
        print("The row 0 has less than one element")
        return False
    print("Constraint 3 passed: The row 0 has exactly one element.")
    return True

def constraint_4(adjacency_matrix, N):
    suma = np.sum(adjacency_matrix[:, N-1])
    if suma > 1:
        print("The column N-1 has more than one element")
        return False
    elif suma < 1:
        print("The column N-1 has less than one element")
        return False
    print("Constraint 4 passed: The column N-1 has exactly one element.")
    return True


# Draw the graph
def draw_graph(adjacency_matrix, distance_matrix):
    """
    It assumes the adjacency matrix and the distance matrix are in the adjacency representation.
    """
    # Verificar si las matrices son cuadradas y del mismo tamaño
    adjacency_matrix = np.array(adjacency_matrix)
    distance_matrix = np.array(distance_matrix)
    N = adjacency_matrix.shape[0]
    if adjacency_matrix.shape[1] != N or distance_matrix.shape != (N, N):
        raise ValueError("Las matrices deben ser cuadradas y del mismo tamaño")
    
    # Crear el grafo
    G = nx.Graph()
    for i in range(N):
        G.add_node(i)
    
    # Agregar las aristas basadas en la matriz de adyacencia
    for i in range(N):
        for j in range(i + 1, N):  # Solo recorrer la parte superior para evitar duplicados en grafos no dirigidos
            if adjacency_matrix[i, j] != 0:
                G.add_edge(i, j, weight=distance_matrix[i, j])
    
    # Calcular la disposición de los nodos usando la matriz de distancias
    pos = nx.kamada_kawai_layout(G, weight='weight')
    
    # Dibujar el grafo
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=10)
    labels = {k: f"{v:.2f}" for k, v in nx.get_edge_attributes(G, 'weight').items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()