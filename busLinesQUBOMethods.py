from IPython.display import display, HTML
display(HTML("<style>.container{width:100% !important;}</style>"))
from random import uniform, seed
from tabulate import tabulate
from numpy import argmax
import numpy as np
from dadk.QUBOSolverCPU import *
import matplotlib.pyplot as plt
import networkx as nx
from random import random

def create_complete_graph(N: int, seed: int, zeroSelfDistance: bool = True):
    np.random.seed(seed)
    
    # Crear una matriz de distancias aleatorias (valores entre 1 y 10)
    distances = np.random.uniform(1, 10, size=(N, N))
    
    if zeroSelfDistance:
        np.fill_diagonal(distances, 0)  # Distancia cero para x_{ii}
    else:
        max_distance = distances.max()
        np.fill_diagonal(distances, max_distance)  # Distancia máxima para x_{ii}
    
    # Crear un grafo dirigido y asignar las distancias como pesos
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j or not zeroSelfDistance:  # Añadir aristas (i, i) si zeroSelfDistance es False
                G.add_edge(i, j, weight=round(distances[i, j], 2))  # Redondear a 2 decimales
    
    return G, distances

def draw_graph(G):
    pos = nx.spring_layout(G, seed=42)  # Posiciones de los nodos con un seed para reproducibilidad
    plt.figure(figsize=(10, 8))
    
    # Dibujar nodos
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # Dibujar aristas con etiquetas
    curved_edges = [(u, v) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, edgelist=curved_edges, connectionstyle="arc3,rad=0.2", arrowsize=15, edge_color='black')

    # Dibujar etiquetas de los pesos de las aristas más cerca de las aristas
    edge_labels = nx.get_edge_attributes(G, 'weight')
    formatted_labels = {edge: f"{weight:.2f}" for edge, weight in edge_labels.items()}  # Formatear los pesos
    nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_labels, font_color='red', font_size=10, label_pos=0.6)
    
    plt.title("Grafo Completo Dirigido con Pesos", fontsize=14)
    plt.show()


# Crear el QUBO para el problema de autobuses
def build_qubo(graph, distances, N, L, A, B):
    var_shape_set = VarShapeSet(BitArrayShape('x', (N, N, L)))
    BinPol.freeze_var_shape_set(var_shape_set)

    # Función objetivo: minimizar distancias
    H_distances = BinPol()
    for i in range(N):
        for j in range(N):
            for l in range(L):
                H_distances.add_term(distances[i, j], ('x', i, j, l))

    # Restricciones:
    # 1. \sum_{l=0}^{L-1}\sum_{i=0}^{N-1} x_{ij}^{l} \geq 1 \forall j
    H_constraints_1 = BinPol()
    for j in range(N):
        aux = BinPol()
        for l in range(L):
            for i in range(N):
                aux.add_term(-1, ('x', i, j, l))
        aux.add_term(1, ())
        aux.power(2)
        H_constraints_1.add(aux)

    # 2. \sum_{l=0}^{L-1}\sum_{j=0}^{N-1} x_{ij}^{l} \geq 1 \forall i
    H_constraints_2 = BinPol()
    for i in range(N):
        aux = BinPol()
        for l in range(L):
            for j in range(N):
                aux.add_term(-1, ('x', i, j, l))
        aux.add_term(1, ())
        aux.power(2)
        H_constraints_2.add(aux)

    # 3. x_{ii}^{l} = 0 \forall i, l
    H_constraints_3 = BinPol()
    for l in range(L):
        for i in range(N):
            aux = BinPol()
            aux.add_term(1, ('x', i, i, l))
            H_constraints_3.add(aux.power(2))

    # 4. if x_{ij}^{l} = 1 then \sum_{k=0}^{N-1} x_{jk}^{l} = 1 \forall i, j, l
    H_constraints_4 = BinPol()
    for l in range(L):
        for i in range(N):
            for j in range(N):
                aux = BinPol()
                aux.add_term(1, ('x', i, j, l))  # x_{ij}^l
                for k in range(N):
                    aux.add_term(-1, ('x', j, k, l))  # -sum_k x_{jk}^l
                H_constraints_4.add(aux.power(2))  # Penalización cuadrática

    # 5. Cada linea debe ser cerrada y cubrir todos los nodos

    H_constraints_5_closed = BinPol()
    for l in range(L):
        aux = BinPol()
        for i in range(N):
            for j in range(N):
                aux.add_term(1, ('x', i, j, l))  # Contar aristas activas
        aux.add_term(-N, ())  # Penalización para ciclos abiertos
        H_constraints_5_closed.add(aux.power(2))

    H_constraints_5_cover = BinPol()
    for j in range(N):
        aux = BinPol()
        for l in range(L):
            for i in range(N):
                aux.add_term(-1, ('x', i, j, l))  # Garantizar al menos una visita por línea
        aux.add_term(1, ())
        H_constraints_5_cover.add(aux.power(2))

    H_constraints_5 = H_constraints_5_closed + H_constraints_5_cover


    # 6. Debe existir al menos un camino cerrado que pase por todos los nodos

    H_combined_adjacency = BinPol()
    aux = BinPol()
    for i in range(N):
        for j in range(N):
            for l in range(L):
                aux.add_term(1, ('x', i, j, l))  # Suma sobre todas las líneas
    aux.add_term(-N, ())  # Garantizar que la suma total sea igual a N
    H_combined_adjacency.add(aux.power(2))  # Penalización cuadrática

    H_connectivity = BinPol()
    for i in range(N):
        aux = BinPol()
        for j in range(N):
            for l in range(L):
                aux.add_term(1, ('x', i, j, l))  # Salidas desde i
                aux.add_term(-1, ('x', j, i, l))  # Entradas hacia i
        H_connectivity.add(aux.power(2))  # Penalización cuadrática para balance


    H_constraints_6 = 5*(H_combined_adjacency + H_connectivity)



    H_constraints = H_constraints_1 + H_constraints_2 + H_constraints_3 + H_constraints_4 + H_constraints_5 + H_constraints_6
    HQ = A * H_distances + B * H_constraints
    return H_distances, H_constraints, HQ

# Procesar la solución QUBO
def prep_bus_solution(HQ, H_distances, H_constraints, solution_list, N, L):
    solution = solution_list.min_solution
    edge_activations = solution['x'].data  # Variables binarias x_{ij}^l

    colors = [(random(), random(), random()) for _ in range(L)]
    active_edges = []

    for i in range(N):
        for j in range(N):
            for l in range(L):
                if edge_activations[i][j][l]:
                    active_edges.append((i, j, l))

    print("HQ  = %10.6f" % (HQ.compute(solution.configuration)))
    print("H_distances = %10.6f" % (H_distances.compute(solution.configuration)))
    print("H_constraints = %10.6f" % (H_constraints.compute(solution.configuration)))

    return active_edges, colors

# Reporte de resultados
def report_bus_solution(N, L, graph, active_edges):
    print(("Number of nodes:       {0:5d}\n" +
           "Number of lines:       {1:5d}\n" +
           "Number of edges:       {2:5d}\n" +
           "Active edges:          {3:5d}\n" +
           "Active edges details:  {4:s}\n"
          ).format(N, L, len(graph.edges), len(active_edges), str(active_edges)))

# Dibujar el grafo con líneas activas
def draw_bus_graph(graph, active_edges, colors):
    subgraph = nx.DiGraph()
    for i, j, l in active_edges:
        if graph.has_edge(i, j):
            subgraph.add_edge(i, j, color=colors[l], label=f"Line {l}")

    edge_colors = [subgraph[u][v]['color'] for u, v in subgraph.edges]
    edge_labels = nx.get_edge_attributes(subgraph, 'label')

    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 8))
    nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, edge_color=edge_colors, arrows=True, arrowstyle='-|>', width=2)
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_color='black')
    plt.title("Grafo con líneas de autobús activas")
    plt.show()