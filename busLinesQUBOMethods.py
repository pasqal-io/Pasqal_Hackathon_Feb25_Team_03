# from IPython.display import display, HTML
# display(HTML("<style>.container{width:100% !important;}</style>"))
from random import uniform, seed
from tabulate import tabulate
from numpy import argmax
import numpy as np
from dadk.QUBOSolverCPU import *
import matplotlib.pyplot as plt
import networkx as nx
from random import random
from typing import List


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
            if (
                i != j or not zeroSelfDistance
            ):  # Añadir aristas (i, i) si zeroSelfDistance es False
                G.add_edge(
                    i, j, weight=round(distances[i, j], 2)
                )  # Redondear a 2 decimales

    return G, distances


def draw_graph(G):
    pos = nx.spring_layout(
        G, seed=42
    )  # Posiciones de los nodos con un seed para reproducibilidad
    plt.figure(figsize=(10, 8))

    # Dibujar nodos
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=700)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    # Dibujar aristas con etiquetas
    curved_edges = [(u, v) for u, v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        edgelist=curved_edges,
        connectionstyle="arc3,rad=0.2",
        arrowsize=15,
        edge_color="black",
    )

    # Dibujar etiquetas de los pesos de las aristas más cerca de las aristas
    edge_labels = nx.get_edge_attributes(G, "weight")
    formatted_labels = {
        edge: f"{weight:.2f}" for edge, weight in edge_labels.items()
    }  # Formatear los pesos
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=formatted_labels,
        font_color="red",
        font_size=10,
        label_pos=0.6,
    )

    plt.title("Grafo Completo Dirigido con Pesos", fontsize=14)
    plt.show()


# Crear el QUBO para el problema de autobuses
def build_qubo(graph, distances, N: int, L: int, A: float, B: List[float]):
    var_shape_set = VarShapeSet(BitArrayShape("x", (N, N, L)))  #type: ignore
    BinPol.freeze_var_shape_set(var_shape_set)  #type: ignore

    # Función objetivo: minimizar distancias
    H_distances = BinPol()  #type: ignore
    for i in range(N):
        for j in range(N):
            for l in range(L):
                H_distances.add_term(distances[i, j], ("x", i, j, l))

    H_constraints = []
    # Restricciones:
     # 1. \sum_{l=0}^{L-1}\sum_{i=0}^{N-1} x_{ij}^{l} \geq 1 \forall j (toda parada tien al menos una salida)
    H_constraints_1 = BinPol()  #type: ignore
    for j in range(N):
        aux = BinPol()  #type: ignore
        for l in range(L):
            for i in range(N):
                aux.add_term(-1, ('x', i, j, l))
        aux.add_term(1, ())
        aux.power(2)
        H_constraints_1.add(aux)

    H_constraints.append(H_constraints_1)

    # 2. \sum_{l=0}^{L-1}\sum_{j=0}^{N-1} x_{ij}^{l} \geq 1 \forall i (toda parada tiene al menos una entrada)
    H_constraints_2 = BinPol()  #type: ignore
    for i in range(N):
        aux = BinPol()  #type: ignore
        for l in range(L):
            for j in range(N):
                aux.add_term(-1, ('x', i, j, l))
        aux.add_term(1, ())
        aux.power(2)
        H_constraints_2.add(aux)

    H_constraints.append(H_constraints_2)

    # 3. x_{ii}^{l} = 0 \forall i, l (una parada no puede ir a sí misma)
    H_constraints_3 = BinPol()  #type: ignore
    for l in range(L):
        for i in range(N):
            aux = BinPol()  #type: ignore
            aux.add_term(1, ("x", i, i, l))
            H_constraints_3.add(aux.power(2))

    H_constraints.append(H_constraints_3)

    # 4. if x_{ij}^{l} = 1 then \sum_{k=0}^{N-1} x_{jk}^{l} = 1 \forall i, j, l (si una linea llega a j, tiene que salir de j)
    H_constraints_4 = BinPol()  #type: ignore
    for l in range(L):
        for i in range(N):
            for j in range(N):
                aux = BinPol()  #type: ignore
                aux.add_term(1, ("x", i, j, l))  # x_{ij}^l
                for k in range(N):
                    aux.add_term(-1, ("x", j, k, l))  # -sum_k x_{jk}^l
                H_constraints_4.add(aux.power(2))  # Penalización cuadrática

    H_constraints.append(H_constraints_4)

    # 5. Cada linea debe ser cerrada y cubrir todos los nodos entre todas las paradas

    H_constraints_5_closed = BinPol()  #type: ignore
    for l in range(L):
        aux = BinPol()  #type: ignore
        for i in range(N):
            for j in range(N):
                aux.add_term(1, ("x", i, j, l))  # Contar aristas activas
        aux.add_term(-N, ())  # Penalización para ciclos abiertos
        H_constraints_5_closed.add(aux.power(2))

    H_constraints_5_cover = BinPol()  #type: ignore
    for j in range(N):
        aux = BinPol()  #type: ignore
        for l in range(L):
            for i in range(N):
                aux.add_term(
                    -1, ("x", i, j, l)
                )  # Garantizar al menos una visita por línea
        aux.add_term(1, ())
        H_constraints_5_cover.add(aux.power(2))

    H_constraints_5 = H_constraints_5_closed + H_constraints_5_cover

    H_constraints.append(H_constraints_5)

    # 6. Debe existir al menos un camino cerrado que pase por todos los nodos

    # Garantizar que el grafo resultante sea completamente conexo
    H_connectivity_strict = BinPol()  #type: ignore
    for i in range(N):
        for j in range(N):
            aux = BinPol()  #type: ignore
            for l in range(L):
                aux.add_term(1, ("x", i, j, l))
            aux.add_term(-1, ())  # Asegurar que cada nodo tenga al menos una conexión
            H_connectivity_strict.add(aux.power(2))

    # Asegurar que cada nodo esté en un ciclo accesible
    H_cycles = BinPol()  #type: ignore
    for i in range(N):
        aux = BinPol()  #type: ignore
        for j in range(N):
            for l in range(L):
                aux.add_term(1, ("x", i, j, l))
                aux.add_term(-1, ("x", j, i, l))  # Balance entre entrada y salida
        H_cycles.add(aux.power(2))

    # Agregar las nuevas restricciones con una penalización más alta
    H_constraints_6 = H_connectivity_strict + H_cycles

    H_constraints.append(H_constraints_6)

    H_constraints_sum = 0
    for i in range(len(H_constraints)):
        H_constraints_sum += B[i] * H_constraints[i]
    HQ = A * H_distances + H_constraints_sum
    return H_distances, H_constraints_sum, HQ


# Procesar la solución QUBO
def prep_bus_solution(HQ, H_distances, H_constraints, solution_list, N, L):
    solution = solution_list.min_solution
    edge_activations = solution["x"].data  # Variables binarias x_{ij}^l

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


def report_bus_solution(N, L, graph, active_edges):
    """
    Reporta la solución obtenida e indica si el grafo es factible o qué restricción falla.

    Parámetros:
        N (int): Número de nodos.
        L (int): Número de líneas de autobuses.
        graph (networkx.DiGraph): Grafo original con pesos en las aristas.
        active_edges (list): Lista de aristas activas en la solución obtenida [(i, j, l)].
    """
    is_feasible, message = check_graph_feasibility(N, L, graph, active_edges)

    print(
        (
            "Number of nodes:       {0:5d}\n"
            + "Number of lines:       {1:5d}\n"
            + "Number of edges:       {2:5d}\n"
            + "Active edges:          {3:5d}\n"
            + "Active edges details:  {4:s}\n"
            + "Feasibility Check:     {5:s}\n"
        ).format(N, L, len(graph.edges), len(active_edges), str(active_edges), message)
    )


def draw_bus_graph(graph, active_edges, colors, distances):
    """
    Dibuja el grafo de autobuses con las líneas activas, posicionando las paradas más cercanas según distances.

    Parámetros:
        graph (networkx.DiGraph): Grafo original con todas las conexiones.
        active_edges (list): Lista de aristas activas en la solución [(i, j, l)].
        colors (list): Lista de colores para cada línea.
        distances (np.array): Matriz NxN con las distancias entre paradas.
    """
    subgraph = nx.DiGraph()

    # Agregar aristas activas al subgrafo con pesos según distances[i, j]
    for i, j, l in active_edges:
        if graph.has_edge(i, j):
            subgraph.add_edge(
                i, j, color=colors[l], label=f"Line {l}", weight=distances[i, j]
            )

    # Obtener colores de las aristas
    edge_colors = [subgraph[u][v]["color"] for u, v in subgraph.edges]
    edge_labels = nx.get_edge_attributes(subgraph, "label")

    # Generar posiciones según las distancias d_{ij}
    pos = nx.kamada_kawai_layout(subgraph, weight="weight")

    # Dibujar el grafo con las líneas activas
    plt.figure(figsize=(10, 8))
    nx.draw(
        subgraph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=10,
        edge_color=edge_colors,
        arrows=True,
        arrowstyle="-|>",
        width=2,
    )
    nx.draw_networkx_edge_labels(
        subgraph, pos, edge_labels=edge_labels, font_color="black"
    )

    plt.title(
        "Grafo con líneas de autobús activas (Paradas más cercanas posicionadas juntas)"
    )
    plt.show()


def check_graph_feasibility(N, L, graph, active_edges):
    """
    Verifica si la solución obtenida cumple con todas las restricciones definidas en build_qubo.

    Parámetros:
        N (int): Número de nodos.
        L (int): Número de líneas de autobuses.
        graph (networkx.DiGraph): Grafo original con pesos en las aristas.
        active_edges (list): Lista de aristas activas en la solución obtenida [(i, j, l)].

    Retorna:
        (bool, str): Un booleano indicando si la solución es válida y un mensaje de error si no lo es.
    """
    subgraph = nx.DiGraph()
    subgraph.add_edges_from([(i, j) for i, j, l in active_edges])

    # 1. Verificar que cada nodo tenga al menos una salida
    for j in range(N):
        if not any(j == i for i, _, _ in active_edges):
            return False, f"Error: La parada {j} no tiene ninguna salida."

    # 2. Verificar que cada nodo tenga al menos una entrada
    for i in range(N):
        if not any(i == j for _, j, _ in active_edges):
            return False, f"Error: La parada {i} no tiene ninguna entrada."

    # 3. Verificar que no existan auto-bucles (x_{ii}^{l} = 0)
    for i, j, _ in active_edges:
        if i == j:
            return False, f"Error: Se encontró un auto-bucle en la parada {i}."

    # 4. Si un nodo tiene una entrada en una línea, debe tener una salida en esa línea
    for l in range(L):
        for j in range(N):
            incoming = [(i, j, l) for i in range(N) if (i, j, l) in active_edges]
            outgoing = [(j, k, l) for k in range(N) if (j, k, l) in active_edges]
            if incoming and not outgoing:
                return (
                    False,
                    f"Error: Nodo {j} recibe un autobús en la línea {l} pero no tiene salida.",
                )
            if outgoing and not incoming:
                return (
                    False,
                    f"Error: Nodo {j} tiene salida en la línea {l} pero no recibe ningún autobús.",
                )

    # 5. Verificar que el grafo resultante sea fuertemente conexo
    if not nx.is_strongly_connected(subgraph):
        return False, "Error: El grafo resultante no es fuertemente conexo."

    # 6. Verificar que cada línea individualmente sea fuertemente conexa en su subgrafo
    for l in range(L):
        line_subgraph = nx.DiGraph()
        line_edges = [(i, j) for i, j, l_aux in active_edges if l_aux == l]
        line_subgraph.add_edges_from(line_edges)

        # Si la línea no tiene aristas activas, es inválida
        if len(line_edges) == 0:
            return False, f"Error: La línea {l} no tiene ninguna arista activa."

        # Extraer los nodos que realmente están en la línea
        line_nodes = set(i for i, j in line_edges) | set(j for i, j in line_edges)

        # Crear subgrafo restringido solo a los nodos que aparecen en la línea
        restricted_subgraph = line_subgraph.subgraph(line_nodes)

        # Verificar que el subgrafo de la línea sea fuertemente conexo
        if not nx.is_strongly_connected(restricted_subgraph):
            return (
                False,
                f"Error: La línea {l} no es fuertemente conexa dentro de sus paradas.",
            )

    return True, "La solución cumple con todas las restricciones."


def average_distance_between_stops(distances, N):
    """
    Calcula la distancia media entre pares de paradas en un grafo dirigido.

    Parámetros:
        distances (np.array): Matriz NxN con las distancias entre paradas.
        N (int): Número de paradas.

    Retorna:
        float: Distancia media entre pares de paradas.
    """
    pairwise_means = []

    for i in range(N):
        for j in range(i + 1, N):  # Solo consideramos cada par una vez
            if i != j:
                mean_distance = (distances[i, j] + distances[j, i]) / 2
                pairwise_means.append(mean_distance)

    return np.mean(pairwise_means) if pairwise_means else 0.0  # Evitar divisiones por 0
