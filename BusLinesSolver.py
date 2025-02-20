from typing import Optional, Tuple, List
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pyqubo
from pyqubo import Array, Placeholder, Constraint, Binary
from random import random
import neal


class Bus_lines_solver:


    def __init__(self, N: int, M: int, restrictions_multipliers: Optional[List[float]]=None, seed_number: Optional[int] = None, numberOfStopsPerLine: Optional[int]=None) -> None:
        self.N = N
        self.M = M
        if numberOfStopsPerLine is None:
            numberOfStopsPerLine = int(self.N / self.M)
        if restrictions_multipliers is None:
            restrictions_multipliers = [50.0 for i in range(5)]

        self.p = numberOfStopsPerLine
        self.restrictions_multipliers = restrictions_multipliers
        self.stops_graph, self.distances_matrix = self._create_complete_graph(seed=seed_number)
        self.QUBOmodel = self._create_QUBO()


    def _create_complete_graph(self, seed: Optional[int] = 42) -> Tuple[nx.Graph, np.ndarray]:
        np.random.seed(seed)

        distances = np.random.uniform(1, 10, size=(self.N, self.N))
        max_distance = distances.max()
        np.fill_diagonal(distances, max_distance)  # Maximum distance for x_{ii}

        G = nx.DiGraph()
        G.add_nodes_from(range(self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    G.add_edge(i, j, weight=round(distances[i, j], 2))

        return G, self.__expand_distance_matrix_to_diagonal(distances)

    def __expand_distance_matrix_to_diagonal(self, distances: np.ndarray) -> np.ndarray:
        """Converts an NxN matrix to an N^2xN^2 matrix with its elements on the diagonal."""
        expanded_distances = np.zeros((self.N ** 2, self.N ** 2))

        for i in range(self.N):
            for j in range(self.N):
                k = j + i * self.N
                expanded_distances[k, k] = distances[i, j]

        return expanded_distances

    def _create_QUBO(self):
        """Creates the QUBO model for the Bus Lines Problem"""

        connections = Array.create('item', shape=(self.N*self.N, self.M), vartype="BINARY")
        H_cost = sum(self.distances_matrix[i, i] * connections[i, l] for i in range(self.N * self.N) for l in range(self.M))
        H = H_cost

        list_of_restrictions = ["EquallyDistributed", 
                                "OneArrivalAtMost", 
                                "OneDepartureAtMost", 
                                "IfLineArrivesToStopItDepartsFromIt", 
                                "AllStopsAreVisited"]

        for index, restriction in enumerate(list_of_restrictions):
            restrictions_hamiltonian, lambda_expression= self._callRestrictionMethod(restriction, connections)
            H += lambda_expression*restrictions_hamiltonian

        return H.compile()

    def solve_QUBO(self):
        feed_dict = {
            "lmbEquallyDistributedRestriction": self.restrictions_multipliers[0],
            "lmbOneArrivalAtMostRestriction": self.restrictions_multipliers[1],
            "lmbOneDepartureAtMostRestriction": self.restrictions_multipliers[2],
            "lmbIfLineArrivesToStopItDepartsFromItRestriction": self.restrictions_multipliers[3],
            "lmbAllStopsAreVisitedRestriction": self.restrictions_multipliers[4]
        }
        qubo, offset = self.QUBOmodel.to_qubo(feed_dict=feed_dict)

        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample_qubo(qubo, num_reads=100)

        best_solution = sampleset.first.sample
        self.active_edges, self.colors = self._process_solution(best_solution)

        return self.active_edges

    def draw_bus_graph(self):
        """
            Draw the bus graph with the active lines, positioning the closest stops according to distances.
        """
        subgraph = nx.DiGraph()

        # Agregar aristas activas al subgrafo con pesos según distances[i, j]
        for i, j, l in self.active_edges:
            if self.stops_graph.has_edge(i, j):
                subgraph.add_edge(i, j, color=self.colors[l], label=f"Line {l}", weight=self.distances_matrix[self.N*i+j, self.N*i+j])

        # Obtener colores de las aristas
        edge_colors = [subgraph[u][v]['color'] for u, v in subgraph.edges]
        edge_labels = nx.get_edge_attributes(subgraph, 'label')

        # Generar posiciones según las distancias d_{ij}
        pos = nx.kamada_kawai_layout(subgraph, weight="weight")

        # Dibujar el grafo con las líneas activas
        fig = plt.figure(figsize=(10, 8))
        nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', node_size=500,
                font_size=10, edge_color=edge_colors, arrows=True, arrowstyle='-|>', width=2)
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_color='black')

        plt.title("Graph with active bus lines (Closest stops positioned together)")

        return fig

    def _process_solution(self, best_solution):
        """
        Reorganize PyQUBO's best_solution into the original edge_activations structure,
        ignoring slack variables.

        Parameters:
            best_solution (dict): Solution obtained from PyQUBO with keys in 'item[index][l]' format.
            N (int): Size of the adjacency matrix (NxN).
            L (int): Number of layers.

        Returns:
            list: List of active edges in format (i, j, l).
            list: List of random colors for each layer.
        """
        edge_activations = np.zeros((self.N, self.N, self.M), dtype=int)

        for key, value in best_solution.items():
            if value == 1 and key.startswith("item"):
                indices = key[key.find("[") + 1:key.rfind("]")].split("][")
                if len(indices) != 2:
                    continue

                index, l = map(int, indices)
                i, j = divmod(index, self.N)
                edge_activations[i, j, l] = 1

        active_edges = [(i, j, l) for i in range(self.N) for j in range(self.N) for l in range(self.M) if
                        edge_activations[i, j, l] == 1]

        colors = [(random(), random(), random()) for _ in range(self.M)]

        return active_edges, colors

    def _callRestrictionMethod(self, restrictionName: str, connections: pyqubo.Array):
        if restrictionName == "EquallyDistributed":
            return self.__stopsEquallyDistributedRestriction(connections)

        elif restrictionName == "OneArrivalAtMost":
            return self.__oneArrivalAtMostRestriction(connections)

        elif restrictionName == "OneDepartureAtMost":
            return self.__oneDepartureAtMostRestriction(connections)

        elif restrictionName == "ClosedPathsOfLenghtP":
            return self.__closedPathsOfLenghtPRestriction(connections)

        elif restrictionName == "NoClosedPathsOfLenghtLessThanP":
            return self.__noClosedPathsOfLenghtLessThanPRestriction(connections)
        
        elif restrictionName == "GlobalPathOfLenghtN":
            return self.__globalPathOfLenghtNRestriction(connections)
        
        elif restrictionName == "IfLineArrivesToStopItDepartsFromIt":
            return self.__ifLineArrivesToStopItDepartsFromIt(connections)
        
        elif restrictionName == "AllStopsAreVisited":
            return self.__allStopsAreVisited(connections)

        else:
            print("Restriction method not recognized")

    def __stopsEquallyDistributedRestriction(self, connections: pyqubo.Array):
        H = 0
        lmb = Placeholder("lmbEquallyDistributedRestriction")
        for l in range(self.M):
            terms = sum(connections[i, l] for i in range(self.N * self.N))
            H += Constraint((terms - self.p) ** 2, label=f"EquallyDistributedRestriction_line{l}")

        return H, lmb

    def __oneArrivalAtMostRestriction(self, connections: pyqubo.Array):
        s2 = {(l, i): Binary(f"s2_{l}_{i}") for l in range(self.M) for i in range(self.N)}
        H2 = 0
        lmb2 = Placeholder("lmbOneArrivalAtMostRestriction")
        for l in range(self.M):
            for i in range(self.N):
                terms2 = sum(connections[i * self.N + j, l] for j in range(self.N))
                H2 += Constraint((s2[l, i] + terms2 - 1) ** 2, label=f"OneArrivalAtMostRestriction_line{l}_stop{i}")

        return H2, lmb2

    def __oneDepartureAtMostRestriction(self, connections: pyqubo.Array):
        s3 = {(l, i): Binary(f"s3_{l}_{i}") for l in range(self.M) for i in range(self.N)}
        H3 = 0
        lmb3 = Placeholder("lmbOneDepartureAtMostRestriction")
        for l in range(self.M):
            for i in range(self.N):
                terms3 = sum(connections[j * self.N + i, l] for j in range(self.N))
                H3 += Constraint((s3[l, i] + terms3 - 1) ** 2, label=f"OneDepartureAtMostRestriction_line{l}_stop{i}")

        return H3, lmb3

    def __closedPathsOfLenghtPRestriction(self, connections: pyqubo.Array):
        H4 = 0
        lmb4 = Placeholder("lmbClosedPathsOfLenghtPRestriction")
        for l in range(self.M):
            terms4 = sum(self._apply_power_with_reshape(connections[:, l], self.p)[k * (self.N + 1)] for k in range(self.N))
            H4 += Constraint((terms4 - self.p) ** 2, label=f"ClosedPathsOfLenghtPRestriction_line{l}")

        return H4, lmb4

    def __noClosedPathsOfLenghtLessThanPRestriction(self, connections: pyqubo.Array):
        H5 = 0
        lmb5 = Placeholder("lmbNoClosedPathsOfLenghtLessThanPRestriction")
        for l in range(self.M):
            terms5 = sum(self._apply_power_with_reshape(connections[:, l], n)[k * (self.N + 1)] for k in range(self.N) for n in
                         range(self.p))
            H5 += Constraint(terms5 ** 2, label=f"NoClosedPathsOfLenghtLessThanPRestriccion_line{l}")

        return H5, lmb5
    
    def __globalPathOfLenghtNRestriction(self, connections: pyqubo.Array):
        H6 = 0
        lmb6 = Placeholder("lmbGlobalPathOfLenghtN")
        matrixTerms = sum(connections[:, l] for l in range(self.M))
        terms = sum(self._apply_power_with_reshape(matrixTerms, self.N)[k * (self.N + 1)] for k in range(self.N))
        H6 += Constraint((terms - self.N) ** 2, label="GlobalPathOfLenghtNRestriction")

        return H6, lmb6
    
    def __ifLineArrivesToStopItDepartsFromIt(self, connections: pyqubo.Array):
        H7 = 0
        lmb7 = Placeholder("lmbIfLineArrivesToStopItDepartsFromItRestriction")
        for l in range(self.M):
            for i in range(self.N):
                for j in range (self.N):
                    terms7 = sum(connections[j*self.N+k, l] for k in range(self.N))
                    H7 += Constraint((connections[i*self.N+j, l] - terms7) ** 2, label=f"IfLineArrivesToStopItDepartsFromItRestriction_line{l}_stop{i}_stop{j}")

        return H7, lmb7
    
    def __allStopsAreVisited(self, connections: pyqubo.Array):
        s41 = {(j): Binary(f"s8_{j}")for j in range(self.N)}
        s42 = {(j): Binary(f"s8_{j}")for j in range(self.N)}
        H8 = 0
        lmb8 = Placeholder("lmbAllStopsAreVisitedRestriction")
        for j in range(self.N):
            terms8 = sum(connections[i*self.N+j, l] for i in range(self.N) for l in range(self.M))
            H8 += Constraint((terms8 - 1 -s41[j]-2*s42[j]) ** 2, label=f"AllStopsAreVisitedRestriction_stop{j}")

        return H8, lmb8


    def _apply_power_with_reshape(self, Y: pyqubo.Array, p: int) -> pyqubo.Array:
        """
        Converts Y (a pyqubo.Array object) into a square matrix X, raises it to the power p, 
        and returns the result as a flattened vector.
        """

        X = Y.reshape((self.N, self.N))

        for _ in range(p - 1):  # p - 1 because we already have the first X
            X = X.matmul(Y.reshape((self.N, self.N)))

        # Step 3: Reshape back into a vector of N^2 components and return
        return X.reshape((self.N * self.N,))





    def draw_stops_graph(self) -> plt.Figure:
        pos = nx.spring_layout(self.stops_graph, seed=42)
        fig = plt.figure(figsize=(10, 8))

        nx.draw_networkx_nodes(self.stops_graph, pos, node_color='lightblue', node_size=700)
        nx.draw_networkx_labels(self.stops_graph, pos, font_size=12, font_weight='bold')

        curved_edges = [(u, v) for u, v in self.stops_graph.edges()]
        nx.draw_networkx_edges(self.stops_graph, pos, edgelist=curved_edges, connectionstyle="arc3,rad=0.2", arrowsize=15,
                               edge_color='black')

        edge_labels = nx.get_edge_attributes(self.stops_graph, 'weight')
        formatted_labels = {edge: f"{weight:.2f}" for edge, weight in edge_labels.items()}
        nx.draw_networkx_edge_labels(self.stops_graph, pos, edge_labels=formatted_labels, font_color='red', font_size=10,
                                     label_pos=0.6)

        plt.title("Bus stops with associated distances", fontsize=14)

        return fig

    def average_distance_between_stops(self):
        """
        Calculates the average distance between pairs of stops in a directed graph.

        Returns:
            float: Average distance between pairs of stops.
        """
        pairwise_means = []

        for i in range(self.N):
            for j in range(i + 1, self.N):
                mean_distance = (self.distances_matrix[i*self.N+j, i*self.N+j] + self.distances_matrix[i+self.N*j, i+self.N*j]) / 2
                pairwise_means.append(mean_distance)

        return np.mean(pairwise_means) if pairwise_means else 0.0