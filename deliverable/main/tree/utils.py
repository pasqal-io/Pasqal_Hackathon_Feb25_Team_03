from __future__ import annotations

from collections.abc import Iterable

import folium
import numpy as np

from .linkageTree import linkageCut


def map_show_array(
    data: Iterable,
    labels: Iterable,
    map_coords: Iterable,
    color="red",
    map: folium.Map | None = None,
):
    """
    Uses folium library to draw a map in a specific coordenate (loc_coords, [latitude, longitude]).
    Draws specifics points given by data with their specific labels
    :param data: Iterable with N points with latitude, longitude.
    :param labels: Iterable with N points labels.
    :param loc_coords: Location coordenates of the main map.
    :param color: Color of the data points.
    :param map: Whether to add over a existing folium Map.
    """
    # Create a map centered on loc_coords [latitude, longitude]
    if not map:
        map = folium.Map(location=map_coords, zoom_start=13)

    # Loop through the data and add markers for each location
    for i in range(len(data)):
        folium.Marker(
            [data[i][1], data[i][0]],
            popup=labels[i],
            icon=folium.Icon(color=color),
        ).add_to(map)
    return map


def map_draw_line(
    centers: np.ndarray,
    line: np.ndarray,
    color: str = "red",
    map: folium.Map | None = None,
    zoom_start: int = 13,
    **map_kwargs,
):
    """
    Draws a list of centroids in a map and add lines between them given an adjacency matrix.
    :param centers: List of centroids (lat, lon)
    :params line: Adjacency matrix
    :params color: Folium color. Defaults to 'red'
    :param zoom_start: Default initial zoom for a new map instance. Only if map is None
    :param map: folium.map
    """
    means = centers.mean(axis=0)
    map_coords = [means[0], means[1]]
    if not map:
        map = folium.Map(location=map_coords, zoom_start=zoom_start, **map_kwargs)
    # Add all center points
    labels = range(1, len(centers) + 1)
    for i in range(len(centers)):
        folium.Marker(centers[i], popup=labels[i], icon=folium.Icon(color=color)).add_to(map)
    # Get all connected positions from line adj matrix
    nonzero = np.nonzero(line)
    for i in range(len(nonzero[0])):
        indx1 = nonzero[0][i]
        indx2 = nonzero[1][i]
        pos_1 = centers[indx1]
        pos_2 = centers[indx2]
        colorline = folium.features.PolyLine([pos_1, pos_2], color=color)
        colorline.add_to(map)
    return map


def draw_centers_on_map(centers: np.ndarray, **kwargs):
    """
    Auxiliar function that calculates map_coords, labels and returns a map_show_array
    :param centers: List of N points (lat, lon)
    :returns: folium.Map
    """
    means = centers.mean(axis=0)
    map_coords = [means[1], means[0]]
    labels = range(1, len(centers) + 1)
    map = map_show_array(centers, labels, map_coords, **kwargs)
    return map


def view_linkage_on_map(
    linkage_matrix: linkageCut,
    levels: int = 2,
    colors: list | None = None,
):
    """
    Given a hierarchical cluster from our linkageCut class, draws N levels using different colors
    :param linkage_matrix: LinkageCut class with hierarchical class clustering
    :param levels: Number of levels to draw.
    :param colors: List of colors for each level.
    """
    if not colors:
        # Folium.Icon allowed colors
        colors = [
            "red",
            "blue",
            "green",
            "purple",
            "orange",
            "darkred",
            "lightred",
            "beige",
            "darkblue",
            "darkgreen",
            "cadetblue",
            "darkpurple",
            "white",
            "pink",
            "lightblue",
            "lightgreen",
            "gray",
            "black",
            "lightgray",
        ]
        if levels > len(colors):
            raise ValueError('You must specify the colors list if the number of levels > plt.color_sequences("Set1")')
    map = None
    for level in range(levels):
        centers = linkage_matrix.give_centers_level(level)
        map = draw_centers_on_map(centers, color=colors[level], map=map)
    return map


def convert_bitstring_to_matrix(
    bitstring: str,
    N: int,
    p: int,
):
    """
    Given a dWave solution bitstring for the travelman sales problem (TSP), returns a adjacency matrix.
    :param bitstring: String with the solution using TSP format
    :param N: Number of nodes
    :param p: Number of expected stops.
    """
    adjacency = np.zeros((N, N))
    for k in range(p):
        for i in range(N):
            for j in range(N):
                if bitstring[i + N * k] == 1 and bitstring[j + N * (k + 1)] == 1:
                    adjacency[i, j] = 1
    return adjacency


def string_to_bitstring(string_sol):
    """Changing the format from string to list of bits

    >>> example: string_to_bitstring('01001') = [0 1 0 0 1]
    """
    return [int(x) for x in string_sol]


def assemble_line(level0_sols, level1_sols, nclusters, p):
    """Give a dictionary with {level-0 label:, connections, [startNode, endNode]}
    and return the fully assembled adjacency matrix

    WIP: Currently there is only support for two levels!!

    :return: full line adjacency matrix
    """
    adj_size = int(nclusters * nclusters)
    adj_matrix = np.zeros((adj_size, adj_size))

    # build basic adj matrix without connections
    for i in range(nclusters):
        adj_matrix[
            i * nclusters : (i + 1) * nclusters,
            i * nclusters : (i + 1) * nclusters,
        ] = convert_bitstring_to_matrix(level1_sols[i + 1][0], N=nclusters, p=p)

    # Now connect them all. Let's retrieve first the ordering of bus stops.
    level0_order = np.nonzero(level0_sols.reshape(p + 1, nclusters))[1] + 1

    # Do the first connection outside because its a special case where the actual 'end node' is in the position of start
    first_stop = level0_order[0]
    second_stop = level0_order[1]
    adj_matrix[
        (first_stop - 1) * nclusters + level1_sols[first_stop][1][0] - 1,
        (second_stop - 1) * nclusters + level1_sols[second_stop][1][0] - 1,
    ] = 1

    # this together with bus info is enough
    for this_stop, next_stop in zip(level0_order[1:-1], level0_order[2:]):
        adj_matrix[
            (this_stop - 1) * nclusters + level1_sols[this_stop][1][1] - 1,
            (next_stop - 1) * nclusters + level1_sols[next_stop][1][0] - 1,
        ] = 1

    return adj_matrix
