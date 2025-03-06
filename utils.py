from __future__ import annotations

from collections.abc import Iterable

import folium
import matplotlib.pyplot as plt
import numpy as np
import overpy
import pandas as pd

from linkageTree import linkageCut


def fetch_overpy_data(
    query: str | None = None,
):
    """
    Creates a overpy.Overpass and run a query to fetch amenities
    :param query: Specific overpy query. If none, one for Granada is used.
    :returns: overpy.Result
    """
    api = overpy.Overpass()
    # Selecting ALL amenities
    if not query:
        query = """
        [out:json];
        area[name=Granada][admin_level=8]->.granada;
        (
        node(area.granada)[amenity](37.120, -3.650, 37.300, -3.570);
        );
        out body;
        >;
        out skel qt;
        """
    response = api.query(query)
    return response


def fetch_amenities_from(query: str | None = None):
    """
    Fetch amenities from a certain location using overpy
    :param query: Specific overpy query. If none, one for Granada is used.
    :returns: pd.DataFrame
    """
    # Preparing the dataframe [id,latitude,longitude]
    df = pd.DataFrame(columns=["id", "lat", "lon"])
    response = fetch_overpy_data(query=query)
    for node in response.get_nodes():
        # Adding all the position information of nodes
        new_row = pd.DataFrame(
            {
                "id": node.id,
                "lat": node.lat,
                "lon": node.lon,
            },
            index=[0],
        )

        df = pd.concat([df, new_row], axis=0)

    # Formatted information into a DataFrame, only for convenience
    df.reset_index(inplace=True, drop=True)
    return df


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
        map = folium.Map(location=map_coords, zoom_start=12)

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
    zoom_start: int = 8,
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
    for center in centers:
        folium.Marker(center).add_to(map)
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
        colors = plt.color_sequences("Set1")
        if levels > len(colors):
            raise ValueError('You must specify the colors list if the number of levels > plt.color_sequences("Set1")')
    map = None
    for level in range(levels):
        centers = linkage_matrix.give_centers_level(0)
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
