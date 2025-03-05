import requests
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import linkage, cut_tree


class linkageCut:
    """
    This class is in charge of managing the hierarchical clusters of a city (amenities grouping).

    - Receives a dataframe with longituds (lon) and latitudes (lat);

    Examples
    --------
    >>> from linkageTree import linkageCut
    ...
    >>> df = pd.DataFrame(data, columns=["id","lat","lon"])
    >>> linkage_matrix = linkageCut(df)
    >>> from sklearn.base import BaseEstimator, TransformerMixin
    >>> nclusters = 6
    >>> levels = 2
    >>> # We hierarchically distribute all points
    >>> top_down = linkage_matrix.top_down_view_recur(nclusters, levels)
    >>> X = linkage_matrix.data
    >>> fig, ax = plt.subplots()
    >>> agg_labels = top_down[:, 0] # level 0
    >>> ax.scatter(X.T[0], X.T[1], c=agg_labels)
    >>> ax.set_title(f'Ward clustering, level 0: {nclusters} clusters ')
    >>> plt.show()
    """

    def __init__(self, df):
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(df[["lon", "lat"]].values)
        self.data_lon_lat = df[["lon", "lat"]].values
        self.linkage_matrix = linkage(self.data, method="ward")
        self.tree_cut = cut_tree(self.linkage_matrix)

        self.top_down = None
        self.distance_matrix = None
        self.nclusters = None

    def give_tree_cut(self):
        return self.tree_cut

    def __nunique(self, a, axis):
        """Count the number of unique elements in an array and axis"""
        return (np.diff(np.sort(a, axis=axis), axis=axis) != 0).sum(axis=axis) + 1

    def __recursive_down(self, nclusters, level, total_levels, mask):
        if len(str(level)) < total_levels:

            # Selecting specific parent cluster
            small_tree = self.tree_cut[mask]

            # Checking how many subclusters there are in the parent cluster for each step in
            # the clustering (ward) process
            small_tree_nclusters = self.__nunique(small_tree, axis=0)

            try:
                # This is the step where there are exactly nclusters subclusters
                sub_tree_step = np.where(small_tree_nclusters == nclusters)[0][-1]
            except:
                raise Exception("Some cluster cannot be further divided")
            # Now we truly have the subdivision

            counter = 1
            for sub_lbl in np.unique(small_tree[:, sub_tree_step]):
                # Now we prepare the mask for each subcluster and the recursion
                sub_data_mask = np.where(self.tree_cut[:, sub_tree_step] == sub_lbl)[0]
                write = self.__recursive_down(
                    nclusters,
                    int(str(level) + str(counter)),
                    total_levels,
                    sub_data_mask,
                )
                if write == None:
                    write = str(level) + str(counter)
                self.top_down[sub_data_mask, len(str(level))] = int(write)
                counter += 1

            return None
        else:
            return level

    def top_down_view_recur(self, nclusters, levels=1):
        """Constructs a top down view in which each cluster is
        subsequently divided in 'nclusters'. This process
        is iterated 'level' times.
        In each step, all data is labelled accordingly.

        The naming conventions for the labels is 1,2, ..., nclusters for
        the first level, 11 for the first subcluster of cluster 1 and so on. Example:

        132 has 3 levels, in the order 1(top)-3(middle)-2(lowest)

        Obviously, the maximum nclusters is 9 for our proof of concept, for larger values,
        char implementations should be considered

        :input nclusters: number of clusters per level
        :input levels: number of layers or levels
        :return: (len_data)X(level) matrix with labels
        """
        self.nclusters = nclusters
        # First level
        if levels < 1:
            raise Exception("levels must be >= 1")

        self.top_down = np.zeros((len(self.data), levels))

        # The first level is, by definition, in the tree_cut indx
        # len_data - nclusters
        tree_step = len(self.tree_cut) - nclusters
        first_lbls = np.unique(self.tree_cut[:, tree_step])

        # Second level
        for lbl in first_lbls:
            data_mask = np.where(self.tree_cut[:, tree_step] == lbl)[0]
            self.top_down[data_mask, 0] = lbl + 1
            self.__recursive_down(nclusters, lbl + 1, levels, data_mask)

        return self.top_down

    def give_center_label(self, label):
        """Returns the positions in lon-lat space of the centers with a given label.

        Examples:
        give_centers_label(11) returns the centroid "11"
        """
        level = len(str(int(label))) - 1
        sub_top_down = self.top_down[:, level]
        cluster_mask = sub_top_down == label

        center = np.mean(self.data[cluster_mask], axis=0)
        # Find the closest location
        center = self.data[cluster_mask][
            np.argmin(
                euclidean_distances(self.data[cluster_mask], center.reshape(1, -1))
            )
        ]

        return center

    def give_centers_level(self, level):
        """Return the possible location closer to the cluster centroid in a certain level"""

        if type(self.top_down) != np.ndarray:
            raise StopIteration("You must first execute top_down_view_recur first")
        else:
            if level > self.top_down.shape[1]:
                raise IndexError("Level out of bounds")
            else:
                # Calculate the centers in normalized space and return to lon-lat
                sub_top_down = self.top_down[:, level]
                level_labels = np.unique(sub_top_down)
                centers = np.zeros((len(level_labels), 2))

                for i in range(len(level_labels)):
                    centers[i] = self.give_center_label(level_labels[i])

                centers = self.scaler.inverse_transform(centers)
            return centers

    def give_centers_label_down(self, label):
        """Return the centers locations from a label down in order: 11, 12, 13, 14
        Example: give_centers_label_down(2) returns the position of stops corresponding to 21, 22, 23,...
        """
        level = len(str(int(label))) - 1
        if type(self.top_down) != np.ndarray:
            raise StopIteration("You must first execute top_down_view_recur first")
        else:
            if level > self.top_down.shape[1]:
                raise IndexError("Level out of bounds")
            else:
                # Calculate the centers in normalized space and return to lon-lat
                sub_top_down = self.top_down[:, level + 1]

                # Checking upper level
                sub_top_down = sub_top_down[(sub_top_down / 10).astype(int) == label]
                level_labels = np.unique(sub_top_down)
                centers = np.zeros((self.nclusters, 2))

                for i in range(len(level_labels)):
                    centers[i] = self.give_center_label(level_labels[i])

                centers = self.scaler.inverse_transform(centers)
            return centers

    def __OSRM_query(self, coords, sources=None, destinations=None):
        url = "http://router.project-osrm.org/table/v1/driving/"
        routes = ""

        # Query with longitude1,latitude1;longitude2,latitude2;...
        for cl_ in coords:
            routes += str(cl_[0]) + "," + str(cl_[1]) + ";"
        routes = routes[:-1]
        dir_query = url + routes + "?annotations=distance"

        if (sources == None) & (destinations == None):
            pass
        else:
            if sources != None:
                dir_query += "&sources="
                for indx in sources:
                    dir_query += str(indx) + ";"
                dir_query = dir_query[:-1]
            if destinations != None:
                dir_query += "&destinations="
                for indx in destinations:
                    dir_query += str(indx) + ";"
                dir_query = dir_query[:-1]

        routes_response = requests.get(dir_query)
        dist_table_json = routes_response.json()
        dist_matrix = np.array(dist_table_json["distances"]) / 1000  # meters to km
        return dist_matrix

    def dist_matrix_level(self, level, return_labels=True):
        """Returns the distance matrix for all center clusters in a given level"""
        centers_obj = self.give_centers_level(level)
        dist_matrix = self.__OSRM_query(centers_obj)
        dist_matrix += dist_matrix.T
        ind_labels = np.array(range(1, len(centers_obj) + 1))

        if return_labels:
            return dist_matrix, ind_labels
        else:
            return dist_matrix

    def dist_matrix_label_down(self, label, connections=[], return_labels=True):
        """Returns the distance matrix of a single cluster divided under a certain level
        given by the label returned by top_down_view_recur, with the
        connected nodes (if there are any), in the first and last position of the array.
        The connections must be the other clusters connected in the upper level.

        Cases
        (i) No assumptions about higher level connection (connected_clusters=None). Returns the distance array
        without any particular ordering
        (ii) Only one element in connected_cluster. It is placed in the first index of the distance array
        (iii) Two elements in connected_cluster. They are placed in the first and last indices of the distance array

        Examples:

        dist_array(1, connections = [2,3]) would assign the closest stops from 1 (from 11, 12, ...) to
        2 and 3 (21,22,23... and 31,32,33...) as first and last stops to ensure a smooth 2-1-3 route.

        :return:
        """

        clusters_obj = self.give_centers_label_down(label)
        ind_labels = np.array(range(1, len(clusters_obj) + 1))

        if len(connections) == 0:
            return None
        else:
            clusters_conn_1 = self.give_centers_label_down(connections[0])
            coords_obj1 = np.append(clusters_obj, clusters_conn_1, axis=0)
            ins = range(len(clusters_obj))
            outs = range(len(clusters_obj), len(clusters_obj) + len(clusters_conn_1))

            # Bidirectional matrix
            dist_obj1 = self.__OSRM_query(coords_obj1, sources=ins, destinations=outs)
            dist_obj1 += self.__OSRM_query(
                coords_obj1, sources=outs, destinations=ins
            ).T

            if len(connections) == 2:
                clusters_conn_2 = self.give_centers_label_down(connections[0])
                coords_obj2 = np.append(clusters_obj, clusters_conn_2, axis=0)
                ins = range(len(clusters_obj))
                outs = range(
                    len(clusters_obj), len(clusters_obj) + len(clusters_conn_2)
                )

                # Bidirectional matrix
                dist_obj2 = self.__OSRM_query(
                    coords_obj2, sources=ins, destinations=outs
                )
                dist_obj2 += self.__OSRM_query(
                    coords_obj2, sources=outs, destinations=ins
                )
                min_dist_indx_2 = np.unravel_index(
                    np.argmin(dist_obj2), dist_obj2.shape
                )[0]

                # Swaping places

                ind_labels[[min_dist_indx_2, -1]] = ind_labels[[-1, min_dist_indx_2]]
                clusters_obj[[min_dist_indx_2, -1]] = clusters_obj[
                    [-1, min_dist_indx_2]
                ]

            min_dist_indx_1 = np.unravel_index(np.argmin(dist_obj1), dist_obj1.shape)[0]
            # Swapping places
            ind_labels[[min_dist_indx_1, 0]] = ind_labels[[0, min_dist_indx_1]]
            clusters_obj[[min_dist_indx_1, 0]] = clusters_obj[[0, min_dist_indx_1]]

        dist_matrix = self.__OSRM_query(clusters_obj)
        dist_matrix += dist_matrix.T

        ind_labels += 10 * label  # Adding prefix to label
        if return_labels:
            return dist_matrix, ind_labels
        else:
            return dist_matrix


def main(args):
    datapath = args.data
    nclusters = args.nclusters
    levels = args.levels

    df = pd.read_csv(datapath)

    linkage_matrix = linkageCut(df)
    # fancy scatter
    fig, ax = plt.subplots()
    fig.set_size_inches((5, 5))
    X = np.array(linkage_matrix.data_lon_lat, dtype=np.float64)

    top_down = linkage_matrix.top_down_view_recur(nclusters, levels)
    ax.scatter(X[:, 0], X[:, 1], c=top_down[:, 1] % 10, cmap="Accent")
    alpha_list = np.ones(nclusters) * 100

    for i in range(1, nclusters + 1):
        cluster = X[top_down[:, 0] == i]
        alpha = alpha_list[i - 1]
        hull = alphashape.alphashape(cluster, alpha)
        if type(hull) == shapely.geometry.multipolygon.MultiPolygon:
            areas = [geom.area for geom in hull.geoms]
            # Select the component with larger area
            big = np.argmax(areas)
            hull_pts = hull.geoms[big].exterior.coords.xy
        else:
            hull_pts = hull.exterior.coords.xy
        poly_patch = Polygon(np.array(hull_pts).T, facecolor="none", edgecolor="red")
        ax.add_patch(poly_patch)

    centers = linkage_matrix.give_centers_level(0)
    plt.scatter(
        *centers.T,
        marker="X",
        edgecolors="black",
        color="red",
        linewidth=0.5,
        label="level 0"
    )
    centers = linkage_matrix.give_centers_level(1)
    plt.scatter(
        *centers.T, marker="X", edgecolors="black", linewidth=0.5, label="level 1"
    )
    fig.suptitle("Hierarchical division")
    plt.xlabel(r"lon (deg)")
    plt.ylabel(r"lat (deg)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    import argparse
    import alphashape
    import shapely
    from matplotlib.patches import Polygon

    parser = argparse.ArgumentParser(
        prog="LinkageCut Class",
        description="Allows for hierarchical clustering with top-down transversal",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(os.pardir, "data", "amenities-granada.csv"),
    )
    parser.add_argument("--nclusters", type=int, default=6)
    parser.add_argument("--levels", type=int, default=2)
    args = parser.parse_args()
    main(args)
