def map_show_array(data, labels, loc_coords, color = 'red', map=None):
    import folium
    # Create a map centered on loc_coords [latitude, longitude]
    if map == None:
        map = folium.Map(location=loc_coords, zoom_start=12)

    # Loop through the data and add markers for each location
    for i in range(len(data)):
        folium.Marker(
            [data[i][1], data[i][0]],
            popup=labels[i],
            icon=folium.Icon(color=color) 
        ).add_to(map)
    return map

def map_draw_line(centers, line, color, map=None):
    import folium
    import numpy as np
    means = centers.mean(axis=0)
    loc_coords = [means[0], means[1]]
    if map == None:
        map = folium.Map(location=loc_coords, zoom_start=8)
    # Get all connected positions from line adj matrix
    for center in centers:
        folium.Marker(center).add_to(map)
    nonzero = np.nonzero(line)
    for i in range(len(nonzero[0])):
        indx1 = nonzero[0][i]
        indx2 = nonzero[1][i]
        pos_1 = centers[indx1]
        pos_2 = centers[indx2]
        colorline = folium.features.PolyLine([pos_1,pos_2], color=color)
        colorline.add_to(map)
    return map


def draw_centers_on_map(centers, **kwargs):
    means = centers.mean(axis=0)
    means_lat_lon = [means[1], means[0]]
    labels_0 = range(1, len(centers) + 1)
    map = map_show_array(centers, labels_0, means_lat_lon, **kwargs)
    return map


def view_linkage_on_map(linkage_matrix):
    centers_0 =  linkage_matrix.give_centers_level(0)
    map = draw_centers_on_map(centers_0, color='red')

    centers_1 =  linkage_matrix.give_centers_level(1)
    map = draw_centers_on_map(centers_1, color='blue', map = map)
    return map


def convert_bitstring_to_matrix(bitstring, N: int, p: int):
    import numpy as np
    adjacency = np.zeros((N, N))
    for k in range(p):
        for i in range(N):
            for j in range(N):
                if bitstring[i + N*k] == 1 and bitstring[j + N*(k+1)] == 1:
                    adjacency[i,j] = 1
    return adjacency