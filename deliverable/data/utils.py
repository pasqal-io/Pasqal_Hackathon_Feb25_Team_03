from __future__ import annotations

import overpy
import pandas as pd


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
