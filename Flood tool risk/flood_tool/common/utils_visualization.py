import pandas as pd
import numpy as np
from flood_tool.visualization.live import get_live_station_data
from flood_tool.geo import haversine_distance
import folium
from folium.plugins import HeatMap
import branca.colormap as cm


def format_data(data: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Merge predicitons with a mapping dataframe on postcode to map the post code
    to their latitude and longitude.

    :param data: DataFrame containing the predictions.
    :param mapping: DataFrame containing postcode mappings with
    latitude and longitude
    :return: Formatted DataFrame with postcode, latitude, longitude,
    risk label, and house price.
    """
    merged_data = pd.merge(data, mapping, on="postcode")
    return merged_data[
        ["postcode", "latitude", "longitude", "riskLabel", "house_price"]
    ]


def dummy_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add random risk labels and house prices to a dataset for
    simulation purposes.

    :param data: DataFrame to augment with dummy data.
    :return: DataFrame with added 'riskLabel' and 'house_price' columns.
    """
    data["riskLabel"] = np.random.randint(1, 8, size=len(data))
    data["house_price"] = np.random.randint(100000, 1000000, size=len(data))
    return data


def categorise_rainfall(dataframe: pd.DataFrame) -> str:
    """
    Categorize rainfall intensity based on hourly maximum rainfall values.

    :param dataframe: DataFrame containing 'dateTime' and 'value' columns
    for rainfall data.
    :return: String category ('slight', 'moderate', 'heavy', 'violent')
    or NaN for invalid data.
    """
    try:
        dataframe["dateTime"] = pd.to_datetime(
            dataframe["dateTime"], errors="coerce"
        )
        dataframe.set_index("dateTime", inplace=True)
        hourly_sum = dataframe["value"].resample("h").sum()
        max_rainfall = hourly_sum.max()

        if max_rainfall < 2:
            return "slight"
        elif max_rainfall < 5:
            return "moderate"
        elif max_rainfall < 50:
            return "heavy"
        elif max_rainfall >= 50:
            return "violent"
    except (KeyError, TypeError, ValueError):
        # Handle missing or invalid 'dateTime' column
        return np.nan


def categorise_rainfall_light(dataframe: pd.DataFrame) -> str:
    """
    Categorize rainfall intensity based on hourly maximum rainfall values.

    :param dataframe: DataFrame containing 'dateTime' and 'value'
    columns for rainfall data.
    :return: Hourly maximum value.
    """
    try:
        dataframe["dateTime"] = pd.to_datetime(
            dataframe["dateTime"], errors="coerce"
        )
        dataframe.set_index("dateTime", inplace=True)
        hourly_sum = dataframe["value"].resample("h").sum()
        max_rainfall = hourly_sum.max()
        return max_rainfall
    except (KeyError, TypeError, ValueError):
        # Handle missing or invalid 'dateTime' column
        return np.nan


def categorise_station(station_name: str) -> str:
    """
    Categorize rainfall intensity for a specific station using live data.

    :param station_name: Name of the station to fetch live data for.
    :return: String category of rainfall intensity for the station.
    """
    station_data = get_live_station_data(station_name)
    category = categorise_rainfall(station_data)
    return category


def format_rainfall(station_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Format station data with latitude, longitude, and categorized rainfall.

    :param station_dataframe: DataFrame containing station data including
    'stationReference'.
    :return: DataFrame with station reference, latitude, longitude,
    and rainfall category.
    """
    data = station_dataframe[["stationReference", "latitude", "longitude"]]
    data["category"] = data["stationReference"].map(categorise_station)
    return data


def high_risk_stations(stations_dataframe: pd.DataFrame, data) -> pd.DataFrame:
    """
    Identify and return high-risk stations near high-risk areas.

    :param stations_dataframe: DataFrame with station location data
    (latitude, longitude).
    :param data: DataFrame containing high-risk areas with 'latitude',
    'longitude', and 'riskLabel'.
    :return: Filtered DataFrame of high-risk stations.
    """
    index = []
    high_risk_data = data[data["riskLabel"] == 7]
    for i, row in high_risk_data.iterrows():

        distances = stations_dataframe.apply(
            lambda x: haversine_distance(
                x["latitude"],
                x["longitude"],
                row["latitude"],
                row["longitude"],
                deg=False,
            ),
            axis=1,
        )
        min_index = distances.idxmin()
        index.append(min_index)

    return stations_dataframe.iloc[index].drop_duplicates()


def format_latest_rainfall_readings(latest_rainfall, stations_dataframe):
    """
    Merge and format the latest rainfall readings with station data.

    :param latest_rainfall: DataFrame containing recent rainfall readings.
    :param stations_dataframe: DataFrame with station metadata including
    latitude and longitude.
    :return: DataFrame with station reference, coordinates, datetime,
    and rainfall value.
    """
    latest_rainfall = pd.merge(
        latest_rainfall, stations_dataframe, on="stationReference"
    )
    return latest_rainfall[
        "stationReference", "latitude", "longitude", "dateTime", "value"
    ]


def get_heavy_rainfall_data(latest_rainfall_reading, stations_dataframe):
    """
    Return a dataframe with the station reference, coordinates, and max
    hourly rainfall value.

    :param latest_rainfall_reading: DataFrame of recent rainfall readings with
    'stationReference'.
    :param stations_dataframe: DataFrame with station latitude and longitude.
    :return: DataFrame with station reference,
    coordinates, and categorized rainfall values.
    """
    latest_rainfall_reading = latest_rainfall_reading.reset_index()
    result = (
        latest_rainfall_reading.groupby("stationReference")
        .apply(categorise_rainfall_light)
        .reset_index(name="rainfall_category")
    )
    result = pd.merge(result, stations_dataframe, on="stationReference")
    return result[
        ["stationReference", "latitude", "longitude", "rainfall_category"]
    ]


def get_house_head(postcode, df):
    """
    Retrieve household and headcount data for a specific postcode sector.

    :param postcode: Full postcode string to query.
    :param df: DataFrame containing household and
    headcount data by postcode sector.
    :return: Tuple containing household and headcount values.
    """
    postcode_first = postcode.split()[0]  # first part
    postcode_second = postcode.split()[1][0]  # first character of the 2nd part
    postcodeSector = postcode_first + " " + postcode_second

    df = df[df["postcodeSector"] == postcodeSector]
    return df["households"], df["headcount"]


def Risk_HeatMap(df, sector_data):
    """
    Generate a heatmap of risk levels with markers for high-risk areas.

    :param df: DataFrame containing location and risk data
    (latitude, longitude, riskLabel).
    :param sector_data: DataFrame containing household and
    headcount data by postcode.
    :return: Folium map object displaying the heatmap and high-risk markers.
    """
    df["riskLabel"] = pd.to_numeric(df["riskLabel"], errors="coerce")

    m = folium.Map(
        location=[
            df["latitude"].mean(),
            df["longitude"].mean(),
        ],
        zoom_start=8,
    )

    # Add a heat map layer
    heat_data = [
        [row["latitude"], row["longitude"], row["riskLabel"]]
        for _, row in df.iterrows()
    ]
    HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)

    # Create a color map for the color bar
    colormap = cm.LinearColormap(
        colors=["blue", "green", "yellow", "red"],
        vmin=min(df["riskLabel"]),
        vmax=max(df["riskLabel"]),
        caption="Risk Level",
    )
    colormap.add_to(m)

    for _, row in df.iterrows():
        if row["riskLabel"] == 7:
            households, headcount = get_house_head(
                row["postcode"], sector_data
            )
            if households.empty or headcount.empty:
                folium.Marker(
                    location=[row["latitude"], row["longitude"]],
                    popup=f"Postcode: {row['postcode']}",
                    icon=folium.Icon(color="red", icon="info-sign"),
                ).add_to(m)
            else:
                folium.Marker(
                    location=[row["latitude"], row["longitude"]],
                    popup=(
                        f"Postcode: {row['postcode']}, "
                        f"Households: {households.values}, "
                        f"Headcount: {headcount.values}"
                    ),
                    icon=folium.Icon(color="red", icon="info-sign"),
                ).add_to(m)

    return m

def get_nearest_station(lat,long, stations_dataframe):
    distances = stations_dataframe.apply(
    lambda x: haversine_distance(
        x["latitude"],
        x["longitude"],
        lat,
        long,
        deg=False,
    ),
    axis=1,
)
    return stations_dataframe.iloc[distances.idxmin()].stationReference

def plot_flood_risk_map(df1, df2):
    """
    Plot an interactive map with flood risk markers and rainfall heatmap.
    
    Args:
        df1 (pd.DataFrame): DataFrame containing 'latitude', 'longitude', and 'riskLabel'.
        df2 (pd.DataFrame): DataFrame containing 'latitude', 'longitude', and 'value'.
    """
    center_lat = df1['latitude'].mean()
    center_lon = df1['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)

    for _, row in df1.iterrows():
        color = f"#{int((row['riskLabel'] - 1) * 255 / 6):02x}0000"  # Gradient from light to dark red
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Flood Risk: {row['riskLabel']}",
            icon=folium.Icon(color="white", icon_color=color, icon="info-sign")
        ).add_to(m)

    heat_data = [[row['latitude'], row['longitude'], row['rainfall_category']] for _, row in df2.iterrows()]
    HeatMap(heat_data, min_opacity=0.5, max_val=max(df2['rainfall_category']), radius=25, blur=15).add_to(m)

    # Render map in Streamlit
    return m