import folium

all = ['plot_circle']


def plot_circle(lat, lon, radius, map=None, **kwargs):
    '''
    Plot a circle on a map (creating a new folium map instance if necessary).

    Parameters
    ----------

    lat: float
        latitude of circle to plot (degrees)
    lon: float
        longitude of circle to plot (degrees)
    radius: float
        radius of circle to plot (m)
    map: folium.Map
        existing map object

    Returns
    -------

    Folium map object

    Examples
    --------

    >>> import folium
    >>> plot_circle(52.79, -2.95, 1e3, map=None) # doctest: +SKIP
    '''

    if not map:
        map = folium.Map(location=[lat, lon], control_scale=True)

    folium.Circle(
        location=[lat, lon],
        radius=radius,
        fill=True,
        fillOpacity=0.6,
        **kwargs,
    ).add_to(map)

    return map


import pandas as pd
from folium.plugins import HeatMap
import branca.colormap as cm


def Risk_HeatMap(file_path):
    """
    Create a heat map of the risk levels in the given file.
    """

    df = pd.read_csv(file_path)
    df['riskLabel'] = pd.to_numeric(df['riskLabel'], errors='coerce')

    m = folium.Map(location=[51.5074, -1], zoom_start=10)

    # Add a heat map layer
    heat_data = [[row['Latitude'], row['Longitude'], row['riskLabel']] for _, row in df.iterrows()]
    HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)

    # Create a color map for the color bar
    colormap = cm.LinearColormap(colors=['blue', 'green', 'yellow', 'red'], 
                                vmin=min(df['riskLabel']), 
                                vmax=max(df['riskLabel']),
                                caption='Risk Level')
    colormap.add_to(m)

    for _, row in df.iterrows():
        if row["riskLabel"] == 7:
            households, headcount = get_house_head(row["postcode"])
            if households.empty or headcount.empty:
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=f"Postcode: {row['postcode']}",
                    icon=folium.Icon(color='red', icon='info-sign')).add_to(m)
            else:
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=f"Postcode: {row['postcode']}, Households: {households.values}, Headcount: {headcount.values}",
                    icon=folium.Icon(color='red', icon='info-sign')).add_to(m)

    # Add a custom legend to the map
    legend_html = """
    <div style="
        position: fixed;
        bottom: 20px; left: 20px; width: 200px; height: 90px;
        background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
        padding: 10px;">
        <strong>Info-sign Marker:</strong><br>
        Most Dangerous Area (Risk:7)
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def get_house_head(postcode):
    """
    [tool function]
    This should be a tool function, so I'm not sure it should be here.
    so if it is a tool function, I'll mark it with: -> [tool function]
    """

    df = pd.read_csv("../resources/sector_data.csv")

    postcode_first = postcode.split()[0]  # first part
    postcode_second = postcode.split()[1][0]  # first character of the second part
    postcodeSector = postcode_first + " " + postcode_second

    df = df[df["postcodeSector"] == postcodeSector]
    return df[["households"]], df[["headcount"]]


def normalise_rainfall(df):
    """
    [tool function]
    Normalize rainfall values to meters/mAOD/mASD if unit is mm.
    """

    df.loc[df["unitName"] != "mm", "value"] *= 1000
    df.loc[df["unitName"] != "mm", "unitName"] = "mm"
    return df


def Rainfall_merged_data(station_name):
    """ 
    [tool function]
    Merge the station data with the rainfall data in typical day and wet day based on the station name.
    """

    # read the station data
    station_df = pd.read_csv("../resources/stations.csv")
    station_info = station_df[station_df["stationName"] == station_name][['stationReference', 'latitude', 'longitude']]
    
    if station_info.empty:
        print(f"No station found with name: {station_name}")
        return None
    
    # read the rainfall data
    typical_data = pd.read_csv("../example_data/typical_day.csv")
    wet_data = pd.read_csv("../example_data/wet_day.csv")
    typical_data['value'] = pd.to_numeric(typical_data['value'], errors='coerce')
    wet_data['value'] = pd.to_numeric(wet_data['value'], errors='coerce')

    # normalise the rainfall data
    typical_data = normalise_rainfall(typical_data)
    wet_data = normalise_rainfall(wet_data)

    # merge the station data with the rainfall data
    merged_data = pd.merge(
        station_info,
        typical_data[["stationReference", "parameter", "value"]],
        on="stationReference",
        how="left"
    ).rename(columns={"parameter": "typical_parameter", "value": "typical_value"})

    merged_data = pd.merge(
        merged_data,
        wet_data[["stationReference", "parameter", "value"]],
        on="stationReference",
        how="left"
    ).rename(columns={"parameter": "wet_parameter", "value": "wet_value"})

    merged_data.drop_duplicates(inplace=True)
    merged_data.dropna(inplace=True)

    return merged_data


def Wet_Rainfall_HeatMap(station_name):
    """
    Create a heatmap with rainfall data in wet day based on the given station.
    """

    merged_data = Rainfall_merged_data(station_name)

    m = folium.Map(location=[merged_data['latitude'][0], merged_data['longitude'][0]], zoom_start=7)

    # Filter data for rainfall
    rainfall_data = [
        [row['latitude'], row['longitude'], row['wet_value']]
        for _, row in merged_data.iterrows() if row['wet_parameter'] == 'rainfall'
    ]

    # Filter data for level
    level_data = [
        [row['latitude'], row['longitude'], row['wet_value']]
        for _, row in merged_data.iterrows() if row['wet_parameter'] == 'level'
    ]

    # Create color maps for the color bars
    merged_vmin = 0.2
    merged_vmax = 0.8
    rainfall_colormap = cm.LinearColormap(['blue', 'yellow', 'red'], vmin=merged_vmin, vmax=merged_vmax, caption='Rainfall (mm)')
    level_colormap = cm.LinearColormap(['green', 'orange', 'purple'], vmin=merged_vmin, vmax=merged_vmax, caption='Water Level (mm)')
    rainfall_colormap.add_to(m)
    level_colormap.add_to(m)
    
    # Add HeatMap for rainfall with a gradient (e.g., blue to red)
    heatmap_rainfall = folium.FeatureGroup(name='Rainfall Heatmap')
    HeatMap(
        rainfall_data,
        radius=15,
        blur=10,
        max_zoom=1,
        gradient={0.2: 'blue', 0.5: 'yellow', 0.8: 'red'}
    ).add_to(heatmap_rainfall)
    heatmap_rainfall.add_to(m)

    # Add HeatMap for level with a different gradient (e.g., green to purple)
    heatmap_level = folium.FeatureGroup(name='Level Heatmap')
    HeatMap(
        level_data,
        radius=15,
        blur=10,
        max_zoom=1,
        gradient={0.2: 'green', 0.5: 'orange', 0.8: 'purple'}
    ).add_to(heatmap_level)
    heatmap_level.add_to(m)

    # Add layer control to toggle between heatmaps
    folium.LayerControl().add_to(m)

    return m


def Typical_Rainfall_HeatMap(station_name):
    """
    Create a heatmap with rainfall data in typical day based on the given station.
    """
    
    merged_data = Rainfall_merged_data(station_name)

    m = folium.Map(location=[merged_data['latitude'][0], merged_data['longitude'][0]], zoom_start=7)

    # Filter data for rainfall
    rainfall_data = [
        [row['latitude'], row['longitude'], row['typical_value']]
        for _, row in merged_data.iterrows() if row['typical_parameter'] == 'rainfall'
    ]

    # Filter data for level
    level_data = [
        [row['latitude'], row['longitude'], row['typical_value']]
        for _, row in merged_data.iterrows() if row['typical_parameter'] == 'level'
    ]

    # Create color maps for the color bars
    merged_vmin = 0.2
    merged_vmax = 0.8
    rainfall_colormap = cm.LinearColormap(['blue', 'yellow', 'red'], vmin=merged_vmin, vmax=merged_vmax, caption='Rainfall (mm)')
    level_colormap = cm.LinearColormap(['green', 'orange', 'purple'], vmin=merged_vmin, vmax=merged_vmax, caption='Water Level (mm)')
    rainfall_colormap.add_to(m)
    level_colormap.add_to(m)
    
    # Add HeatMap for rainfall with a gradient (e.g., blue to red)
    heatmap_rainfall = folium.FeatureGroup(name='Rainfall Heatmap')
    HeatMap(
        rainfall_data,
        radius=15,
        blur=10,
        max_zoom=1,
        gradient={0.2: 'blue', 0.5: 'yellow', 0.8: 'red'}
    ).add_to(heatmap_rainfall)
    heatmap_rainfall.add_to(m)

    # Add HeatMap for level with a different gradient (e.g., green to purple)
    heatmap_level = folium.FeatureGroup(name='Level Heatmap')
    HeatMap(
        level_data,
        radius=15,
        blur=10,
        max_zoom=1,
        gradient={0.2: 'green', 0.5: 'orange', 0.8: 'purple'}
    ).add_to(heatmap_level)
    heatmap_level.add_to(m)

    # Add layer control to toggle between heatmaps
    folium.LayerControl().add_to(m)

    return m