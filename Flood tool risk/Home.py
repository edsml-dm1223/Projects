import streamlit as st
import pandas as pd
import folium

from folium.plugins import HeatMap
from streamlit_folium import folium_static
from flood_tool.common.utils_visualization import get_heavy_rainfall_data
from flood_tool.visualization.live import get_latest_rainfall_readings
from streamlit_extras.let_it_rain import rain


stations = pd.read_csv("flood_tool/resources/stations.csv")  # change path
STATION_ID = stations.stationReference.tolist()
# Set up the Streamlit app

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
rain(
    emoji="💧",
    font_size=54,
    falling_speed=2,
    animation_length=1,
)
st.title("Flood Prediction 🌊")

st.sidebar.success("Select a tool between FloodRisk and Rainfall data.")

st.markdown(
    """
    Welcome to the Flood Prediction app! 🌊
    We developed this app to help you predict flood risk and
    analyze rainfall data in the UK. We use state-of-the-art
    machine learning models to provide you with accurate predictions.

    **👈 Select a tool from the sidebar** to see some examples
    of what JubileeTech can do!

    ### Want to learn more?
    - Check out [UK GOV database](https://check-for-flooding.service.gov.uk)
    ### See more complex tools
    - [A simple inertial formulation for flood inundation modelling](https://www.sciencedirect.com/science/article/pii/S0022169410001538?via%3Dihub)
"""
)
if st.button("Get latest rainfall"):
    with st.spinner(
        "Getting latest rainfall data, it can take up to 1 minute..."
    ):
        latest_rainfall_reading = get_latest_rainfall_readings()
    df = get_heavy_rainfall_data(
        latest_rainfall_reading, stations
    )
    st.dataframe(df)
    df = df.dropna(subset=["latitude", "longitude"])

    if not {"latitude", "longitude", "rainfall_category"}.issubset(df.columns):
        st.error(
            "The DataFrame does not have the required columns:",
            + "'latitude', 'longitude', 'rainfall'."
        )
    else:
        map_center = [df["latitude"].mean(), df["longitude"].mean()]
        folium_map = folium.Map(location=map_center, zoom_start=7)

        heat_data = [
            [row["latitude"], row["longitude"], row["rainfall_category"]]
            for _, row in df.iterrows()
        ]
        HeatMap(heat_data).add_to(folium_map)

        # Afficher la carte
        st.markdown("### Rainfall Heatmap")
        folium_static(folium_map)
