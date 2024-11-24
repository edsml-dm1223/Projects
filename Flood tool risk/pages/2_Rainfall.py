import streamlit as st
import pandas as pd
from flood_tool.visualization.live import get_live_station_data
from flood_tool.common.utils_visualization import get_nearest_station, categorise_rainfall
import plotly.express as px
import time

REQUIRED_COLUMNS = [
    "postcode",
    "easting",
    "northing",
    "soilType",
    "elevation",
    "nearestWatercourse",
    "distanceToWatercourse",
    "localAuthority",
]

post = pd.read_csv(
    "flood_tool/visualization/unique_postcode_mapping.csv"
)
POSTCODE = post.postcode.tolist()

station = pd.read_csv("flood_tool/resources/stations.csv")
STATION_ID = station.stationReference.tolist()

st.sidebar.title("Rainfall Data")
option = st.sidebar.selectbox(
    "Choose an option :", ["Station ID", "Latitude and Longitude","Post Code"]
)
rainfall_data = None 
if option == "Station ID":
    st.sidebar.subheader("Choose a Station ID")
    selected_station = st.sidebar.selectbox("Select Station ID", STATION_ID)
    rainfall_data = get_live_station_data(selected_station)


elif option == "Latitude and Longitude":
    st.sidebar.subheader("Type coordinates")
    latitude = st.sidebar.number_input("Latitude", format="%.6f")
    longitude = st.sidebar.number_input("Longitude", format="%.6f")
    if latitude or longitude:
        st.sidebar.write(f"Latitude : {latitude}, Longitude : {longitude}")
    selected_station = get_nearest_station(latitude, longitude, station)
    st.sidebar.write(f"Nearest station : {selected_station}")
    rainfall_data = get_live_station_data(selected_station)

elif option == "Post Code":
    st.sidebar.subheader("Choose Post Code")
    selected_postcode = st.sidebar.selectbox(
        "Select Post Code:", POSTCODE
        )
    loc = post[post["postcode"]==selected_postcode]
    if not loc.empty:
        latitude = loc["latitude"].iloc[0]
        longitude = loc["longitude"].iloc[0]
        selected_station = get_nearest_station(latitude, longitude, station)
        st.sidebar.write(f"Nearest station : {selected_station}")
        rainfall_data = get_live_station_data(selected_station)
        if rainfall_data.empty:
            st.sidebar.error("No data available for the selected postcode.")
            rainfall_data = None

    else:
        st.sidebar.error("No data available for the selected postcode.")
if rainfall_data is not None:
    st.title("Latest rainfall 🌧️")
    fig = px.line(rainfall_data, x='dateTime', y='value', title='Rainfall on the last days', labels={'value':'Rainfall (mm)', 'dateTime':'Date'})
    st.plotly_chart(fig)

rainfall_category = categorise_rainfall(rainfall_data)
if rainfall_category:
    st.write(f"The Rainfall is considered {rainfall_category}")

if st.button("Predict the risk of flood in this area"):
    with st.spinner(
        "Predicting flood risk, it can take up to 1 minute..."
    ):
        time.sleep(5)
    st.write("Precition not available yet.")
