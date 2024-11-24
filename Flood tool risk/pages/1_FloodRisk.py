import streamlit as st
import pandas as pd
from streamlit_folium import folium_static
from flood_tool.common.utils_visualization import (
    format_data,
    dummy_pipeline,
    high_risk_stations,
    format_rainfall,
    Risk_HeatMap,
    plot_flood_risk_map,
    get_heavy_rainfall_data,
)
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

sector_data = pd.read_csv("flood_tool/resources/sector_data.csv")

# typical_weather = pd.read_csv("flood_tool/example_data/typical_day.csv")
# typical_weather = get_heavy_rainfall_data(typical_weather, station)
# wet_weather = pd.read_csv("flood_tool/example_data/wet_day.csv")
# wet_weather = get_heavy_rainfall_data(wet_weather, station)

st.sidebar.title("Flood Risk")

option = st.sidebar.selectbox(
    "Choose an option :", ["CSV File", "Post Code", "Latitude and Longitude"]
)

if option == "CSV File":
    st.sidebar.subheader("Upload your CSV")
    uploaded_file = st.sidebar.file_uploader(
        "Select your CSV file", type="csv"
        )
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            missing_columns = [
                col for col in REQUIRED_COLUMNS if col not in df.columns
                ]
            if missing_columns:
                st.sidebar.error(
                    f"Missing Columns : {', '.join(missing_columns)}"
                )
            else:
                st.sidebar.success("All necessary columns are present.")

                if set(df.columns) != set(REQUIRED_COLUMNS):
                    st.sidebar.warning(
                        "Some columns are not useful and will be removed."
                    )
                    df = df[REQUIRED_COLUMNS]

                st.sidebar.write("Preview of the dataset :")
                st.sidebar.dataframe(df)
                st.write("### Risk map of the area selected")
                data = dummy_pipeline(df)
                merged_data = format_data(data, post)
                total_value = merged_data["house_price"].sum()
                folium_static(Risk_HeatMap(merged_data, sector_data))

                st.markdown(
                    """
                <div style="background-color: white;
                    border: 2px solid grey; padding: 10px; font-size: 14px;">
                    <strong>Legend:</strong><br>
                    <span style="color: black;">Info-sign Marker:</span>
                    Most Dangerous Area (Risk: 7)
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.metric(
                    label="Total value of houses", value=f"{total_value:,} £"
                    )
                # if st.button("Plot plot"):
                #     st.dataframe(merged_data)
                #     st.dataframe(wet_weather)
                #     folium_static(plot_flood_risk_map(merged_data,wet_weather))

                if st.button("Get rainfall data for high risk points"):
                    st.write(
                        "Rainfall data for high risk points (Risk Label = 7)"
                        )
                    hr_station = high_risk_stations(station, merged_data)
                    rainfall_data_station = format_rainfall(hr_station)
                    st.write(rainfall_data_station)

        except Exception as e:
            st.sidebar.error(f"Error will reading the file : {e}")

elif option == "Post Code":
    st.sidebar.subheader("Choose Post Code")
    selected_postcode = st.sidebar.selectbox(
        "Select Post Code:", POSTCODE
        )
    st.write(selected_postcode)
    # Predict postcode and display the postcode on the map

elif option == "Latitude and Longitude":
    st.sidebar.subheader("Type coordinates")
    latitude = st.sidebar.number_input("Latitude", format="%.6f")
    longitude = st.sidebar.number_input("Longitude", format="%.6f")
    if latitude or longitude:
        st.sidebar.write(f"Latitude : {latitude}, Longitude : {longitude}")
        # Predict latitude and longitude
