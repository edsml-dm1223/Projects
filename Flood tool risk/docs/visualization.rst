***************************
Visualising Risk & Rainfall
***************************

Flood Risk Visualization
========================

The **Flood Risk Visualization** is designed to help users understand how flood risk is distributed across the UK. 
It also allows users to check the intensity of rainfall in specific areas to assess the immediate risk of flooding.

Introduction
------------

The flood risk visualization is a web-based tool that enables users to explore flood risk in the Jubilee project.  
We designed it to be user-friendly and interactive, so users can easily explore the data and understand the flood risk 
in our project without needing to code or use a terminal.

Features
--------

The visualization web tool is built using **Streamlit** and is composed of three main pages:

1. **Home Page**
   - Provides information about our project and displays a map of the latest rainfall in the UK.  
   - We make API calls to the UK Government's environmental database to retrieve the latest rainfall data.

2. **Flood Risk Page**
   - Allows users to explore flood risk across the UK.  
   - Users can calculate flood risk for a specific location, a postcode, or upload a CSV file with multiple locations.  
   - We use our machine learning model to calculate flood risk and display each point on a map, layered with rainfall data.

3. **Rainfall Page**
   - Lets users explore the latest rainfall data in the UK.  
   - Users can search by postcode, latitude and longitude, or directly select a station.  
   - The tool categorizes rainfall intensity (from "slight" to "violent") and allows users to calculate flood risk for 
     specific locations based on the rainfall data.

Technical Details
-----------------

- Built using **Streamlit** for a seamless and interactive web-based experience.
- Leverages real-time data from the UK Government's environmental database via API calls.
- Incorporates a machine learning model to evaluate flood risk dynamically.
