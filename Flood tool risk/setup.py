#!/usr/bin/env python

from setuptools import setup

# Modify, update, and improve this file as necessary.

setup(
    name="Flood Tool",
    version="0.5",
    description="Flood Risk Analysis Tool",
    author="ACDS project Team Jubilee",  # update this
    packages=["flood_tool"],  # DON'T install the scoring package
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "folium",
        "scikit-learn",
        "xgboost",
        "streamlit_extras",
        "streamlit_folium",
        "streamlit",
        "plotly.express"
    ],
)
