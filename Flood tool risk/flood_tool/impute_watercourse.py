import pandas as pd
import numpy as np

def lat_lon_to_cartesian(lat, lon, elevation, earth_radius=6371):
    """
    Converts latitude, longitude, and elevation to 3D Cartesian coordinates.
    
    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        elevation (float): Elevation in meters.
        earth_radius (float): Radius of the Earth in kilometers (default: 6371).
    
    Returns:
        tuple: Cartesian coordinates (x, y, z).
    """
    # Convert latitude and longitude to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Earth's radius + elevation (convert elevation from meters to kilometers)
    r = earth_radius + (elevation / 1000.0)
    
    # Cartesian coordinates
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    return x, y, z

def impute_watercourse_3d(df, lat_col="latitude", lon_col="longitude", elev_col="elevation", watercourse_col="nearestWatercourse"):
    """
    Impute missing watercourse names using 3D Cartesian distance.
    
    Args:
        df (pd.DataFrame): The dataset with latitude, longitude, elevation, and watercourse names.
        lat_col (str): Column name for latitude.
        lon_col (str): Column name for longitude.
        elev_col (str): Column name for elevation.
        watercourse_col (str): Column name for watercourse names.
    
    Returns:
        pd.DataFrame: Dataset with missing watercourse names imputed.
    """
    # Convert known and unknown locations to Cartesian coordinates
    df["cartesian"] = df.apply(lambda row: lat_lon_to_cartesian(row[lat_col], row[lon_col], row[elev_col]), axis=1)
    
    # Split data into known and unknown watercourses
    known = df[df[watercourse_col].notna()]
    unknown = df[df[watercourse_col].isna()]
    
    # Build KDTree for known Cartesian coordinates
    known_coords = np.array(known["cartesian"].tolist())
    tree = KDTree(known_coords)
    
    # Query nearest neighbors for unknown Cartesian coordinates
    unknown_coords = np.array(unknown["cartesian"].tolist())
    distances, indices = tree.query(unknown_coords)
    
    # Impute missing values with the nearest neighbor's watercourse name
    unknown[watercourse_col] = known.iloc[indices][watercourse_col].values
    
    # Combine known and updated unknown datasets
    return pd.concat([known, unknown]).drop(columns=["cartesian"])
