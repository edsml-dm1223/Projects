import pandas as pd
from flood_tool.common.utils_visualization import *

def test_format_data():
    data = pd.DataFrame({
        "postcode": ["A1 1AA", "B2 2BB"],
        "riskLabel": [1, 2],
        "house_price": [250000, 300000],
    })
    mapping = pd.DataFrame({
        "postcode": ["A1 1AA", "B2 2BB"],
        "latitude": [51.5074, 52.2053],
        "longitude": [-0.1278, 0.1218],
    })

    formatted_data = format_data(data, mapping)
    assert list(formatted_data.columns) == ["postcode", "latitude", "longitude", "riskLabel", "house_price"]
    assert len(formatted_data) == 2


def test_dummy_pipeline():
    data = pd.DataFrame({"postcode": ["A1 1AA", "B2 2BB"]})
    result = dummy_pipeline(data)
    assert "riskLabel" in result.columns
    assert "house_price" in result.columns
    assert len(result) == 2


def test_categorise_rainfall():
    dataframe = pd.DataFrame({
        "dateTime": ["2023-11-21 10:00:00", "2023-11-21 11:00:00"],
        "value": [1.5, 4.0],
    })
    category = categorise_rainfall(dataframe)
    assert category == "moderate"

    empty_df = pd.DataFrame(columns=["dateTime", "value"])
    assert pd.isna(categorise_rainfall(empty_df))


def test_categorise_rainfall_light():
    dataframe = pd.DataFrame({
        "dateTime": ["2023-11-21 10:00:00", "2023-11-21 11:00:00"],
        "value": [1.5, 4.0],
    })
    max_rainfall = categorise_rainfall_light(dataframe)
    assert max_rainfall == 4.0




def test_high_risk_stations():
    stations = pd.DataFrame({
        "latitude": [51.5074, 52.2053],
        "longitude": [-0.1278, 0.1218],
    })
    data = pd.DataFrame({
        "latitude": [51.5094],
        "longitude": [-0.1258],
        "riskLabel": [7],
    })

    result = high_risk_stations(stations, data)
    assert len(result) == 1




def test_get_heavy_rainfall_data():
    rainfall = pd.DataFrame({
        "stationReference": ["ST001", "ST001", "ST002"],
        "value": [1.5, 2.5, 4.0],
    })
    stations = pd.DataFrame({
        "stationReference": ["ST001", "ST002"],
        "latitude": [51.5074, 52.2053],
        "longitude": [-0.1278, 0.1218],
    })

    result = get_heavy_rainfall_data(rainfall, stations)
    assert "rainfall_category" in result.columns
    assert len(result) == 2


def test_get_house_head():
    sector_data = pd.DataFrame({
        "postcodeSector": ["A1 1", "B2 2"],
        "households": [100, 200],
        "headcount": [300, 600],
    })
    households, headcount = get_house_head("A1 1AA", sector_data)
    assert households.values[0] == 100
    assert headcount.values[0] == 300
