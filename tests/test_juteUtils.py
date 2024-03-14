from ps_funks import juteUtils as jut
import geopandas as gpd
import pandas as pd
import numpy as np


def test_gpd_get_world():
    world = jut.gpd_get_world()
    assert type(world) is gpd.geodataframe.GeoDataFrame


def test_gpd_get_world_regions():
    world_r = jut.gpd_get_world_regions()
    assert type(world_r) is gpd.geodataframe.GeoDataFrame


def test_text_addLineBreaks():
    test_string = "This is a test string This is a test string This is a test string"
    expected_string = "This is a test string\n This is a test string\n This is a test string"
    out = jut.text_addLineBreaks(test_string, 20)
    print("expected_string:")
    print(expected_string)
    print("out:")
    print(out)
    print("end")
    assert expected_string == out


def test_df_among_duplicate_replace_with_max():
    expected_values = np.array(
        [["Smith", "Sam"], ["Smith", "Sam"], ["Bells", "John"], ["Bells", "John"]], dtype=object
    )
    # create a dataframe
    df = pd.DataFrame(
        {
            "last_name": ["Smith", "Smith", "Bells", "Bells"],
            "first_name": ["Sam", "S", "John", "J."],
        }
    )
    jut.df_among_duplicate_replace_with_max(df, "last_name", "first_name")
    assert np.all(df.values == expected_values)