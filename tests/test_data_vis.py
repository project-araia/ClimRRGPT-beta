import pandas as pd

from src.data_vis.climrr_utils import categorize_fwi, fwi_color, subset_by_crossmodel


def test_categorize_fwi():
    assert categorize_fwi(5) == "Low"
    assert categorize_fwi(15) == "Medium"
    assert categorize_fwi(30) == "High"
    assert categorize_fwi(37) == "Very High"
    assert categorize_fwi(45) == "Extreme"
    assert categorize_fwi(60) == "Very Extreme"


def test_fwi_color():
    assert fwi_color(5) == "rgb(255, 255, 0, 0.5)"  # Low
    assert fwi_color(60) == "rgb(255, 0, 0, 0.5)"  # Very Extreme


def test_subset_by_crossmodel():
    df = pd.DataFrame(
        [{"Crossmodel": "ModelA", "Value": 10}, {"Crossmodel": "ModelB", "Value": 20}]
    )
    subset = subset_by_crossmodel(df, "ModelB")
    assert subset["Value"] == 20
