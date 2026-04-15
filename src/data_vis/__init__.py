from functools import partial
from typing import Any, Dict

from src.data_vis.census import analyze_census_data
from src.data_vis.climrr import (
    ClimRRAnnualProjectionsCDNP,
    ClimRRAnnualProjectionsCoolingDegreeDays,
    ClimRRAnnualProjectionsHeatIndex,
    ClimRRAnnualProjectionsHeatingDegreeDays,
    ClimRRAnnualProjectionsPrecipitation,
    ClimRRAnnualProjectionsTemperature,
    ClimRRAnnualProjectionsWindSpeed,
    ClimRRDailyProjectionsPrecipitation,
    ClimRRSeasonalProjectionsFWI,
    ClimRRSeasonalProjectionsTemperature,
)
from src.data_vis.wildfire_perimeters import analyze_wildfire_perimeters


def dispatch_analyze_fn(keywords) -> Dict[str, Any]:
    dispatch_dict: Dict[str, Any] = {
        "Fire Weather Index (FWI) projections": ClimRRSeasonalProjectionsFWI,
        "Seasonal Temperature Maximum projections": partial(
            ClimRRSeasonalProjectionsTemperature, "Maximum"
        ),
        "Seasonal Temperature Minimum projections": partial(
            ClimRRSeasonalProjectionsTemperature, "Minimum"
        ),
        "Annual Temperature Maximum projections": partial(
            ClimRRAnnualProjectionsTemperature, "Maximum"
        ),
        "Annual Temperature Minimum projections": partial(
            ClimRRAnnualProjectionsTemperature, "Minimum"
        ),
        "Daily Precipitation Max projections": partial(
            ClimRRDailyProjectionsPrecipitation, "Max"
        ),
        "Daily Precipitation Mean projections": partial(
            ClimRRDailyProjectionsPrecipitation, "Mean"
        ),
        "Precipitation projections": ClimRRAnnualProjectionsPrecipitation,
        "Consecutive Dry Days projections": ClimRRAnnualProjectionsCDNP,
        "Wind Speed projections": ClimRRAnnualProjectionsWindSpeed,
        "Cooling Degree Days projections": ClimRRAnnualProjectionsCoolingDegreeDays,
        "Heating Degree Days projections": ClimRRAnnualProjectionsHeatingDegreeDays,
        "Heat Index projections": ClimRRAnnualProjectionsHeatIndex,
        "Census data": analyze_census_data,
        "Recent Fire Perimeters data": analyze_wildfire_perimeters,
    }
    analyze_fn_dict: Dict[str, Any] = {}
    for keyword in keywords:
        factory: Any = dispatch_dict[keyword]
        if "projections" in keyword:
            instance = factory()
            analyze_fn_dict[keyword] = instance.analyze
        else:
            analyze_fn_dict[keyword] = factory
    return analyze_fn_dict
