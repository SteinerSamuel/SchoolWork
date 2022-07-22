import math
from datetime import datetime
from transitpredictor.data.offline_processing import ProcessedData

"""
See class DataProcessor for weather translation details.
"""
weather_ty = int


class StationMaintenance:

    def __init__(self, weather: weather_ty, num_entries: float):
        self.weather = weather
        self.entry_data = num_entries
        assert not math.isnan(self.entry_data)

    def get_num_entries(self) -> float:
        """How many fare gate entries (DFM)"""
        return self.entry_data


    def get_weather(self) -> weather_ty:
        """What is it like, broadly speaking"""
        return self.weather
