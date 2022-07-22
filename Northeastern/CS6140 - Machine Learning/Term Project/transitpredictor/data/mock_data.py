from random import Random, random, randint
from typing import List, Iterator

from transitpredictor.data import (
    DataDay,
    DataDaySegment,
    StationMaintenance,
    weather_ty,
    TrainDeparture,
)
from transitpredictor.constants import NUM_STATIONS


class MockTrainDeparture(TrainDeparture):
    def __init__(self, ts, train_id, station_id):
        self.ts = ts
        self.train_id = train_id
        self.station_id = station_id
        r = Random(train_id)
        self.vals = r.random()

    def get_train_headway(self) -> float:
        return self.vals

    def get_train_timestamp(self) -> float:
        return self.ts

    def get_train_dwell_time(self) -> float:
        return self.vals

    def get_actual_delays(self) -> List[float]:
        return [self.vals] * (NUM_STATIONS - self.station_id - 1)

    def get_station_id(self) -> int:
        return self.station_id

    def get_trip_id(self) -> int:
        return self.train_id


class MockStationMaintenance(StationMaintenance):
    def get_num_entries(self) -> float:
        return random()

    def get_weather(self) -> weather_ty:
        return randint(0, 2)


class MockDataDaySegment(DataDaySegment):
    def __init__(self, timestamp, trip_id_start):
        self.timestamp = timestamp
        self.trip_id_start = trip_id_start

    def get_day_timestamp(self) -> int:
        return 0

    def get_train_events(self) -> Iterator[List[TrainDeparture]]:
        for i in range(19):
            for t in range(10):
                yield MockTrainDeparture(self.timestamp + (i * 10), t + self.trip_id_start, i)

    def get_station_maintenance(self) -> List[StationMaintenance]:
        return [MockStationMaintenance()] * NUM_STATIONS


class MockDataDay(DataDay):
    def get_data_day_iterator(self) -> Iterator[DataDaySegment]:
        for i in range(30):
            yield MockDataDaySegment(i * 300, i * 10)

    def num_trips(self) -> int:
        return 300
