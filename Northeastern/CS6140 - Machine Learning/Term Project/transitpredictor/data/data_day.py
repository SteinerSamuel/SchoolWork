from datetime import datetime, timedelta
from typing import Iterator, List, Set

import pandas as pd
import numpy as np

from transitpredictor.data import TrainDeparture
from transitpredictor.data.offline_processing import ProcessedData
from transitpredictor.data.station_maintenance import StationMaintenance


def batch_train_events(events: Iterator[TrainDeparture]) -> Iterator[List[TrainDeparture]]:
    seen_this_run: Set[int] = set()
    this_run: List[TrainDeparture] = []
    for event in events:
        if event.get_trip_id() in seen_this_run:
            yield this_run
            this_run = []
            seen_this_run = set()

        this_run.append(event)
        seen_this_run.add(event.get_trip_id())

    if len(this_run) > 0:
        yield this_run


class DataDaySegment:
    """
    Represents 30 minutes of continuous data. start_time is in seconds past midnight.
    """

    def __init__(self, train_departure_events_segment: pd.DataFrame, train_real_delays_day: pd.DataFrame,
                 weather_segment: int, num_entries_segment: pd.DataFrame, averages):
        # TODO: get necessary data from ProcessedData
        self.train_departures_in_segment = train_departure_events_segment
        self.train_delays_all_day = train_real_delays_day
        self.weather = weather_segment
        self.num_entries = num_entries_segment
        self.averages = averages

    def get_station_maintenance(self) -> List[StationMaintenance]:
        """
        The number of fare gate entries between (day_timestamp - 30 minutes) and day_timestamp,
        as well as the weather for a given time.
        If this is the first segment for a day, there were zero entries
        """
        num_entry_list = self.num_entries.set_index("station_name").sort_index()["gated_entries"] - self.averages["num_entries"]["gated_entries"]
        num_entry_list = np.where(np.isnan(num_entry_list), -self.averages["num_entries"]["gated_entries"], num_entry_list)
        return list(StationMaintenance(self.weather, ne) for ne in num_entry_list)

    def get_train_events(self) -> Iterator[TrainDeparture]:
        """
        Get all train departure events that occurred within this 30-minute time interval, in chronological order
        """
        deps = []
        for train_dep in self.train_departures_in_segment.itertuples():
            event_pairs = self.train_delays_all_day[(self.train_delays_all_day["train_id"] == train_dep.train_id) & (
                    self.train_delays_all_day["stop_id_dep"] == train_dep.stop_id)]
            actual_delays = list(event_pair.total_travel_time - self.averages["time_between_stations"][
                (event_pair.stop_id_dep, event_pair.stop_id_arr)][0] for event_pair in event_pairs.itertuples())

            deps.append(TrainDeparture(
                trip_id=train_dep.train_id,
                station_id=train_dep.stop_id,
                train_ts=train_dep.event_time_sec - self.averages["timestamp"][train_dep.stop_id][0],
                train_dwell_time=train_dep.dwell_time - self.averages["dwell_time"][train_dep.stop_id][0],
                train_headway=train_dep.headway - self.averages["headway"][train_dep.stop_id][0],
                actual_delays=actual_delays
            ))

        return deps


class DataDay:
    """
        Create internal list of DataDaySegment so it's not recreated 
        for every get_data_day_iterator call. Input datetime must be 
        a new day starting at 12:00am.
    """

    def __init__(self, date: datetime, source: ProcessedData):
        self.train_day_data_event_pairs = source.get_train_data(date)
        self.train_day_data_departure_events = source.get_full_run_data(date)
        self.station_entries = source.get_entry_data()
        self.weather_data = source.get_weather_data()
        self.averages = source.get_averages()

        # set start_time to the nearest 30 min rounded down
        start_time = int(self.train_day_data_departure_events.iloc[0]["event_time_sec"])
        start_time = start_time - start_time % (30 * 60)
        end_time = int(self.train_day_data_departure_events.iloc[-1]["event_time_sec"])

        self.day_segments = []

        # to avoid off by one error, add 1 extra 30 min chunk so no data is lost.
        half_hours = (end_time - start_time) // (30 * 60) + 1
        for n in range(half_hours):
            curr_interval = start_time + (30 * 60) * n
            train_time_chunk = self.train_day_data_departure_events[
                (self.train_day_data_departure_events["event_time_sec"] >= curr_interval) & (
                        self.train_day_data_departure_events["event_time_sec"] < curr_interval + 30 * 60)]

            this_datetime =(date + timedelta(seconds=curr_interval))
            num_entries_seg = self.station_entries[self.station_entries.index == this_datetime.strftime("%Y-%m-%d %H:%M:%S")]

            this_datetime_hour_round = datetime(this_datetime.year, this_datetime.month, this_datetime.day, this_datetime.hour)
            weather_code = int(self.weather_data[self.weather_data.index > this_datetime_hour_round.strftime("%Y-%m-%d %H:%M:%S")].iloc[0]["weather_code"])

            self.day_segments.append(DataDaySegment(train_time_chunk, self.train_day_data_event_pairs,
                                                    weather_code, num_entries_seg, self.averages))
                        
    def get_data_day_iterator(self) -> Iterator[DataDaySegment]:
        return self.day_segments

    def num_trips(self) -> int:
        """
        NOTE: The dataset should only contain trains which begin at Forest Hills and make the full journey to Oak Grove
        Also, the dataset doesn't include the first trip of the day, so that we don't need to deal with missing headways
        A single trip between these two stations gets its own trip id
        :return: The number of trips for a given day. Trip ids are between 0 and num_trips exclusive.
        """
        return self.train_day_data_event_pairs["train_id"].max() + 1
