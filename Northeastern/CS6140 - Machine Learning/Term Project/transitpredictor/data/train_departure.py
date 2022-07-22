import math
from typing import List


class TrainDeparture:
    """A single departure of a train from a station"""

    def __init__(self, trip_id, station_id, train_ts, train_dwell_time, train_headway, actual_delays):
        self.trip_id = trip_id
        self.station_id = station_id
        self.train_ts = train_ts
        self.train_dwell_time = train_dwell_time
        self.train_headway = train_headway
        self.actual_delays = actual_delays
        assert len(actual_delays) + self.station_id == 19
        assert not math.isnan(self.train_ts)
        for delay in self.actual_delays:
            assert not math.isnan(delay)

    def get_trip_id(self) -> int:
        """Refer to DataDay for information about trip IDs"""
        return self.trip_id

    def get_station_id(self) -> int:
        """0 = Forest Hills, 18 = Malden Center. Trains never depart from Oak Grove."""
        return self.station_id

    def get_train_timestamp(self) -> float:
        """
        The number of seconds since the train departed Forest Hills.
        NOTE: This should be difference from mean.
        The mean in this context is the average of train timestamps at _this_ station ID, from the _entire_ dataset
        """
        return self.train_ts

    def get_train_dwell_time(self) -> float:
        """
        Number of seconds between the train arriving and departing from the station, also difference from mean.
        For Forest Hills, since trains never arrive, this should be 0
        (as if all trains had an average dwell time at Forest Hills)
        """
        return 0 if math.isnan(self.train_dwell_time) else self.train_dwell_time

    def get_train_headway(self) -> float:
        """
        Number of seconds since the last train departed, as a difference from mean for this station,
         across the whole dataset.
        The cleaned dataset doesn't include the first train of the day, precisely to avoid needing to deal with headway.
        """
        return 0 if math.isnan(self.train_headway) else self.train_headway

    def get_actual_delays(self) -> List[float]:
        """
        The amount of time between the train's departure from this station, and its _arrival_ to another station.
        For example, if this event is a departure from Assembly, the returned list should be:
        [Seconds until Wellington arrival, Seconds until Malden Center arrival, Seconds until Oak Grove arrival]
        Again, these are all be normalized as the difference from mean.
        The mean is calculated as the average time between each _pair_ of stations, so in the above example,
        the Wellington, Malden Center, and Oak Grove arrival delays would each be subtracted by a different amount.
        """
        return self.actual_delays
