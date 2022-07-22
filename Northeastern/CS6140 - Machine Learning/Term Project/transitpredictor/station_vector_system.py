from typing import Iterator, List, NamedTuple

import torch
import torch.nn.functional as F
from torch.types import Device

import transitpredictor.data as data
from transitpredictor.constants import NUM_STATIONS, STATION_VECTOR_DIMS
from transitpredictor.data import DataDay, batch_train_events
from transitpredictor.models import TransitPredictorGroup, StationMaintenance


def station_maint_to_tensor(
    maintenance_segment: List[StationMaintenance], device: Device
):
    sm_entries = torch.Tensor(
        [sm.get_num_entries() for sm in maintenance_segment]
    ).view([-1, 1])
    sm_weather = F.one_hot(
        torch.LongTensor([sm.get_weather() for sm in maintenance_segment]),
        num_classes=7,
    )
    return sm_entries.to(device), sm_weather.to(device)


def train_departures_to_tensor(
    train_departures: List[data.TrainDeparture],
    device: Device,
):
    td_trip_ids = torch.LongTensor([td.get_trip_id() for td in train_departures]).to(
        device
    )
    td_station_ids = torch.LongTensor(
        [td.get_station_id() for td in train_departures]
    ).to(device)
    td_timestamps = torch.tensor(
        [td.get_train_timestamp() for td in train_departures]
    ).to(device)
    td_dwell_times = torch.tensor(
        [td.get_train_dwell_time() for td in train_departures]
    ).to(device)
    td_headways = torch.tensor([td.get_train_headway() for td in train_departures]).to(
        device
    )
    td_actual_delays = []
    td_delay_mask = []

    for td in train_departures:
        zeroes_before_current_station = torch.zeros((td.get_station_id(),))
        delays_after_station = torch.Tensor(td.get_actual_delays())
        ones_after_station = torch.ones_like(delays_after_station)
        td_actual_delays.append(
            torch.cat((zeroes_before_current_station, delays_after_station)).view(
                (1, -1)
            )
        )
        td_delay_mask.append(
            torch.cat((zeroes_before_current_station, ones_after_station)).view((1, -1))
        )

    return (
        td_trip_ids,
        td_station_ids,
        td_timestamps,
        td_dwell_times,
        td_headways,
        torch.cat(td_actual_delays).to(device),
        torch.cat(td_delay_mask).to(device),
    )


class TrainDepartureBatch(NamedTuple):
    # List of train vectors (batch, TRAIN_VECTOR_DIMS)
    train_vecs: torch.Tensor

    # Locations of each train (batch)
    train_locs: torch.Tensor

    # Station vectors right now (batch, NUM_STATIONS, STATION_VECTOR_DIMS)
    stations_now: torch.Tensor

    # Actual delays from current departure to dest stations (batch, NUM_STATIONS-1)
    actual_delays: torch.Tensor

    # 1 where actual_delays is meaningful (batch, NUM_STATIONS-1)
    # This will be 1 where train_locs_one_hot is 1, as well as all stations after
    delay_mask: torch.Tensor

    @staticmethod
    def super_batch(batches: List["TrainDepartureBatch"]):
        return TrainDepartureBatch(
            torch.cat([batch.train_vecs for batch in batches], dim=0),
            torch.cat([batch.train_locs for batch in batches], dim=0),
            torch.cat([batch.stations_now for batch in batches], dim=0),
            torch.cat([batch.actual_delays for batch in batches], dim=0),
            torch.cat([batch.delay_mask for batch in batches], dim=0),
        )


def iter_events_for_day(
    predictor: TransitPredictorGroup,
    data_day: DataDay,
    device: Device,
) -> Iterator[TrainDepartureBatch]:
    trains = torch.tile(
        predictor.initial_train_vector.view((1, -1)), (data_day.num_trips(), 1)
    )
    stations = predictor.initial_station_vectors
    for data_segment in data_day.get_data_day_iterator():
        sm_entries, sm_weather = station_maint_to_tensor(
            data_segment.get_station_maintenance(), device
        )
        stations = predictor.station_maintenance(stations, sm_entries, sm_weather)

        for train_departures in batch_train_events(data_segment.get_train_events()):
            (
                batch_trip_ids,
                batch_station_ids,
                batch_timestamps,
                batch_dwell_times,
                batch_headways,
                actual_delays,
                delay_mask,
            ) = train_departures_to_tensor(train_departures, device)

            selected_trains = torch.index_select(trains, 0, batch_trip_ids)
            selected_stations = torch.index_select(stations, 0, batch_station_ids)
            new_trains, new_stations = predictor.train_departure(
                selected_trains,
                selected_stations,
                batch_timestamps,
                batch_dwell_times,
                batch_headways,
            )

            trains = torch.index_put(
                trains,
                (torch.LongTensor([t for t in batch_trip_ids]).to(device),),
                new_trains,
            )
            stations = torch.index_put(
                stations,
                (torch.LongTensor([s for s in batch_station_ids]).to(device),),
                new_stations,
            )
            stations_grid = stations.view((1, NUM_STATIONS, STATION_VECTOR_DIMS)).tile(
                (len(batch_trip_ids), 1, 1)
            )
            yield TrainDepartureBatch(
                new_trains, batch_station_ids, stations_grid, actual_delays, delay_mask
            )
