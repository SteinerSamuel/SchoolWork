import glob
import meteostat
import os
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm.auto import tqdm
from typing import Iterator


"""
Parse stop_ids into station_ids in TrainDeparture
using this dictionary key. 0 = Forest Hills
"""
translate_stop_ids = {
    70001: 0,
    70003: 1,
    70005: 2,
    70007: 3,
    70009: 4,
    70011: 5,
    70013: 6,
    70015: 7,
    70017: 8,
    70019: 9,
    70021: 10,
    70023: 11,
    70025: 12,
    70027: 13,
    70029: 14,
    70031: 15,
    70279: 16,
    70033: 17,
    70035: 18,
    70036: 19
}

translate_station_names = {
    "Forest Hills": 0,
    "Green Street": 1,
    "Stony Brook": 2,
    "Jackson Square": 3,
    "Roxbury Crossing": 4,
    "Ruggles": 5,
    "Massachusetts Avenue": 6,
    "Back Bay": 7,
    "Tufts Medical Center": 8,
    "Chinatown": 9,
    "Downtown Crossing": 10,
    "State Street": 11,
    "Haymarket": 12,
    "North Station": 13,
    "Community College": 14,
    "Sullivan Square": 15,
    "Assembly": 16,
    "Wellington": 17,
    "Malden Center": 18,
    "Oak Grove": 19
}

translate_event_type = {
    "ARR": "ARR",
    "DEP": "DEP",
    "PRA": "ARR"
}


"""
Map to encode weather via DataFrame.replace().
Translate from float meteostat codes to our 0-6 int encoding.
For details on weather codes from meteostat, please see
https://dev.meteostat.net/formats.html#weather-condition-codes
"""
translate_weather = {
    0: [1.0, 2.0, 3.0, 4.0, 23.0], # everything else (clear, fair, cloudy, overcast)
    1: [5.0, 6.0], # fog
    2: [7.0, 8.0, 9.0, 17.0, 18.0], # all forms of rain
    3: [10.0, 11.0], # freezing rain
    4: [12.0, 13.0, 19.0, 20.0], # sleet
    5: [14.0, 15.0, 16.0, 21.0, 22.0], # all forms of snow
    6: [24.0, 25.0, 26.0, 27.0], # storm+hail
}


def get_averages(full_run_data: pd.DataFrame, train_data: pd.DataFrame, gated_entries: pd.DataFrame) -> Iterator:
    
    station_pairs = train_data.groupby(["stop_id_dep","stop_id_arr"]).mean()[["total_travel_time"]]
    # key is a tuple of (starting station, target station)
    avg_time_between_stations = station_pairs.T.to_dict("list")

    station_rows = full_run_data.groupby(["stop_id"]).mean()[["train_timestamp"]]
    avg_station_timestamp = station_rows.T.to_dict("list")
    # print(avg_station_timestamp)

    station_rows = full_run_data.groupby(["stop_id"]).mean()[["dwell_time"]]
    avg_dwell_time = station_rows.T.to_dict("list")
    # print(avg_dwell_time)

    station_rows = full_run_data.groupby(["stop_id"]).mean()[["headway"]]
    avg_headway = station_rows.T.to_dict("list")
    # print(avg_headway)

    num_entries = gated_entries.groupby(["station_name"]).mean()[["gated_entries"]]

    return {
        "timestamp": avg_station_timestamp,
        "dwell_time": avg_dwell_time,
        "headway": avg_headway,
        "time_between_stations": avg_time_between_stations,
        "num_entries": num_entries
    }

class DataProcessor:

    def __init__(self, path: str = "transitpredictor/data"):
        self.path = path
        
    """
    Import gated entry data. Filter by 2019 Orange Line data.
    """
    def parse_gate_data(self) -> pd.DataFrame:
        csv_filepath = os.path.join(self.path, "MBTA_Gated_Station_Entries.csv")
        df = pd.read_csv(csv_filepath)

        df = df[(df.service_date > '2019/01/01') & (df.service_date < '2020/01/01')]
        df = df[df.route_or_line == "Orange Line"]
        df = df.drop(columns=["ObjectId", "stop_id"])
        df = df.sort_values(["service_date"])
        df["station_name"] = df["station_name"].replace(translate_station_names)
        service_dates = df["service_date"].apply(lambda x: datetime.strptime(x.split(" ")[0], "%Y/%m/%d"))
        service_times = df["time_period"].apply(lambda x: datetime.strptime(x, "(%H:%M:%S)"))
        service_timedeltas = service_times.apply(lambda x: timedelta(hours=x.hour, minutes=x.minute))
        service_timedeltas = service_timedeltas.apply(
            lambda x: x + timedelta(days=1) if x < timedelta(hours=3) else x)
        df["actual_times"] = service_dates + service_timedeltas
        df["actual_times"] = df["actual_times"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        df = df.set_index("actual_times")
        df.to_csv(os.path.join(self.path, "processed_entry_data.csv"), index=True)

        return df

    """
    Filter out rows of Orange line North to Oak Grove and remove excess columns.
    Sort by trip_id -> event_time. This way ARR/DEP rows are properly aligned.
    Translate stop_id to station_id 0-19.
    """
    def parse_orange_line_data(self) -> pd.DataFrame:
        files = glob.iglob(os.path.join(self.path, "Events_2019", "*.csv"))

        df = pd.concat((pd.read_csv(f) for f in files))
        df = df[(df.route_id == "Orange") & (df.direction_id == 1)]
        df = df.drop(columns=["direction_id", "stop_sequence", "vehicle_id", "vehicle_label"])
        df = df.sort_values(["service_date","trip_id", "event_time"])
        # df = df[~df["trip_id"].str.contains("ADDED")]
        df["stop_id"] = df["stop_id"].replace(translate_stop_ids)
        df["event_type"] = df["event_type"].replace(translate_event_type)
        df["event_time"] = df["event_time"].apply(datetime.fromtimestamp)

        run_results = {}
        train_results = {}

        dfs_per_day = list(df.groupby("service_date"))

        for date, d in tqdm(dfs_per_day):
            events_by_train = list(t_events[(t_events["event_type"] == "ARR") | (t_events["event_type"] == "DEP")].drop_duplicates(subset=["service_date", "trip_id", "stop_id", "event_type"]) for _, t_events in d.groupby("trip_id"))

            full_runs = list(t_events for t_events in events_by_train if len(t_events) == 39)

            if len(full_runs) == 0:
                print("Skipping " + date)
                continue
            full_runs = pd.concat(full_runs)
            tids, _ = pd.factorize(full_runs["trip_id"])
            full_runs["train_id"] = tids

            runs_minim = full_runs[["train_id", "stop_id", "event_time", "event_time_sec", "event_type"]].groupby("train_id")
            e = []
            for tid, r in runs_minim:
                arrs = r[r["event_type"] == "ARR"]
                deps = r[r["event_type"] == "DEP"]

                event_pairs = arrs.join(deps, how="cross", lsuffix="_arr", rsuffix="_dep")
                event_pairs = event_pairs[event_pairs["stop_id_arr"] > event_pairs["stop_id_dep"]] # Sequential order
                event_pairs["total_travel_time"] = event_pairs["event_time_sec_arr"] - event_pairs["event_time_sec_dep"]
                event_pairs = event_pairs[["train_id_arr", "stop_id_dep", "stop_id_arr", "total_travel_time", "event_time_arr", "event_time_sec_arr"]]
                event_pairs.rename(columns={"train_id_arr":"train_id"}, inplace=True)
                event_pairs = event_pairs.sort_values(["train_id", "stop_id_dep", "stop_id_arr"])
                e.append(event_pairs)
            train_results[datetime.strptime(date, "%Y-%m-%d")] = pd.concat(e)
        
            runs = full_runs

            # Calculate dwell time
            runs[["prev_event_type", "prev_stop_id", "prev_trip_id", "prev_event_time_sec"]] = runs[["event_type", "stop_id", "trip_id", "event_time_sec"]].shift(1)
            runs["dwell_time"] = np.where((runs["prev_event_type"] == "ARR") & (runs["event_type"] == "DEP") & (runs["prev_stop_id"] == runs["stop_id"]) & (runs["prev_trip_id"] == runs["trip_id"]), runs["event_time_sec"] - runs["prev_event_time_sec"], math.nan)
            runs = runs.drop(columns=["prev_event_type", "prev_stop_id", "prev_trip_id", "prev_event_time_sec"])

            # We no longer care about arrivals at all
            runs = runs[runs["event_type"] == "DEP"]
            runs = runs.drop(columns=["event_type"])

            # Pair up each train for the day with the next one that departs from the same station
            runs = runs.sort_values(["stop_id", "event_time_sec"])
            runs[["prev_stop_id", "prev_event_time_sec"]] = runs[["stop_id", "event_time_sec"]].shift(1)
            runs["headway"] = np.where(runs["prev_stop_id"] == runs["stop_id"], runs["event_time_sec"] - runs["prev_event_time_sec"], math.nan)
            runs = runs.drop(columns=["prev_stop_id", "prev_event_time_sec"])

            forest_hills_deps = runs[runs["stop_id"] == 0].drop(columns=["stop_id", "route_id", "dwell_time", "event_time", "service_date", "trip_id", "headway"]).reindex(columns=["train_id", "event_time_sec"])
            runs = runs.merge(forest_hills_deps, left_on="train_id", right_on="train_id", how="inner", suffixes=("", "_fh_dep"))
            runs["train_timestamp"] = runs["event_time_sec"] - runs["event_time_sec_fh_dep"]
            runs = runs.drop(columns="event_time_sec_fh_dep")

            runs = runs.sort_values(["event_time_sec"])

            train_results[datetime.strptime(date, "%Y-%m-%d")] = pd.concat(e)
            run_results[datetime.strptime(date, "%Y-%m-%d")] = runs

        # make dict a Dataframe to save to csv
        run_results = pd.concat(run_results)
        run_results.index.names = ["index_to_cut1", "index_to_cut2"]
        run_results = run_results.reset_index(level=["index_to_cut1", "index_to_cut2"]).drop(columns=["index_to_cut1", "index_to_cut2"])
        run_results.to_csv(os.path.join(self.path, "processed_full_run_data.csv"), index=False)

        train_results = pd.concat(train_results)
        train_results.index.names = ["service_date", "index_to_cut"]
        train_results = train_results.reset_index(level="index_to_cut").drop(columns=["index_to_cut"])
        train_results.to_csv(os.path.join(self.path, "processed_train_data.csv"))

        return train_results


    """
    Pull 2019 hourly weather data from Boston Logan weather station.
    """
    def parse_weather_data(self, station_id: str = "72509") -> pd.DataFrame:
        # Default Boston Logan weather station id = 72509
        start_date = datetime(2019, 1, 1)
        end_date = datetime(2019, 12, 31)
        hourly_weather = meteostat.Hourly(station_id, start_date, end_date)
        weather_df = hourly_weather.fetch()
        for key in translate_weather:
            weather_df["coco"] = weather_df["coco"].replace(dict.fromkeys(translate_weather[key], key))

        weather_df = weather_df.rename(columns={"coco": "weather_code"})
        weather_df[["weather_code"]].to_csv(os.path.join(self.path, "processed_weather_data.csv"))
        return weather_df.weather_code


class ProcessedData:
    def __init__(self, path: str = "transitpredictor/data"):
        weather_filepath = os.path.join(path,"processed_weather_data.csv")
        self.weather_data =  pd.read_csv(weather_filepath, index_col="time")

        entry_filepath = os.path.join(path,"processed_entry_data.csv")
        self.entry_data =  pd.read_csv(entry_filepath, index_col="actual_times").sort_index()

        train_filepath = os.path.join(path,"processed_train_data.csv")
        self.train_data = pd.read_csv(train_filepath, index_col="service_date")

        full_runs_filepath = os.path.join(path,"processed_full_run_data.csv")
        self.full_run_data = pd.read_csv(full_runs_filepath, index_col="service_date")

        self.averages = get_averages(train_data=self.train_data, full_run_data=self.full_run_data, gated_entries=self.entry_data)


    def get_train_data(self, date: datetime) -> pd.DataFrame:
        return self.train_data.loc[date.strftime("%Y-%m-%d")]

    def get_weather_data(self) -> pd.DataFrame:
        return self.weather_data

    def get_entry_data(self) -> pd.DataFrame:
        return self.entry_data
    
    def get_full_run_data(self, date: datetime) -> pd.DataFrame:
        return self.full_run_data.loc[date.strftime("%Y-%m-%d")]
    
    def get_averages(self) -> pd.DataFrame:
        return self.averages


def process_data_for_baseline(p : ProcessedData, path="transitpredictor/data"):
    departures_days = list(p.full_run_data.groupby(level=0))

    average_dwell_times = pd.DataFrame(p.averages["dwell_time"]).T.rename(columns={0: "avg_dwell_time"})
    average_timestamps = pd.DataFrame(p.averages["timestamp"]).T.rename(columns={0: "avg_timestamp"})
    average_headways = pd.DataFrame(p.averages["headway"]).T.rename(columns={0: "avg_headway"})

    average_station_entries = p.averages["num_entries"].rename(columns={"gated_entries": "avg_gated_entries"})

    average_times_between_stations = pd.DataFrame(p.averages["time_between_stations"]).T.rename(columns={0: "avg_travel_time"})

    train_days = []
    valid_days = []
    test_days = []

    for i, (day, departures_today) in enumerate(tqdm(departures_days)):
        event_pairs_today = p.train_data.loc[day].set_index(["train_id", "stop_id_dep"])
        departures_with_corresponding_arrivals = departures_today.join(event_pairs_today, on=["train_id", "stop_id"])
        departures_with_corresponding_arrivals = departures_with_corresponding_arrivals.drop(columns=["event_time_arr", "event_time_sec_arr"])

        with_averages = departures_with_corresponding_arrivals \
            .join(average_dwell_times, on=["stop_id"]) \
            .join(average_timestamps, on=["stop_id"]) \
            .join(average_headways, on=["stop_id"]) \
            .join(average_times_between_stations, on=["stop_id", "stop_id_arr"])

        with_averages["dwell_time"] -= with_averages["avg_dwell_time"]
        with_averages["train_timestamp"] -= with_averages["avg_timestamp"]
        with_averages["headway"] -= with_averages["avg_headway"]
        with_averages["delay"] = with_averages["total_travel_time"] - with_averages["avg_travel_time"]

        with_averages = with_averages.drop(columns=["avg_dwell_time", "avg_timestamp", "avg_headway", "total_travel_time", "avg_travel_time", "route_id", "trip_id"])

        with_averages["dwell_time"] = np.where(np.isnan(with_averages["dwell_time"]), 0, with_averages["dwell_time"])
        with_averages["headway"] = np.where(np.isnan(with_averages["headway"]), 0, with_averages["headway"])

        p.entry_data.index = pd.to_datetime(p.entry_data.index, utc=True)
        p.weather_data.index = pd.to_datetime(p.weather_data.index, utc=True)
        with_averages["event_time"] = pd.to_datetime(with_averages["event_time"], utc=True)

        with_averages = pd.merge_asof(with_averages, p.weather_data, left_on="event_time", right_index=True)
        with_averages = pd.merge_asof(with_averages, p.entry_data, left_on="event_time", right_index=True, left_by="stop_id", right_by="station_name")

        with_averages = with_averages.join(average_station_entries, on="stop_id")
        with_averages["gated_entries"] -= with_averages["avg_gated_entries"]
        with_averages = with_averages.drop(columns=["avg_gated_entries", "route_or_line", "station_name", "time_period", "service_date"])

        assert not with_averages.isnull().any().any()  # No NaNs ever!!!

        if i % 10 < 7:
            train_days.append(with_averages)
        elif i % 10 == 7:
            valid_days.append(with_averages)
        else:
            test_days.append(with_averages)

    pd.concat(train_days).to_csv(os.path.join(path, "processed_for_baseline_train_data.csv"))
    pd.concat(valid_days).to_csv(os.path.join(path, "processed_for_baseline_valid_data.csv"))
    pd.concat(test_days).to_csv(os.path.join(path, "processed_for_baseline_test_data.csv"))



