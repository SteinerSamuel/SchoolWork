import argparse
import torch
import os
from datetime import datetime, timedelta
from transitpredictor.data.mock_data import MockDataDay
from transitpredictor.models import TransitPredictorGroup
from transitpredictor.train_loop import training_loop, get_sse_and_num_samples
from transitpredictor.data.offline_processing import DataProcessor, ProcessedData, process_data_for_baseline
from transitpredictor.data.data_day import DataDay 

DEVICE = "cuda:0"


def run_mock(run_name: str):
    mock_day = MockDataDay()
    predictor = TransitPredictorGroup().to(DEVICE)

    training_loop(predictor, [mock_day] * 3, [mock_day] * 2, DEVICE, run_name)


def run(run_name: str, eval_from_checkpoint=None):
    predictor = TransitPredictorGroup().to(DEVICE)

    if eval_from_checkpoint is not None:
        check_name = os.path.join("checkpoints", str(run_name), f"e{eval_from_checkpoint}.pt")
        predictor.load_state_dict(torch.load(check_name, map_location=DEVICE))

    pd = ProcessedData()
    jan_1 = datetime(2019, 1, 1)

    train_set = []
    valid_set = []
    test_set = []

    # data_structure to skip days with no train data for now
    no_data_days = (
        datetime(2019,3,27),
        datetime(2019,5,9),
        datetime(2019,8,10),
        datetime(2019,8,11),
        datetime(2019,8,17),
        datetime(2019,8,18),
        datetime(2019,8,24),
        datetime(2019,8,25),
        datetime(2019,9,7),
        datetime(2019,9,8),
        datetime(2019,9,14),
        datetime(2019,9,15),
        datetime(2019,9,21),
        datetime(2019,9,22),
        datetime(2019,9,28),
        datetime(2019,9,29),
        datetime(2019,10,5),
        datetime(2019,10,6),
        datetime(2019,10,12),
        datetime(2019,10,13),
        datetime(2019,10,19),
        datetime(2019,10,20),
        datetime(2019,10,26),
        datetime(2019,10,27),
        datetime(2019,11,2),
        datetime(2019,11,3),
        datetime(2019,11,9),
        datetime(2019,11,10),
        datetime(2019,12,30),
        datetime(2019,12,31)
    )
    for i in range(363):
        date = jan_1 + timedelta(days=i)
        if date in no_data_days: # no train data for this day
            continue
        if i % 10 < 7:
            if eval_from_checkpoint is None:
                day = DataDay(date, pd)
                train_set.append(day)
        elif i % 10 == 7:
            if eval_from_checkpoint is None:
                day = DataDay(date, pd)
                valid_set.append(day)
        else:
            if eval_from_checkpoint is not None:
                day = DataDay(date, pd)
                test_set.append(day)

    if eval_from_checkpoint is None:
        training_loop(predictor, train_set, valid_set, DEVICE, run_name)

    total_test_error, num_test_samples, abs_error = get_sse_and_num_samples(
        predictor, test_set, DEVICE
    )

    print(float(total_test_error / num_test_samples))
    print(abs_error/float(num_test_samples))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mock", action="store_true", help="Run predictor with mock data.")
    parser.add_argument("-f","--offline", action="store_true", help="Process data into batch-friendly CSV for predictor.")
    parser.add_argument("-n", "--run-name", type=str, help="A name for the run and checkpoint")
    parser.add_argument("-b", "--baseline", action="store_true", help="Process the data for the baseline models")
    parser.add_argument("-e", "--eval", action="store_true", help="Get the MSE on the test set for a given checkpoint")
    args = parser.parse_args()
    if args.mock:
        if args.run_name:
            run_mock(args.run_name)
        else:
            print("Runs must have a name")
    elif args.eval:
        run("test2", 400)
    elif args.offline:
        dp = DataProcessor("transitpredictor/data")
        dp.parse_orange_line_data()
        dp.parse_gate_data()
        dp.parse_weather_data()

        # --Testing purposes only past this line--
        # pd = ProcessedData()
        # d = DataDay(datetime(2019,1,1), pd)
    elif args.baseline:
        pd = ProcessedData()
        process_data_for_baseline(pd)
    else:
        if args.run_name:
            run(args.run_name)
        else:
            print("Runs must have a name")


