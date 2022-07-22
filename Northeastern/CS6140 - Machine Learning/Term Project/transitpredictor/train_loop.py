import os
from typing import List

import torch
import torch.optim as optim
from torch.types import Device
from torch.utils.tensorboard import SummaryWriter

from transitpredictor.data import DataDay
from transitpredictor.models import TransitPredictorGroup
from transitpredictor.prediction_vector_system import get_predicted_delays
from transitpredictor.station_vector_system import (
    iter_events_for_day,
    TrainDepartureBatch,
)


def get_sse_and_num_samples(
    predictor: TransitPredictorGroup, data: List[DataDay], device: Device
):
    sse = torch.zeros(1, device=device)
    total_samples = torch.zeros(1, device=device)
    sum_abs_error = 0.0

    for day in data:
        day_batches = list(iter_events_for_day(predictor, day, device))
        day_superbatch = TrainDepartureBatch.super_batch(day_batches)

        predicted_delay = get_predicted_delays(
            predictor,
            day_superbatch.train_vecs,
            day_superbatch.train_locs,
            day_superbatch.stations_now,
            device,
        )
        squared_errors = (predicted_delay - day_superbatch.actual_delays) ** 2
        sum_abs_error += float((torch.abs(predicted_delay - day_superbatch.actual_delays) * day_superbatch.delay_mask).sum())
        sse += (squared_errors * day_superbatch.delay_mask).sum()
        total_samples += day_superbatch.delay_mask.sum()

    return sse, total_samples, sum_abs_error


def training_loop(
    predictor: TransitPredictorGroup,
    train_data: List[DataDay],
    valid_data: List[DataDay],
    device: Device,
    run_name: str,
):
    os.mkdir(f"./checkpoints/{run_name}")
    writer = SummaryWriter("./runs/" + run_name)
    optimizer = optim.Adam(predictor.parameters())

    for epoch in range(10000):
        print(f"Epoch {epoch}")

        total_train_error = 0
        num_train_samples = 0

        for data in train_data:
            optimizer.zero_grad()
            day_train_error, day_train_samples, _ = get_sse_and_num_samples(
                predictor, [data], device
            )

            day_train_error.backward()
            optimizer.step()

            total_train_error += float(day_train_error)
            num_train_samples += float(day_train_samples)

        print(float(total_train_error))
        writer.add_scalar("SSE/train", float(total_train_error), epoch)
        writer.add_scalar(
            "MSE/train", float(total_train_error / num_train_samples), epoch
        )

        total_valid_error, num_valid_samples, _ = get_sse_and_num_samples(
            predictor, valid_data, device
        )
        writer.add_scalar("SSE/valid", float(total_valid_error), epoch)
        writer.add_scalar(
            "MSE/valid", float(total_valid_error / num_valid_samples), epoch
        )

        torch.save(predictor.state_dict(), f"./checkpoints/{run_name}/e{epoch}.pt")
