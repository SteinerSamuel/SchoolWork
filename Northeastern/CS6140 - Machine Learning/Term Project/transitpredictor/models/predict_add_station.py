import torch
import torch.nn as nn
from transitpredictor.constants import *


class PredictAddStation(nn.Module):
    """
    Inputs: pred_vec, station
    - pred_vec: Prediction vector (batch, PREDICTION_VECTOR)
    - station: Station vector representing the station that the train is "passing" (batch, STATION_VECTOR_DIMS)

    Output: predict_vector (batch, PREDICTION_VECTOR_DIMS)
     ┌────────┐        ┌───────────┐
     │pred vec│        │station vec│
     └────────┴────┬───┴───────────┘
                   │
                   │
                   │
            ┌──────▼───────┐
            │gru_addstation│
            └──────┬───────┘
                   │
                   │
                   │
             ┌─────▼─────┐
             │predict_vec│
             └───────────┘
    """

    def __init__(self):
        super().__init__()

        self.gru_add_station = nn.GRUCell(STATION_VECTOR_DIMS, PREDICTION_VECTOR_DIMS)

    def forward(self, pred, station):
        return self.gru_add_station(station, pred)
