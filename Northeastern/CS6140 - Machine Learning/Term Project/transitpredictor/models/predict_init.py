import torch
import torch.nn as nn
from transitpredictor.constants import *


class PredictInit(nn.Module):
    """
    Inputs: train, station
    - train: Train vector (batch, TRAIN_VECTOR_DIMS)
    - station: Station vector representing the station that the train just departed (batch, STATION_VECTOR_DIMS)

    Output: predict_vector (batch, PREDICTION_VECTOR_DIMS)
     ┌─────────┐       ┌───────────┐
     │train vec│       │station vec│
     └─────────┴───┬───┴───────────┘
                   │
                   │
                   │
             ┌─────▼─────┐
             │nn_predinit│
             └─────┬─────┘
                   │
                   │
                   │
             ┌─────▼─────┐
             │predict_vec│
             └───────────┘
    """

    def __init__(self):
        super().__init__()

        self.nn_predinit = nn.Sequential(
            nn.Linear(TRAIN_VECTOR_DIMS + STATION_VECTOR_DIMS, PREDICTION_VECTOR_DIMS),
            nn.LeakyReLU(),
            nn.Linear(PREDICTION_VECTOR_DIMS, PREDICTION_VECTOR_DIMS),
            nn.LeakyReLU(),
            nn.Linear(PREDICTION_VECTOR_DIMS, PREDICTION_VECTOR_DIMS),
            nn.Sigmoid(),
        )

    def forward(self, train, station):
        inp_features = torch.cat((train, station), dim=1)
        return self.nn_predinit(inp_features)
