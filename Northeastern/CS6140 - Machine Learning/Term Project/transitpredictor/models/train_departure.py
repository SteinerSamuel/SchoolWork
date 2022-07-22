import torch
import torch.nn as nn
from transitpredictor.constants import *

TRAIN_DEPART_EVENT_DIMS = 64


class TrainDeparture(nn.Module):
    """
    Inputs: train, station, timestamp, dwell_time, headway:
    - train: Train vector (batch, TRAIN_VECTOR_DIMS)
    - station: Station vector (batch, STATION_VECTOR_DIMS)
    - timestamp: Number of seconds since the train departed Forest Hills (batch)
    - dwell_time: Number of seconds between arrival and departure for this station (batch)
    - headway: Number of seconds since the last train departed form this station (batch)

    Outputs: new_train_vector, new_station_vector
    - new_train_vector: Train vector (batch, TRAIN_VECTOR_DIMS)
    - new_station_vector: Station vector (batch, STATION_VECTOR_DIMS)

    It seems reasonable to model a train's interaction with a station like:
     ┌─────────┐  ┌───────────┐   ┌────────────┐
     │train vec│  │station vec│   │other inputs│
     └──┬─┬────┘  └─┬───┬─────┘   └──────┬─────┘
        │ │         │   │                │
        │ └─────────┼───┴──────┬─────────┘
        │           │          │
        │           │   ┌──────▼─────┐
        │           │   │nn_summarize│
        │           │   └───────────┬┘
        │           │               │
        │           │   ┌───────────▼─────────────┐
        │           │   │vector representing event│
        │           │   └───┬───────┬─────────────┘
        │           │       │       │
     ┌──▼──────┐    │       │       │
     │gru_train◄────┼───────┘       │
     └────┬────┘    │               │
          │         │        ┌──────▼────┐
          │         └────────►gru_station│
          │                  └──────┬────┘
          │                         │
      ┌───▼─────────┐       ┌───────▼───────┐
      │new train vec│       │new station vec│
      └─────────────┘       └───────────────┘

    Since we have a discrete event, we can update the train using a gated unit
    """

    def __init__(self):
        super().__init__()
        # There's nothing sacred about this, it should just be a "boring" neural network
        self.nn_summarize = nn.Sequential(
            nn.Linear(
                TRAIN_VECTOR_DIMS + STATION_VECTOR_DIMS + 1 + 1 + 1,
                TRAIN_DEPART_EVENT_DIMS,
            ),
            nn.LeakyReLU(),
            nn.Linear(TRAIN_DEPART_EVENT_DIMS, TRAIN_DEPART_EVENT_DIMS),
            nn.LeakyReLU(),
            nn.Linear(TRAIN_DEPART_EVENT_DIMS, TRAIN_DEPART_EVENT_DIMS),
            nn.Sigmoid(),
        )

        # The train/station vectors _are_ the hidden state
        self.gru_train = nn.GRUCell(TRAIN_DEPART_EVENT_DIMS, TRAIN_VECTOR_DIMS)
        self.gru_station = nn.GRUCell(TRAIN_DEPART_EVENT_DIMS, STATION_VECTOR_DIMS)

    def forward(self, train, station, timestamp, dwell_time, headway):
        timestamp_with_dim = timestamp.view(-1, 1)
        dwell_time_with_dim = dwell_time.view(-1, 1)
        headway_with_dim = headway.view(-1, 1)

        inp_features = torch.cat(
            (train, station, timestamp_with_dim, dwell_time_with_dim, headway_with_dim),
            dim=1,
        )
        event_summary = self.nn_summarize(inp_features)

        new_train_vec = self.gru_train(event_summary, train)
        new_station_vec = self.gru_station(event_summary, station)

        return new_train_vec, new_station_vec
