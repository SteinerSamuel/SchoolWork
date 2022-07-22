import torch
import torch.nn as nn
from transitpredictor.constants import *

STATION_MAINTENANCE_EVENT_DIMS = 64


class StationMaintenance(nn.Module):
    """
    Inputs: station, entries, weather:
    - station: Station vector (batch, STATION_VECTOR_DIMS)
    - entries: Number of people who entered the station in the last 30 minutes (batch)
    - weather: One-hot encoding of current weather (nice, rain, snow) (batch, 3)

    Output: new_station_vector
    - new_station_vector: Station vector (batch, STATION_VECTOR_DIMS)

    This has a similar architecture to TrainDeparture, but a bit simpler:
    ┌───────────┐   ┌────────────┐
    │station vec│   │other inputs│
    └─┬───┬─────┘   └──────┬─────┘
      │   │                │
      │   └──────┬─────────┘
      │          │
      │   ┌──────▼─────┐
      │   │nn_summarize│
      │   └───────────┬┘
      │               │
      │   ┌───────────▼─────────────┐
      │   │vector representing event│
      │   └───────────┬─────────────┘
      │               │
      │        ┌──────▼────┐
      └────────►gru_station│
               └──────┬────┘
                      │
              ┌───────▼───────┐
              │new station vec│
              └───────────────┘

    This should run every 30 simulator-minutes
    """

    def __init__(self):
        super().__init__()
        # There's nothing sacred about this, it should just be a "boring" neural network
        self.nn_summarize = nn.Sequential(
            nn.Linear(
                STATION_VECTOR_DIMS + 1 + 7,
                STATION_MAINTENANCE_EVENT_DIMS,
            ),
            nn.LeakyReLU(),
            nn.Linear(STATION_MAINTENANCE_EVENT_DIMS, STATION_MAINTENANCE_EVENT_DIMS),
            nn.LeakyReLU(),
            nn.Linear(STATION_MAINTENANCE_EVENT_DIMS, STATION_MAINTENANCE_EVENT_DIMS),
            nn.Sigmoid(),
        )

        # The station vector is the hidden state
        self.gru_station = nn.GRUCell(
            STATION_MAINTENANCE_EVENT_DIMS, STATION_VECTOR_DIMS
        )

    def forward(self, station, entries, weather_ohe):
        entries_with_dim = entries.view(-1, 1)

        inp_features = torch.cat(
            (station, entries_with_dim, weather_ohe),
            dim=1,
        )
        event_summary = self.nn_summarize(inp_features)

        new_station_vec = self.gru_station(event_summary, station)

        return new_station_vec
