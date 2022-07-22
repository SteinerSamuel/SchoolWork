import torch
import torch.nn as nn
from transitpredictor.constants import *


class PredictOutput(nn.Module):
    """
    Input: prediction vector (batch, PREDICTION_VECTOR_DIMS)

    Output: delay times (batch, 1)
     ┌────────┐
     │pred_vec│
     └────┬───┘
          │
     ┌────▼─────┐
     │nn_predict│
     └────┬─────┘
          │
          │
          │
      ┌───▼──┐
      │output│
      └──────┘
    """

    def __init__(self):
        super().__init__()

        self.nn_predict = nn.Sequential(
            nn.Linear(PREDICTION_VECTOR_DIMS, PREDICTION_VECTOR_DIMS),
            nn.LeakyReLU(),
            nn.Linear(PREDICTION_VECTOR_DIMS, PREDICTION_VECTOR_DIMS),
            nn.LeakyReLU(),
            nn.Linear(PREDICTION_VECTOR_DIMS, 1),
        )

    def forward(self, pred_vec):
        return self.nn_predict(pred_vec)
