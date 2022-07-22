import torch
import torch.nn as nn

from transitpredictor.constants import *
from transitpredictor.models.station_maintenance import StationMaintenance
from transitpredictor.models.train_departure import TrainDeparture
from transitpredictor.models.predict_output import PredictOutput
from transitpredictor.models.predict_add_station import PredictAddStation
from transitpredictor.models.predict_init import PredictInit


class TransitPredictorGroup(nn.Module):
    """
    Just a utility class to hold all the sub-models, so that they can be treated nicely as a group
    """

    def __init__(self):
        super(TransitPredictorGroup, self).__init__()
        self.initial_train_vector = nn.Parameter(torch.zeros(TRAIN_VECTOR_DIMS))
        self.initial_station_vectors = nn.Parameter(
            torch.zeros(NUM_STATIONS, STATION_VECTOR_DIMS)
        )
        self.train_departure = TrainDeparture()
        self.station_maintenance = StationMaintenance()
        self.pred_init = PredictInit()
        self.pred_add_station = PredictAddStation()
        self.pred_output = PredictOutput()

    def forward(self):
        raise Exception(
            "TransitPredictor is not a single model; the class is just used to hold sub-models"
        )
