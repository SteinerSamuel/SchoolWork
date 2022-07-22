import torch

from transitpredictor.constants import NUM_STATIONS
from transitpredictor.models import TransitPredictorGroup


def incr_train_locs_one_hot(train_locs_one_hot: torch.Tensor):
    zeros_on_left = torch.zeros_like(train_locs_one_hot[:, 0]).view((-1, 1))
    return torch.cat((zeros_on_left, train_locs_one_hot[:, :-1]), dim=1)


def incr_train_locs(train_locs: torch.Tensor):
    return torch.where(
        train_locs >= NUM_STATIONS, NUM_STATIONS, train_locs + 1
    )  # When loc = NUM_STATIONS, it means the train is "done"


def get_predicted_delays(
    predictor: TransitPredictorGroup,
    train_vecs: torch.Tensor,
    initial_train_locs: torch.Tensor,
    stations_now: torch.Tensor,
    device: torch.device,
):
    """
    train_vecs: List of train vectors (batch, TRAIN_VECTOR_DIMS)
    initial_train_locs: List of train departure locations (batch)
    stations_now: Station vectors for each train (batch, NUM_STATIONS-1, STATION_VECTOR_DIMS)
    delay_mask:  1 where actual_delays is meaningful (batch, NUM_STATIONS-1)
    """
    batch_size = initial_train_locs.size(0)
    train_locs = initial_train_locs
    increasing_indices = torch.arange(
        0, batch_size, step=1, dtype=torch.long, device=device
    )
    pred_vectors = predictor.pred_init(
        train_vecs, stations_now[increasing_indices, train_locs]
    )
    pred_delays_with_padding = torch.zeros((batch_size, NUM_STATIONS)).to(
        device
    )  # Last row is junk

    for i in range(NUM_STATIONS - 1):
        train_locs = incr_train_locs(
            train_locs
        )  # A train that started at station x will now be at station (x + i + 1)
        these_predictions = predictor.pred_output(pred_vectors)
        pred_delays_with_padding[:, train_locs - 1] = these_predictions
        train_locs_capped = torch.where(
            train_locs >= NUM_STATIONS - 2, NUM_STATIONS - 2, train_locs
        )
        pred_vectors = predictor.pred_add_station(
            pred_vectors, stations_now[increasing_indices, train_locs_capped]
        )

    # This actually gets the results
    predicted_delays = pred_delays_with_padding[:, :-1]
    return predicted_delays
