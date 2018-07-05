import math
import pytest
import torch
import trajnettools.lstm


def test_simple():
    gaussian_parameters = torch.Tensor([
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
    ])
    coordinates = torch.Tensor([
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    loss = trajnettools.lstm.PredictionLoss()(gaussian_parameters, coordinates).numpy().tolist()
    gauss_denom = 1/math.sqrt(2*math.pi)**2
    assert loss == pytest.approx([-math.log(gauss_denom), -math.log(gauss_denom)], 1e-4)


def test_narrower_progression():
    gaussian_parameters = torch.Tensor([
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.1, 0.1, 0.0],
    ])
    coordinates = torch.Tensor([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    loss = trajnettools.lstm.PredictionLoss()(gaussian_parameters, coordinates).numpy().tolist()
    assert loss[0] > loss[1]
    assert loss[1] > loss[2]
