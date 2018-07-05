from collections import defaultdict
import random

import pysparkling
import torch
import torch.nn.functional as F

from .. import augmentation
from .. import readers
from .loss import PredictionLoss
from .occupancy import OLSTM, OLSTMPredictor
from .vanilla import VanillaLSTM, VanillaPredictor


def train_vanilla(scenes, epochs=90):
    model = VanillaLSTM()
    criterion = PredictionLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-3,
                                momentum=0.9,
                                weight_decay=1e-4)

    model.train()
    for epoch in range(1, epochs + 1):
        print('epoch', epoch)
        if epoch == 30:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
        if epoch == 60:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
        random.shuffle(scenes)
        epoch_loss = 0.0
        for paths in scenes:
            path = paths[0]
            path = augmentation.random_rotation([path])[0]

            observed = torch.Tensor([[(r.x, r.y)] for r in path[:9]])
            target = torch.Tensor([[(r.x, r.y)] for r in path])

            model.zero_grad()
            outputs = model(observed)

            velocity_targets = target[2:] - target[1:-1]
            loss = criterion(outputs, velocity_targets)

            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()
        print('loss', epoch_loss / len(scenes))

    return VanillaPredictor(model)


def others_xy_truth(scene):
    others_xy = defaultdict(dict)
    frames = [r.frame for r in scene[0]]
    pedestrians = set()
    for path in scene[1:]:
        for row in path:
            others_xy[row.pedestrian][row.frame] = (row.x, row.y)
            pedestrians.add(row.pedestrian)

    # preserve order
    pedestrians = list(pedestrians)

    return [[others_xy.get(ped_id, {}).get(frame, (None, None))
             for ped_id in pedestrians]
            for frame in frames]


def train_olstm(scenes, vanilla_model, epochs=35, directional=False):
    model = OLSTM(directional=directional)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.01,
                                momentum=0.9,
                                weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        print('epoch', epoch)
        if epoch == 15:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
        if epoch == 25:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        random.shuffle(scenes)
        epoch_loss = 0.0
        for scene in scenes:
            scene = augmentation.random_rotation(scene)
            path = scene[0]

            observed = torch.Tensor([[(r.x, r.y)] for r in path[:9]])
            target = torch.Tensor([[(r.x, r.y)] for r in path])

            model.zero_grad()
            others_xy = others_xy_truth(scene)
            outputs = model(observed, others_xy)

            velocity_targets = target[2:] - target[1:-1]
            velocity_outputs = outputs[1:] - outputs[:-1]
            loss = F.mse_loss(velocity_outputs, velocity_targets)

            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()
        print('loss', epoch_loss / len(scenes))

    return OLSTMPredictor(model, vanilla_model)


def main(input_files):
    sc = pysparkling.Context()
    scenes = (sc
              .wholeTextFiles(input_files)
              .values()
              .map(readers.trajnet)
              .collect())

    # Vanilla LSTM training
    lstm_predictor = train_vanilla(scenes)
    lstm_predictor.save('output/vanilla_lstm.pkl')
    # lstm_predictor = VanillaPredictor.load('output/vanilla_lstm.pkl')

    # O-LSTM training
    # olstm_predictor = train_olstm(scenes, lstm_predictor.model)
    # olstm_predictor.save('output/olstm.pkl')

    # DO-LSTM training
    # dolstm_predictor = train_olstm(scenes, lstm_predictor.model, directional=True)
    # dolstm_predictor.save('output/dolstm.pkl')


if __name__ == '__main__':
    main('output/test/biwi_eth/*.txt')
    # main('output/train/**/*.txt')
