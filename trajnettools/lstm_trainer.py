import random

import pysparkling
import torch
import torch.nn.functional as F

from . import augmentation
from . import readers
from .lstm import occupancy, VanillaLSTM, VanillaPredictor


def train_vanilla(paths, epochs=90):
    model = VanillaLSTM()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.01,
                                momentum=0.9,
                                weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        print('epoch', epoch)
        if epoch == 30:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
        if epoch == 60:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        random.shuffle(paths)
        epoch_loss = 0.0
        for path in paths:
            path = augmentation.random_rotation([path])[0]

            observed = torch.Tensor([[(r.x, r.y)] for r in path[:9]])
            target = torch.Tensor([[(r.x, r.y)] for r in path[1:]])
            velocity_inputs = observed[1:] - observed[:-1]
            velocity_targets = target[1:] - target[:-1]

            model.zero_grad()
            output = model(velocity_inputs)

            loss = F.mse_loss(output, velocity_targets)
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()
        print('loss', epoch_loss / len(paths))

    return VanillaPredictor(model)


def train_olstm(scenes, lstm_predictor, epochs=100):
    model = SocialLSTM(input_dim=6)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(1, epochs + 1):
        print('epoch', epoch)
        if epoch == 30:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.01
        if epoch == 70:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
        random.shuffle(scenes)
        epoch_loss = 0.0
        for paths in scenes:
            paths = augmentation.random_rotation(paths)
            path = paths[0]

            observed = torch.Tensor([[(r.x, r.y)] for r in path[:9]])
            target = torch.Tensor([[(r.x, r.y)] for r in path[1:]])
            velocity_inputs = observed[1:] - observed[:-1]
            velocity_targets = target[1:] - target[:-1]

            occupancy_inputs = torch.Tensor(occupancy(paths)[1:9])
            all_inputs = torch.cat((velocity_inputs, occupancy_inputs), dim=2)

            model.zero_grad()
            output = model(all_inputs)

            loss = F.mse_loss(output, velocity_targets)
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()
        print('loss', epoch_loss / len(scenes))

    return VanillaPredictor(model)


def main(input_files):
    sc = pysparkling.Context()
    paths = (sc
             .wholeTextFiles(input_files)
             .mapValues(readers.trajnet)
             .cache())

    # Vanilla LSTM training
    training_paths = paths.values().map(lambda paths: paths[0]).collect()
    lstm_predictor = train_vanilla(training_paths)
    lstm_predictor.save('output/vanilla_lstm.pkl')

    # O-LSTM training
    # training_paths = paths.values().collect()
    # olstm_predictor = train_olstm(training_paths, lstm_predictor)
    # olstm_predictor.save('output/olstm.pkl')


if __name__ == '__main__':
    main('output/test/biwi_eth/*.txt')
