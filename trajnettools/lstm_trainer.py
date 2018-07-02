import random

import pysparkling
import torch
import torch.nn.functional as F

from . import augmentation
from . import readers
from .lstm import occupancy, OLSTM, VanillaLSTM, VanillaPredictor


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
            target = torch.Tensor([[(r.x, r.y)] for r in path])

            model.zero_grad()
            outputs = model(observed)

            velocity_targets = target[2:] - target[1:-1]
            velocity_outputs = outputs[1:] - outputs[:-1]
            loss = F.mse_loss(velocity_outputs, velocity_targets)

            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()
        print('loss', epoch_loss / len(paths))

    return VanillaPredictor(model)


def train_olstm(paths, lstm_predictor, epochs=90):
    model = OLSTM()
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
