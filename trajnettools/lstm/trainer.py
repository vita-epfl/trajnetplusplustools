import time
import random

import pysparkling
import torch

from .. import augmentation
from .. import readers
from .loss import PredictionLoss
from .lstm import LSTM, LSTMPredictor, scene_to_xy


class Trainer(object):
    def __init__(self, model=None, criterion=None, optimizer=None, lr_scheduler=None):
        self.model = model if model is not None else LSTM()
        self.criterion = criterion if criterion is not None else PredictionLoss()
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(
            self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        self.lr_scheduler = (lr_scheduler
                             if lr_scheduler is not None
                             else torch.optim.lr_scheduler.StepLR(self.optimizer, 30))

    def train(self, scenes, eval_scenes, epochs=100):
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            preprocess_time = 0.0
            optim_time = 0.0

            print('epoch', epoch)
            self.lr_scheduler.step()

            random.shuffle(scenes)
            epoch_loss = 0.0
            self.model.train()
            for scene_i, scene in enumerate(scenes):
                scene_start = time.time()
                scene = augmentation.random_rotation(scene)
                xy = scene_to_xy(scene)
                preprocess_time += time.time() - scene_start

                optim_start = time.time()
                loss = self.train_batch(xy)
                optim_time += time.time() - optim_start

                epoch_loss += loss

                if scene_i % 100 == 0:
                    print({
                        'type': 'train',
                        'epoch': epoch,
                        'batch': scene_i,
                        'n_batch': len(scenes),
                        'loss': loss,
                    })

            eval_loss = 0.0
            eval_start = time.time()
            self.model.train()  # so that it does not return positions but still normals
            for scene in eval_scenes:
                xy = scene_to_xy(scene)
                eval_loss += self.eval_batch(xy)
            eval_time = time.time() - eval_start

            print({
                'train_loss': epoch_loss / len(scenes),
                'eval_loss': eval_loss / len(eval_scenes),
                'duration': time.time() - start_time,
                'preprocess_time': preprocess_time,
                'optim_time': optim_time,
                'eval_time': eval_time,
            })

    def train_batch(self, xy):
        observed = xy[:9]
        prediction_truth = xy[9:-1].clone()  ## CLONE

        self.optimizer.zero_grad()
        outputs = self.model(observed, prediction_truth)

        targets = xy[2:, 0] - xy[1:-1, 0]
        loss = self.criterion(outputs, targets)
        loss.backward()

        self.optimizer.step()
        return loss.item()

    def eval_batch(self, xy):
        observed = xy[:9]
        prediction_truth = xy[9:-1].clone()  ## CLONE

        with torch.no_grad():
            outputs = self.model(observed, prediction_truth)

            targets = xy[2:, 0] - xy[1:-1, 0]
            loss = self.criterion(outputs, targets)

        return loss.item()


def train_vanilla(scenes, eval_scenes):
    model = LSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    trainer = Trainer(model, optimizer=optimizer)
    trainer.train(scenes, eval_scenes)
    return LSTMPredictor(trainer.model)


def train_olstm(scenes, vanilla_model, directional=False):
    model = OLSTM(directional=directional)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    trainer = Trainer(model, optimizer=optimizer)
    trainer.train(scenes)
    return OLSTMPredictor(trainer.model, vanilla_model)


def main(train_input_files, eval_input_files):
    sc = pysparkling.Context()
    scenes = (sc
              .wholeTextFiles(train_input_files)
              .values()
              .map(readers.trajnet)
              .collect())

    eval_scenes = (sc
                   .wholeTextFiles(eval_input_files)
                   .values()
                   .map(readers.trajnet)
                   .collect())

    # Vanilla LSTM training
    lstm_predictor = train_vanilla(scenes, eval_scenes)
    lstm_predictor.save('output/vanilla_lstm.pkl')
    # lstm_predictor = VanillaPredictor.load('output/vanilla_lstm.pkl')

    # O-LSTM training
    # olstm_predictor = train_olstm(scenes, lstm_predictor.model)
    # olstm_predictor.save('output/olstm.pkl')

    # DO-LSTM training
    # dolstm_predictor = train_olstm(scenes, lstm_predictor.model, directional=True)
    # dolstm_predictor.save('output/dolstm.pkl')


if __name__ == '__main__':
    # main('output/train/biwi_hotel/*.txt', 'output/test/biwi_eth/*.txt')
    main('output/train/**/*.txt', 'output/test/**/*.txt')
