from collections import defaultdict

import torch

from .data import Row


class VanillaLSTM(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=32):
        super(VanillaLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.input_embeddings = torch.nn.Sequential(
            torch.nn.Linear(2, embedding_dim),
            torch.nn.ReLU(),
        )
        self.lstm = torch.nn.LSTMCell(embedding_dim, hidden_dim)
        self.hidden2vel = torch.nn.Linear(hidden_dim, 2)

    def forward(self, observed, n_predict=12):
        """forward

        observed shape is (seq, batch, observables)
        """
        hidden_state = (torch.zeros(1, self.hidden_dim),
                        torch.zeros(1, self.hidden_dim))

        predicted = []
        for obs1, obs2 in zip(observed[:-1], observed[1:]):
            emb = self.input_embeddings(obs2 - obs1)
            hidden_state = self.lstm(emb, hidden_state)

            new_vel = self.hidden2vel(hidden_state[0])
            predicted.append(obs2 + new_vel)

        for _ in range(n_predict):
            emb = self.input_embeddings(new_vel)
            hidden_state = self.lstm(emb, hidden_state)

            new_vel = self.hidden2vel(hidden_state[0])
            predicted.append(predicted[-1] + new_vel)

        return torch.stack(predicted, dim=0)


class OLSTM(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=32):
        super(OLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.input_embeddings = torch.nn.Sequential(
            torch.nn.Linear(2, embedding_dim),
            torch.nn.ReLU(),
        )
        self.lstm = torch.nn.LSTMCell(embedding_dim + 4, hidden_dim)
        self.hidden2vel = torch.nn.Linear(hidden_dim, 2)

    def forward(self, observed, other_paths):
        """forward

        observed shape is (seq, batch, observables)
        """
        hidden_state = (torch.zeros(1, self.hidden_dim),
                        torch.zeros(1, self.hidden_dim))

        predicted = []
        for obs1, obs2, others_obs in zip(observed[:-1], observed[1:], other_paths[1:]):
            emb = torch.cat([
                self.input_embeddings(obs2 - obs1),
                occupancy(obs2, others_obs),
            ], dim=1)
            hidden_state = self.lstm(emb, hidden_state)

            new_vel = self.hidden2vel(hidden_state[0])
            predicted.append(obs2 + new_vel)

        for others_obs in other_paths[len(observed) + 1:]:
            emb = torch.cat([
                self.input_embeddings(new_vel),
                occupancy(predicted[-1], others_obs),
            ], dim=1)
            hidden_state = self.lstm(emb, hidden_state)

            new_vel = self.hidden2vel(hidden_state[0])
            predicted.append(predicted[-1] + new_vel)

        return torch.stack(predicted, dim=0)


def occupancy(xy, other_xy):
    """Returns the occupancy."""

    def is_occupied(x_min, y_min, x_max, y_max):
        for x, y in other_xy:
            if x_min < x < x_max and y_min < y < y_max:
                return True

        return False

    x, y = xy
    return torch.Tensor([[
        is_occupied(x - 1, y - 1, x, y),
        is_occupied(x, y - 1, x + 1, y),
        is_occupied(x - 1, y, x, y + 1),
        is_occupied(x, y, x + 1, y + 1),
    ]])


class VanillaPredictor(object):
    def __init__(self, model):
        self.model = model

    def save(self, filename):
        with open(filename, 'wb') as f:
            torch.save(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return torch.load(f)

    def __call__(self, observed_path):
        ped_id = observed_path[0].pedestrian
        with torch.no_grad():
            observed = torch.Tensor([[(r.x, r.y)] for r in observed_path[:9]])
            outputs = self.model(observed)[9-1:]

        return [Row(0, ped_id, x, y) for ((x, y),) in outputs]
