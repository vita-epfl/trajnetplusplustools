from collections import defaultdict

import torch

from .data import Row


class VanillaLSTM(torch.nn.Module):
    def __init__(self, input_dim=2, embedding_dim=64, hidden_dim=32):
        super(VanillaLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.input_embeddings = torch.nn.Sequential(
            torch.nn.Linear(input_dim, embedding_dim),
            torch.nn.ReLU(),
        )
        self.lstm = torch.nn.LSTMCell(embedding_dim, hidden_dim)
        self.hidden2vel = torch.nn.Linear(hidden_dim, 2)

    def forward(self, observed, n_predict=11):
        """forward

        observed shape is (seq, batch, observables)
        """
        hidden_state = (torch.zeros(1, self.hidden_dim),
                        torch.zeros(1, self.hidden_dim))

        predicted = []
        for obs in observed:
            emb = self.input_embeddings(obs)
            hidden_state = self.lstm(emb, hidden_state)

            new_vel = self.hidden2vel(hidden_state[0])
            predicted.append(new_vel)

        for _ in range(n_predict):
            new_vel = self.hidden2vel(hidden_state[0])
            predicted.append(new_vel)

            # update LSTM
            emb = self.input_embeddings(new_vel)
            hidden_state = self.lstm(emb, hidden_state)

        return torch.stack(predicted, dim=0)


def occupancy(scene):
    """Returns the occupancy grid for every frame of the primary path."""

    other_locations = defaultdict(list)
    for path in scene[1:]:
        for row in path:
            other_locations[row.frame].append((row.x, row.y))

    def is_occupiend(frame, x_min, y_min, x_max, y_max):
        if frame not in other_locations:
            return False

        for x, y in other_locations[frame]:
            if x_min < x < x_max and y_min < y < y_max:
                return True

        return False

    return [[(
        is_occupiend(row.frame, row.x - 1, row.y - 1, row.x, row.y),
        is_occupiend(row.frame, row.x, row.y - 1, row.x + 1, row.y),
        is_occupiend(row.frame, row.x - 1, row.y, row.x, row.y + 1),
        is_occupiend(row.frame, row.x, row.y, row.x + 1, row.y + 1),
    )] for row in scene[0]]


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
            velocity_inputs = observed[1:] - observed[:-1]
            velocity_outputs = self.model(velocity_inputs)[9-1:]

        last_row = observed_path[9-1]
        predicted_path = []
        for (vel,) in velocity_outputs:
            row = Row(0, ped_id, last_row.x + vel[0], last_row.y + vel[1])
            predicted_path.append(row)
            last_row = row

        return predicted_path
