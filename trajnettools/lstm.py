from collections import defaultdict

import torch

from .data import Row


class VanillaLSTM(torch.nn.Module):
    def __init__(self, embedding_dim=16, hidden_dim=128):
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
    def __init__(self, embedding_dim=16, hidden_dim=128, directional=False):
        super(OLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.grid_fn = occupancy if not directional else directional_occupancy

        self.input_embeddings = torch.nn.Sequential(
            torch.nn.Linear(2, embedding_dim),
            torch.nn.ReLU(),
        )
        self.lstm = torch.nn.LSTMCell(embedding_dim + 36, hidden_dim)
        self.hidden2vel = torch.nn.Linear(hidden_dim, 2)

    def forward(self, observed, other_paths):
        """forward

        observed shape is (seq, batch, observables)
        """
        hidden_state = (torch.zeros(1, self.hidden_dim),
                        torch.zeros(1, self.hidden_dim))

        predicted = []
        for obs1, obs2, others_obs1, others_obs2 in zip(
                observed[:-1], observed[1:],
                other_paths[:-1], other_paths[1:]):
            emb = torch.cat([
                self.input_embeddings(obs2 - obs1),
                self.grid_fn(obs1, obs2, others_obs1, others_obs2),
            ], dim=1)
            hidden_state = self.lstm(emb, hidden_state)

            new_vel = self.hidden2vel(hidden_state[0])
            predicted.append(obs2 + new_vel)

        for others_obs1, others_obs2 in zip(other_paths[len(observed) - 1:-1],
                                            other_paths[len(observed):]):
            emb = torch.cat([
                self.input_embeddings(new_vel.detach()),  # DETACH!!!!!
                self.grid_fn(predicted[-2], predicted[-1], others_obs1, others_obs2),
            ], dim=1)
            hidden_state = self.lstm(emb, hidden_state)

            new_vel = self.hidden2vel(hidden_state[0])
            predicted.append(predicted[-1] + new_vel)

        return torch.stack(predicted, dim=0)


def occupancy(_, xy2, __, other_xy2, cell_side=1.0, nx=6, ny=6):
    """Returns the occupancy."""

    def is_occupied(x_min, y_min, x_max, y_max):
        for x, y in other_xy2:
            if x is None or y is None:
                continue
            if x_min < x and x < x_max and y_min < y and y < y_max:
                return True

        return False

    x, y = xy2[0]
    grid_x = torch.linspace(-nx/2 * cell_side, nx/2 * cell_side, nx + 1) + x
    grid_y = torch.linspace(-ny/2 * cell_side, ny/2 * cell_side, ny + 1) + y
    return torch.Tensor([[
        is_occupied(xx1, yy1, xx2, yy2)
        for xx1, xx2 in zip(grid_x[:-1], grid_x[1:])
        for yy1, yy2 in zip(grid_y[:-1], grid_y[1:])
    ]])


def directional_occupancy(xy1, xy2, other_xy1, other_xy2, cell_side=1.0, nx=4, ny=4):
    """Returns the occupancy."""
    xy1 = xy1[0]
    xy2 = xy2[0]
    ref_direction = torch.atan2(xy2[1] - xy1[1], xy2[0] - xy1[0])

    def is_occupied_with_direction(x_min, y_min, x_max, y_max):
        for (x1, y1), (x2, y2) in zip(other_xy1, other_xy2):
            if x1 is None or y1 is None or x2 is None or y2 is None:
                continue
            if x_min < x2 and x2 < x_max and y_min < y2 and y2 < y_max:
                direction = ref_direction - torch.atan2(y2 - y1, x2 - x1)
                forward = -torch.pi/2 < direction < torch.pi < 2
                if forward:
                    return (1, 0)
                return (1, 1)

        return (0, 0)

    x, y = xy2
    grid_x = torch.linspace(-nx/2 * cell_side, nx/2 * cell_side, nx + 1) + x
    grid_y = torch.linspace(-ny/2 * cell_side, ny/2 * cell_side, ny + 1) + y
    return torch.Tensor([[
        v
        for xx1, xx2 in zip(grid_x[:-1], grid_x[1:])
        for yy1, yy2 in zip(grid_y[:-1], grid_y[1:])
        for v in is_occupied_with_direction(xx1, yy1, xx2, yy2)
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

    def __call__(self, paths, n_predict=12):
        observed_path = paths[0]
        ped_id = observed_path[0].pedestrian
        with torch.no_grad():
            observed = torch.Tensor([[(r.x, r.y)] for r in observed_path[:9]])
            outputs = self.model(observed, n_predict)[9-1:]

        return [Row(0, ped_id, x, y) for ((x, y),) in outputs]


class OLSTMPredictor(object):
    def __init__(self, model, model_vanilla):
        self.model = model
        self.model_vanilla = model_vanilla

    def save(self, filename):
        with open(filename, 'wb') as f:
            torch.save(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return torch.load(f)

    def others_xy(self, paths, n_predict=12):
        others_xy = defaultdict(dict)
        frames = [r.frame for r in paths[0][:9]]
        frames_set = set(frames)
        pedestrians = set()
        pedestrians_to_predict = set()
        for path in paths[1:]:
            for row in path:
                if row.frame not in frames_set:
                    continue
                others_xy[row.pedestrian][row.frame] = (row.x, row.y)
                pedestrians.add(row.pedestrian)
                if row.frame == frames[-1]:
                    pedestrians_to_predict.add(row.pedestrian)

        predicted = {}
        for ped_id in pedestrians_to_predict:
            xy = [[xy] for _, xy in sorted(others_xy[ped_id].items())]
            if len(xy) < 3:
                continue
            observed = torch.Tensor(xy)
            output = self.model_vanilla(observed, n_predict)[len(observed) - 2:]
            predicted[ped_id] = [xy for (xy,) in output]

        # preserve order
        pedestrians = list(pedestrians)

        # observed part
        obs_result = [[others_xy.get(ped_id, {}).get(frame, (None, None))
                       for ped_id in pedestrians]
                      for frame in frames]
        # predicted part
        pred_result = [[predicted[ped_id][i] if ped_id in predicted else (None, None)
                        for ped_id in pedestrians]
                       for i in range(n_predict)]
        return obs_result + pred_result

    def __call__(self, observed_paths, n_predict=12):
        observed_path = observed_paths[0]
        ped_id = observed_path[0].pedestrian
        with torch.no_grad():
            observed = torch.Tensor([[(r.x, r.y)] for r in observed_path[:9]])

            others_xy = self.others_xy(observed_paths, n_predict)
            outputs = self.model(observed, others_xy)[9-1:]

        return [Row(0, ped_id, x, y) for ((x, y),) in outputs]
