import torch

from ..data import Row


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

        # Predict the parameters of a multivariate normal:
        # mu_vel_x, mu_vel_y, sigma_vel_x, sigma_vel_y, rho
        self.hidden2normal = torch.nn.Linear(hidden_dim, 5)

    def forward(self, observed, n_predict=12):
        """forward

        observed shape is (seq, batch, observables)
        """
        batch_size = observed.shape[1]
        hidden_cell_state = (torch.zeros(batch_size, self.hidden_dim),
                             torch.zeros(batch_size, self.hidden_dim))

        normals = []
        positions = []
        for obs1, obs2 in zip(observed[:-1], observed[1:]):
            emb = self.input_embeddings(obs2 - obs1)
            hidden_cell_state = self.lstm(emb, hidden_cell_state)

            normal = self.hidden2normal(hidden_cell_state[0])
            normals.append(normal)
            new_pos = obs2 + normal[:, :2]
            positions.append(new_pos)

        # the previous loop ends with a velocity prediction, so only need to
        # predict n_predict - 1 times
        for _ in range(n_predict - 1):
            emb = self.input_embeddings((positions[-1] - positions[-2]).detach())  # DETACH!!!
            hidden_cell_state = self.lstm(emb, hidden_cell_state)

            normal = self.hidden2normal(hidden_cell_state[0])
            normals.append(normal)
            new_pos = positions[-1] + normal[:, :2]
            positions.append(new_pos)

        return torch.stack(normals if self.training else positions, dim=0)


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
        self.model.eval()

        observed_path = paths[0]
        ped_id = observed_path[0].pedestrian
        with torch.no_grad():
            observed = torch.Tensor([[(r.x, r.y)] for r in observed_path[:9]])
            outputs = self.model(observed, n_predict)[9-1:]

        return [Row(0, ped_id, x, y) for ((x, y),) in outputs]
