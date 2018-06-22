import random
import torch
import torch.nn.functional as F

from .data import Row


class SocialLSTM(torch.nn.Module):

    def __init__(self, embedding_dim=64, hidden_dim=32):
        super(SocialLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.input_embeddings = torch.nn.Sequential(
            torch.nn.Linear(2, embedding_dim),
            torch.nn.ReLU(),
        )
        self.encoder = torch.nn.LSTMCell(embedding_dim, hidden_dim)
        self.decoder = torch.nn.LSTMCell(embedding_dim, hidden_dim)
        self.hidden2vel = torch.nn.Linear(hidden_dim, 2)
        # self.hidden2vel = torch.nn.Sequential(
        #     torch.nn.Linear(hidden_dim, embedding_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(embedding_dim, 2),
        # )

    def discrete_2d_embed(self, xy, range_, embedding_dim_1d):
        embeddings = torch.zeros((xy.shape[0], embedding_dim_1d * embedding_dim_1d))
        indices = (xy - range_[0]) / (range_[1] - range_[0]) * embedding_dim_1d
        indices = torch.clamp(indices, 0, embedding_dim_1d)
        indices = (indices[:, 0:1] * embedding_dim_1d + indices[:, 1:2]).long()
        indices = torch.clamp(indices, 0, embedding_dim_1d * embedding_dim_1d - 1)
        embeddings.scatter_(1, indices, 1.0)
        return embeddings

    def forward(self, observed, n_predict=11):
        """forward

        observed shape is (seq, batch, observables)
        """
        hidden_state = (torch.zeros(1, self.hidden_dim),
                        torch.zeros(1, self.hidden_dim))

        # tag observations
        # obs_tags = torch.zeros((observed.shape[0], observed.shape[1], 1))
        # tagged_observed = torch.cat([observed, obs_tags], dim=2)

        # initialize lstm state with a start tag
        # start_tag = torch.zeros(tagged_observed[0].shape)
        # start_tag[:, 2] = 1.0
        # input = self.input_embeddings(start_tag)
        # input = F.relu(input)
        # input = torch.zeros((observed[0].shape[0], 64))
        # input[:, 63] = 1.0
        # hidden_state = self.encoder(input, hidden_state)

        predicted = []
        for vel in observed:
            emb = self.input_embeddings(vel)
            # emb = self.discrete_2d_embed(vel, (-0.2, 0.2), 8)
            hidden_state = self.encoder(emb, hidden_state)

            new_vel = self.hidden2vel(hidden_state[0])
            predicted.append(new_vel)

        start_tag = torch.zeros(observed[0].shape[0], 64)
        start_tag[:, 63] = 1.0
        # hidden_state = (hidden_state[0], torch.zeros(1, self.hidden_dim))
        hidden_state = self.decoder(start_tag, hidden_state)

        for _ in range(n_predict):
            new_vel = self.hidden2vel(hidden_state[0])
            predicted.append(new_vel)

            # tag input
            # vel_tags = torch.zeros((new_vel.shape[0], 1))
            # tagged_new_vel = torch.cat([new_vel, vel_tags], dim=1)

            # update LSTM
            emb = self.input_embeddings(new_vel)
            # emb = self.discrete_2d_embed(new_vel, (-0.2, 0.2), 8)
            hidden_state = self.decoder(emb, hidden_state)

        return torch.stack(predicted, dim=0)


def train(paths, epochs=100):
    model = SocialLSTM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(1, epochs + 1):
        print('epoch', epoch)
        if epoch == 30:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.01
        if epoch == 70:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
        random.shuffle(paths)
        epoch_loss = 0.0
        for path in paths:
            observed = torch.Tensor([[(r.x, r.y)] for r in path[:9]])
            target = torch.Tensor([[(r.x, r.y)] for r in path[1:]])
            velocity_inputs = observed[1:] - observed[:-1]
            velocity_targets = target[1:] - target[:-1]

            model.zero_grad()
            output = model(velocity_inputs)
            # loss = F.l1_loss(output, velocity_targets)
            loss = torch.pow(output - velocity_targets, 2)
            loss[loss < (0.03**2)] = 0.0
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('loss', epoch_loss / len(paths))

    return Predictor(model)


class Predictor(object):
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
