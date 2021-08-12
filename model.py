from time import time_ns

import numpy as np
import torch.nn
import torch.nn as nn
import yaml
import torch.nn.functional as F

import io

_old_open = io.open


def _new_open(*args, **kwargs):
    print('Open', args[0])
    return _old_open(*args, **kwargs)


io.open = _new_open


class LaneNet(nn.Module):

    # Initialization
    def __init__(self, config_path):
        super(LaneNet, self).__init__()

        with open(config_path, 'r') as yaml_file:
            self._config = yaml.safe_load(yaml_file)['ln_args']
        with open(config_path, 'r') as yaml_file:
            self._ns_config = yaml.safe_load(yaml_file)['ns_args']

        self.use_cuda = self._config['use_cuda']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = self._config['train_flag']
        self.drop = self._config['dropout']
        self.k = self._config['number_of_predictions']
        self.batch_size = self._ns_config['batch_size']
        self.tau = self._ns_config['history_duration'] + 1
        self.M = int((self._ns_config['forward_lane'] + self._ns_config['backward_lane']) // (
            self._ns_config['precision_lane']))
        self.in_size = self._ns_config['nb_columns']
        self.N = self._ns_config['nb_lane_candidates']
        self.hidden_size = self._ns_config['hidden_size']
        self.h = self._ns_config['prediction_duration']
        self.num_heads = self._ns_config['num_heads']

        self.nb_coordinates = self._ns_config['nb_coordinates']

        # #######################################################################################
        # TODO: Restart and see what we can improve
        # TODO: Is there a way to do this graphically/automatically based on the data we provide?
        self.dropout = torch.nn.Dropout(self.drop)
        self.hist_nbrs_cnn_1 = torch.nn.Conv1d(in_channels=self.nb_coordinates, out_channels=64, kernel_size=2,
                                               stride=1, padding=0)
        self.batchnorm1 = torch.nn.BatchNorm1d(64)
        self.hist_nbrs_cnn_2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.hist_nbrs_lstm = torch.nn.GRU(input_size=64, hidden_size=512, num_layers=1, bidirectional=False,
                                           batch_first=True)

        self.lanes_cnn1 = torch.nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lanes_cnn2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lanes_cnn3 = torch.nn.Conv1d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.lanes_cnn4 = torch.nn.Conv1d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)

        # 96, 2018
        self.lanes_lstm = torch.nn.GRU(input_size=96, hidden_size=2048, num_layers=1, bidirectional=False,
                                       batch_first=True)

        self.fc1 = torch.nn.Linear(in_features=3072, out_features=2048)
        self.fc2 = torch.nn.Linear(in_features=2048, out_features=2048)
        self.fc3 = torch.nn.Linear(in_features=2048, out_features=1024)
        self.fc4 = torch.nn.Linear(in_features=1024, out_features=1024)

        # Lane attention block (for the moment not implemented)
        self._la_fc = [
            torch.nn.Linear(in_features=1024 * self.N, out_features=512),
            torch.nn.Linear(in_features=512, out_features=512),
            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.Linear(in_features=256, out_features=64),
            torch.nn.Linear(in_features=64, out_features=64),
            torch.nn.Linear(in_features=64, out_features=self.N),
        ]
        self.softmax = torch.nn.Softmax(dim=1)

        # Trajectory Generator
        # There are K different FC, each connected to another FC, this
        #  one is shared, which will output the Vfk (k being the index
        #  between 1 and K)
        # Input for those are the concatenation between the output
        #  of the LSTM of Vp and the average (weighted using the
        #  lane attention block) of all the lane's LSTM outputs.
        self._mtp_fc_k = list()
        for i in range(self.k):
            # input size, specification, output size
            # B X 1536,   u512,          B X 512
            # B X 512,    u512,          B X 512
            # B X 512,    u256,          B X 256
            self._mtp_fc_k.append([
                torch.nn.Linear(in_features=1536, out_features=512),
                torch.nn.Linear(in_features=512, out_features=512),
                torch.nn.Linear(in_features=512, out_features=256)
            ])

        self._mtp_fc_shared = [
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.Linear(in_features=256, out_features=self.h * self.nb_coordinates)
        ]

        self.new = dict(
            tfe_h_cnn_1=torch.nn.Conv1d(
                in_channels=self.nb_coordinates,
                out_channels=64, kernel_size=2,
                stride=1, padding=0),
            tfe_h_cnn_2=torch.nn.Conv1d(
                in_channels=64, out_channels=64,
                kernel_size=2, stride=1, padding=0),
            tfe_h_lstm=torch.nn.GRU(
                input_size=self.tau - 2, hidden_size=512,
                num_layers=1, bidirectional=False,
                batch_first=True),
            tfe_n_cnn_1=torch.nn.Conv1d(
                in_channels=self.nb_coordinates,
                out_channels=64, kernel_size=2,
                stride=1, padding=0),
            tfe_n_cnn_2=torch.nn.Conv1d(
                in_channels=64, out_channels=64,
                kernel_size=2, stride=1, padding=0),
            tfe_n_lstm=torch.nn.GRU(
                input_size=self.tau - 2, hidden_size=512,
                num_layers=1, bidirectional=False,
                batch_first=True),
            tfe_l_cnn_1=torch.nn.Conv1d(
                in_channels=2, out_channels=64,
                kernel_size=3, stride=1,
                padding=1),
            tfe_l_cnn_2=torch.nn.Conv1d(
                in_channels=64, out_channels=64,
                kernel_size=3, stride=1,
                padding=1),
            tfe_l_cnn_3=torch.nn.Conv1d(
                in_channels=64, out_channels=96,
                kernel_size=3, stride=1,
                padding=1),
            tfe_l_cnn_4=torch.nn.Conv1d(
                in_channels=96, out_channels=64,
                kernel_size=3, stride=1,
                padding=1),
            tfe_l_lstm=torch.nn.GRU(
                input_size=self.M, hidden_size=2048,
                num_layers=1, bidirectional=False,
                batch_first=True),
            tfe_fc_1=torch.nn.Linear(in_features=3072, out_features=2048),
            tfe_fc_2=torch.nn.Linear(in_features=2048, out_features=2048),
            tfe_fc_3=torch.nn.Linear(in_features=2048, out_features=1024),
            tfe_fc_4=torch.nn.Linear(in_features=1024, out_features=1024),
            la_fc=[
                torch.nn.Linear(in_features=1024 * self.N, out_features=512),
                torch.nn.Linear(in_features=512, out_features=512),
                torch.nn.Linear(in_features=512, out_features=256),
                torch.nn.Linear(in_features=256, out_features=256),
                torch.nn.Linear(in_features=256, out_features=64),
                torch.nn.Linear(in_features=64, out_features=64),
                torch.nn.Linear(in_features=64, out_features=self.N),
            ]
        )

    def _tfe_history(self, history):  # history: [B, tau, nb_coordinates]
        first_layer = self.new['tfe_h_cnn_1'](history)  # first_layer: [B, 64, tau - 1]
        second_layer = self.new['tfe_h_cnn_2'](first_layer)  # second_layer: [B, 64, tau - 2]
        _, tfe_hidden = self.new['tfe_h_lstm'](second_layer)  # tfe_hidden: [1, B, 512]
        tfe_output = tfe_hidden.squeeze()  # tfe_output: [B, 512]
        return tfe_output

    def _tfe_neighbors(self, neighbors):  # neighbor: [N, B, tau, nb_coordinates]
        return list(map(self._tfe_neighbor, neighbors))

    def _tfe_neighbor(self, neighbor):  # neighbor: [B, tau, nb_coordinates]
        first_layer = self.new['tfe_n_cnn_1'](neighbor)  # first_layer: [B, 64, tau - 1]
        second_layer = self.new['tfe_n_cnn_2'](first_layer)  # second_layer: [B, 64, tau - 2]
        _, tfe_hidden = self.new['tfe_n_lstm'](second_layer)  # tfe_hidden: [1, B, 512]
        tfe_output = tfe_hidden.squeeze()  # tfe_output: [B, 512]
        return tfe_output

    def _tfe_lane(self, lane):  # lane: [B, 2, M]
        first_layer = self.new['tfe_l_cnn_1'](lane)  # first_layer: [B, 64, M]
        second_layer = self.new['tfe_l_cnn_2'](first_layer)  # first_layer: [B, 64, M]
        third_layer = self.new['tfe_l_cnn_3'](second_layer)  # third_layer: [B, 96, M]
        fourth_layer = self.new['tfe_l_cnn_4'](third_layer)  # fourth_layer: [B, 64, M]
        _, tfe_hidden = self.new['tfe_l_lstm'](fourth_layer)  # tfe_hidden: [1, B, 2048]
        tfe_output = tfe_hidden.squeeze()  # tfe_output: [B, 2048]
        return tfe_output

    def _tfe_lanes(self, lanes):  # lanes: [N, B, 2, M]
        return list(map(self._tfe_lane, lanes))

    def _fm(self, concatenated):
        first_layer = self.new['tfe_fc_1'](concatenated)
        second_layer = self.new['tfe_fc_2'](F.relu(first_layer))
        third_layer = self.new['tfe_fc_3'](F.relu(second_layer))
        fourth_layer = self.new['tfe_fc_4'](F.relu(third_layer))
        return fourth_layer

    def forward(self, history, lanes, neighbors, selected_lanes):
        tfe_history = self._tfe_history(history)
        # tfe_history: [32, 512]
        tfe_lanes = self._tfe_lanes(lanes)
        # tfe_lanes: N x [B, 2048]
        tfe_neighbors = self._tfe_neighbors(neighbors)
        # tfe_neighbors: N x [32, 512]

        # Feature Merger
        eps = []
        for tfe_lane, tfe_neighbor in zip(tfe_lanes, tfe_neighbors):
            # print(tfe_history.shape, tfe_lane.shape, tfe_neighbor.shape)
            concatenated = torch.cat([tfe_history, tfe_lane, tfe_neighbor], 1)
            eps.append(self._fm(concatenated))

        # Lane Selection
        out_l = torch.cat(eps, 1)

        for layer in self.new['la_fc']:
            out_l = layer(F.relu(out_l))
        out_la = self.softmax(out_l)

        # Feature Attention
        total = np.zeros(eps[0].shape)

        for weights, epsilons in zip(out_la.permute(1, 0), eps):  # For each n in N
            # epsilon: B x 1024
            # weights: B
            # Weight is a list of N
            n_total = np.zeros(eps[0].shape)
            for index, (weight, epsilon) in enumerate(zip(weights, epsilons)):
                # Weight: int
                # Epsilon: list(1024)
                n_total[index] = torch.mul(epsilon, weight).detach().numpy()
            total += n_total

        # Last Concatenation (of LA Block)
        final_epsilon = torch.cat([tfe_history, torch.tensor(total)], 1).float()
        # final_epsilon: [B, 1536]

        lane_predictions = []
        for i, layers in enumerate(self._mtp_fc_k):
            output = final_epsilon
            for layer in layers:
                output = layer(F.relu(output))
            lane_predictions.append(output)

        for layer in self._mtp_fc_shared:
            for index, lane in enumerate(lane_predictions):
                lane_predictions[index] = layer(lane)

        return torch.stack(lane_predictions), out_la

    def _forward(self, history, lanes, neighbors):
        eps = []
        for (nbr, lane) in zip(neighbors, lanes):
            # nbr = [batch_size, 2, tau]
            cnn_nbr = self.hist_nbrs_cnn_2(self.batchnorm1(self.hist_nbrs_cnn_1(nbr)))
            # cnn_nbr = [batch_size, 64, tau - 2]
            cnn_nbr = cnn_nbr.permute(0, 2, 1)
            lstm_nbr, _ = self.hist_nbrs_lstm(cnn_nbr)
            # lstm_nbr = [batch_size, tau, 512]
            # lane = [batch_size, 2, M]
            cnn_lane = self.lanes_cnn4(self.lanes_cnn3(self.lanes_cnn2(self.lanes_cnn1(lane))))
            # cnn_lane = [batch_size, 64, M]
            cnn_lane = cnn_lane.permute(0, 2, 1)
            lstm_lane, _ = self.lanes_lstm(cnn_lane)
            # lstm_lane = [batch_size, M, 2048]

            # concat
            #  3, 3, 260

            lstm_nbr = lstm_nbr[:, -1, :]
            # print("lstm_nbr.shape : ", lstm_nbr.shape)
            lstm_lane = lstm_lane[:, -1, :]
            # print("lstm_lane.shape : ", lstm_lane.shape)
            out = torch.cat((lstm_hist, lstm_lane, lstm_nbr), 1)
            # out = torch.cat((out, lstm_nbr), 1)
            # print("out.shape : ", out.shape)
            epsilon_i = self.fc4(self.fc3(self.fc2(self.fc1(out))))
            eps.append(F.relu(epsilon_i))

        concat = torch.cat(eps, 1)

        # Lane Attention #########
        out_la = concat
        for layer in self._la_fc:
            out_la = layer(out_la)
        out_la = self.softmax(out_la)
        # ########################
        # print('outlashape : ', out_la.shape)
        total = np.zeros(eps[0].shape)

        for weights, epsilons in zip(out_la.permute(1, 0), eps):  # For each n in N
            # Weight is a list of N
            n_total = np.zeros(eps[0].shape)
            for index, (weight, epsilon) in enumerate(zip(weights, epsilons)):
                # Weight: int
                # Epsilon: list(1024)
                n_total[index] = torch.mul(epsilon, weight).detach().numpy()
            total += n_total

        # print(torch.tensor(total).shape, lstm_hist.shape)
        final_epsilon = torch.cat([lstm_hist, torch.tensor(total)], 1).float()

        print("final eps : ", final_epsilon.shape)

        lane_predictions = []
        for i, layers in enumerate(self._mtp_fc_k):
            output = final_epsilon
            for layer in layers:
                output = layer(output)
            lane_predictions.append(output)

        for layer in self._mtp_fc_shared:
            for index, lane in enumerate(lane_predictions):
                lane_predictions[index] = layer(lane)

        return torch.stack(lane_predictions)
        # trajectories = self.mtp_fc([fc(epsilon) for fc in self.mtp_fcs])

    def __forward(self):
        # TODO: Use this:
        COORDINATES_SIZE = 2

        # history = [batch_size, 2, tau]

        # two cnn layers for history and neighbors
        cnn_hist = self.hist_nbrs_cnn_2(self.batchnorm1(self.hist_nbrs_cnn_1(history)))
        cnn_hist = F.relu(cnn_hist)
        # cnn_hist = [batch_size, 64, tau - 2 ]
        cnn_hist = cnn_hist.permute(0, 2, 1)
        lstm_hist, _ = self.hist_nbrs_lstm(cnn_hist)

        # lstm_hist = [batch_size, tau - 2, 512]
        lstm_hist = self.dropout(lstm_hist)
        # lstm_hist = torch.flatten(lstm_hist, start_dim=1) # self.batch_size, (self.tau - 2) * lstm_hist.shape[2])
        lstm_hist = lstm_hist[:, -1, :]  # use the last hidden state of LSTM
