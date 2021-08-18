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
        self.batch_size = self._config['batch_size']
        self.tau = self._ns_config['history_duration'] + 1
        self.M = int((self._ns_config['forward_lane'] + self._ns_config['backward_lane']) // (
            self._ns_config['precision_lane']))
        self.in_size = self._ns_config['nb_coordinates']
        self.N = self._config['nb_lane_candidates']
        self.h = self._ns_config['prediction_duration']
        self.bidirectional = False
        self.nb_coordinates = self._ns_config['nb_coordinates']

        # #######################################################################################
        # TODO: Restart and see what we can improve
        # TODO: Is there a way to do this graphically/automatically based on the data we provide?
        self.dropout = torch.nn.Dropout(self.drop)
        self.tfe_h_cnn_1 = torch.nn.Conv1d(in_channels=self.nb_coordinates, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.tfe_h_cnn_2=torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.tfe_h_lstm=torch.nn.LSTM(
                input_size=self.tau - 2, hidden_size=512,
                num_layers=1, bidirectional=self.bidirectional,
                batch_first=True)
        self.tfe_h_lstm = torch.nn.LSTM(
            input_size=self.tau, hidden_size=512,
            num_layers=1, bidirectional=self.bidirectional,
            batch_first=True) # test à retirer
        self.tfe_n_cnn_1=torch.nn.Conv1d(
                in_channels=self.nb_coordinates,
                out_channels=64, kernel_size=2,
                stride=1, padding=0)
        self.tfe_n_cnn_2=torch.nn.Conv1d(
                in_channels=64, out_channels=64,
                kernel_size=2, stride=1, padding=0)

        self.tfe_n_lstm=torch.nn.LSTM(
                input_size=self.tau - 2, hidden_size=512,
                num_layers=1, bidirectional=self.bidirectional,
                batch_first=True)

        self.tfe_l_cnn_1=torch.nn.Conv1d(
                in_channels=2, out_channels=64,
                kernel_size=3, stride=1,
                padding=1)

        self.tfe_l_cnn_2=torch.nn.Conv1d(
                in_channels=64, out_channels=64,
                kernel_size=3, stride=1,
                padding=1)

        self.tfe_l_cnn_3=torch.nn.Conv1d(
                in_channels=64, out_channels=96,
                kernel_size=3, stride=1,
                padding=1)

        self.tfe_l_cnn_4=torch.nn.Conv1d(
                in_channels=96, out_channels=64,
                kernel_size=3, stride=1,
                padding=1)

        self.tfe_l_lstm=torch.nn.LSTM(
                input_size=self.M, hidden_size=2048,
                num_layers=1, bidirectional=self.bidirectional,
                batch_first=True)

        self.tfe_fc_1 = torch.nn.Linear(in_features=3072 * 2 if self.bidirectional else 3072, out_features=2048)
        self.tfe_fc_2 = torch.nn.Linear(in_features=2048, out_features=2048)
        self.tfe_fc_3 = torch.nn.Linear(in_features=2048, out_features=1024)
        self.tfe_fc_4 = torch.nn.Linear(in_features=1024, out_features=1024)
        self.tfe_fc_5 = torch.nn.Linear(in_features= 1024 + 3072 * 2 if self.bidirectional else 6656, out_features=self.h * self.nb_coordinates)
        self.tfe_fc_6 = torch.nn.Linear(in_features=512, out_features=self.h * self.nb_coordinates)

        self.la_fc1 = torch.nn.Linear(in_features=1024 * self.N, out_features=512)
        self.la_fc2 = torch.nn.Linear(in_features=512, out_features=512)
        self.la_fc3 = torch.nn.Linear(in_features=512, out_features=256)
        self.la_fc4 = torch.nn.Linear(in_features=256, out_features=256)
        self.la_fc5 = torch.nn.Linear(in_features=256, out_features=64)
        self.la_fc6 = torch.nn.Linear(in_features=64, out_features=64)
        self.la_fc7 = torch.nn.Linear(in_features=64, out_features=self.N)

        self.softmax = torch.nn.Softmax(dim=1)
        self._mtp_fc = torch.nn.ModuleList([torch.nn.Linear(in_features=1536, out_features=512), torch.nn.Linear(in_features=512, out_features=512),
                                             torch.nn.Linear(in_features=512, out_features=256)])
        self._mtp_fc_k = torch.nn.ModuleList([self._mtp_fc for k in range(self.k)])



        self._mtp_fc_shared = torch.nn.ModuleList([torch.nn.Linear(in_features=256, out_features=256),
                                                   torch.nn.Linear(in_features=256, out_features=self.h * self.nb_coordinates)])

        self.double()


    def _tfe_history(self, history):  # history: [B, tau, nb_coordinates]
        # first_layer = self.tfe_h_cnn_1(history)  # first_layer: [B, 64, tau - 1]
        # second_layer = self.tfe_h_cnn_2(first_layer)  # second_layer: [B, 64, tau - 2]
        out_lstm, (hidden, cell) = self.tfe_h_lstm(history) # second_layer)  # tfe_hidden: [1, B, 512]
        if self.tfe_h_lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
            # hidden = [batch size, hidden dim * 2]
        else:
            hidden = self.dropout(hidden[-1])
        #tfe_output = out_lstm[:, -1, :]  # tfe_output: [B, 2048]
        return hidden

    def _tfe_neighbors(self, neighbors):  # neighbor: [N, B, tau, nb_coordinates]
        return list(map(self._tfe_neighbor, neighbors))

    def _tfe_neighbor(self, neighbor):  # neighbor: [B, tau, nb_coordinates]
        first_layer = self.tfe_n_cnn_1(neighbor)  # first_layer: [B, 64, tau - 1]
        second_layer = self.tfe_n_cnn_2(first_layer)  # second_layer: [B, 64, tau - 2]
        out_lstm, (hidden, cell) = self.tfe_n_lstm(second_layer)  # tfe_hidden: [1, B, 512]
        if self.tfe_n_lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
            # hidden = [batch size, hidden dim * 2]
        else:
            hidden = self.dropout(hidden[-1])

        #tfe_output = out_lstm[:, -1, :]  # tfe_output: [B, 2048]
        return hidden

    def _tfe_lane(self, lane):  # lane: [B, 2, M]
        first_layer = self.tfe_l_cnn_1(lane)  # first_layer: [B, 64, M]
        second_layer = self.tfe_l_cnn_2(first_layer)  # first_layer: [B, 64, M]
        third_layer = self.tfe_l_cnn_3(second_layer)  # third_layer: [B, 96, M]
        fourth_layer = self.tfe_l_cnn_4(third_layer)  # fourth_layer: [B, 64, M]
        out_lstm, (hidden, cell) = self.tfe_l_lstm(fourth_layer)  # tfe_hidden: [1, B, 2048]
        if self.tfe_l_lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
            # hidden = [batch size, hidden dim * 2]
        else:
            hidden = self.dropout(hidden[-1])
        #tfe_output = out_lstm[:,-1,:]  # tfe_output: [B, 2048]
        return hidden

    def _tfe_lanes(self, lanes):  # lanes: [N, B, 2, M]
        return list(map(self._tfe_lane, lanes))

    def _fm(self, concatenated):
        first_layer = self.tfe_fc_1(concatenated)
        second_layer = self.tfe_fc_2(F.relu(first_layer))
        third_layer = self.tfe_fc_3(F.relu(second_layer))
        fourth_layer = self.tfe_fc_4(F.relu(third_layer))
        return fourth_layer

    def forward(self, history, lanes, neighbors):

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

        layer1 = self.la_fc1(out_l)
        layer2 = self.la_fc2(layer1)
        layer3 = self.la_fc3(layer2)
        layer4 = self.la_fc4(layer3)
        layer5 = self.la_fc5(layer4)
        layer6 = self.la_fc6(layer5)
        layer7 = self.la_fc7(layer6)
        t = False
        if t:
            #out = torch.cat([torch.cat(eps, 1), tfe_history], 1)
            res = torch.unsqueeze(self.tfe_fc_6(tfe_history), 0)
            return res, layer7
        out_la = self.softmax(layer7)

        # Feature Attention
        total = np.zeros(eps[0].shape)

        for weights, epsilons in zip(out_la.permute(1, 0), eps):  # For each n in N
            # epsilon: B x 1024
            # weights: B
            # Weight is a list of N

            n_total = torch.from_numpy(np.zeros(eps[0].shape))
            #n_total = nn.ModuleList(n_total)
            for index, (weight, epsilon) in enumerate(zip(weights, epsilons)):
                # Weight: int
                # Epsilon: list(1024)
                n_total[index] = torch.mul(epsilon, weight) #.detach().numpy()
            total = n_total

        # Last Concatenation (of LA Block)
        final_epsilon = torch.cat([tfe_history, total], 1).float()
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

        return torch.stack(lane_predictions).float(), layer7.float()

