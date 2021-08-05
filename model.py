from time import time_ns

import numpy as np
import torch.nn
import torch.nn as nn
import yaml


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

        # #######################################################################################
        # TODO: Restart and see what we can improve
        # TODO: Is there a way to do this graphically/automatically based on the data we provide?

        self.hist_nbrs_cnn_1 = torch.nn.Conv1d(in_channels=2, out_channels=32, kernel_size=2, stride=1, padding=0)
        self.hist_nbrs_cnn_2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=0)
        self.hist_nbrs_lstm = torch.nn.LSTM(input_size=32, hidden_size=64, num_layers=1, bidirectional=False,
                                            batch_first=True)

        self.lanes_cnn1 = torch.nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.lanes_cnn2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.lanes_cnn3 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lanes_cnn4 = torch.nn.Conv1d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1)

        # 96, 2018
        self.lanes_lstm = torch.nn.LSTM(input_size=96, hidden_size=512, num_layers=1, bidirectional=False,
                                        batch_first=True)

        self.fc1 = torch.nn.Linear(in_features=133504, out_features=2048)
        self.fc2 = torch.nn.Linear(in_features=2048, out_features=2048)
        self.fc3 = torch.nn.Linear(in_features=2048, out_features=1024)
        self.fc4 = torch.nn.Linear(in_features=1024, out_features=1024)

        # Lane attention block (for the moment not implemented)
        # TODO: Implement
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
                torch.nn.Linear(in_features=1216, out_features=512),
                torch.nn.Linear(in_features=512, out_features=512),
                torch.nn.Linear(in_features=512, out_features=256)
            ])

        self._mtp_fc_shared = [
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.Linear(in_features=256, out_features=self.h * 2)
        ]

    def forward(self, history, lanes, neighbors):

        # TODO: Use this:
        COORDINATES_SIZE = 2

        # history = [batch_size, 2, tau]

        # two cnn layers for history and neighbors
        cnn_hist = self.hist_nbrs_cnn_2((self.hist_nbrs_cnn_1(history)))
        # cnn_hist = [batch_size, 64, tau - 2 ]
        print(cnn_hist.shape, self.batch_size, self.tau, 64)  # 16, 64, 3
        cnn_hist = torch.reshape(cnn_hist, (self.batch_size, self.tau - 2, cnn_hist.shape[1]))
        lstm_hist, _ = self.hist_nbrs_lstm(cnn_hist)
        # lstm_hist = [batch_size, tau - 2, 512]
        lstm_hist = lstm_hist.reshape(self.batch_size, (self.tau - 2) * lstm_hist.shape[2])
        print("lstm_hist.shape : ", lstm_hist.shape)

        eps = []
        for (nbr, lane) in zip(neighbors, lanes):
            # nbr = [batch_size, 2, tau]
            cnn_nbr = self.hist_nbrs_cnn_2((self.hist_nbrs_cnn_1(nbr)))
            # cnn_nbr = [batch_size, 64, tau - 2]
            cnn_nbr = torch.reshape(cnn_nbr, (self.batch_size, self.tau - 2, cnn_nbr.shape[1]))
            lstm_nbr, _ = self.hist_nbrs_lstm(cnn_nbr)
            # lstm_nbr = [batch_size, tau, 512]
            # lane = [batch_size, 2, M]
            cnn_lane = self.lanes_cnn4(self.lanes_cnn3(self.lanes_cnn2(self.lanes_cnn1(lane))))
            # cnn_lane = [batch_size, 64, M]
            cnn_lane = torch.reshape(cnn_lane, (self.batch_size, self.M, cnn_lane.shape[1]))
            lstm_lane, _ = self.lanes_lstm(cnn_lane)
            # lstm_lane = [batch_size, M, 2048]

            # concat
            #  3, 3, 260

            lstm_nbr = lstm_nbr.reshape(self.batch_size, (self.tau - 2) * lstm_nbr.shape[2])
            print("lstm_nbr.shape : ", lstm_nbr.shape)
            lstm_lane = lstm_lane.reshape(self.batch_size, self.M * lstm_lane.shape[2])
            print("lstm_lane.shape : ", lstm_lane.shape)
            out = torch.cat((lstm_hist, lstm_lane, lstm_nbr), 1)
            # out = torch.cat((out, lstm_nbr), 1)
            print("out.shape : ", out.shape)
            epsilon_i = self.fc4(self.fc3(self.fc2(self.fc1(out))))
            eps.append(epsilon_i)

        concat = torch.cat(eps, 1)

        # Lane Attention #########
        out_la = concat
        for layer in self._la_fc:
            out_la = layer(out_la)
        out_la = self.softmax(out_la)
        # ########################
        print('outlashape : ', out_la.shape)
        total = np.zeros(eps[0].shape)

        for weights, epsilons in zip(out_la.permute(1, 0), eps):  # For each n in N
            # Weight is a list of N
            n_total = np.zeros(eps[0].shape)
            for index, (weight, epsilon) in enumerate(zip(weights, epsilons)):
                # Weight: int
                # Epsilon: list(1024)
                n_total[index] = torch.mul(epsilon, weight).detach().numpy()
            total += n_total

        print(torch.tensor(total).shape, lstm_hist.shape)
        final_epsilon = torch.cat([lstm_hist, torch.tensor(total)],1).float()

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
