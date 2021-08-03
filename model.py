from time import time_ns

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
        self.tau = self._ns_config['history_duration'] - 1
        self.M = (self._ns_config['forward_lane'] + self._ns_config['backward_lane']) // (
        self._ns_config['precision_lane'])
        self.in_size = self._ns_config['nb_columns']

        self.hidden_size = self._ns_config['hidden_size']

        # #######################################################################################
        # TODO: Restart and see what we can improve
        # TODO: Is there a way to do this graphically/automatically based on the data we provide?

        self.hist_nbrs_cnn_1 = torch.nn.Conv1d(in_channels=2, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.hist_nbrs_cnn_2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.hist_nbrs_lstm = torch.nn.LSTM(input_size=64, hidden_size=512, num_layers=1, bidirectional=False,
                                            batch_first=True)

        self.lanes_cnn1 = torch.nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lanes_cnn2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lanes_cnn3 = torch.nn.Conv1d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.lanes_cnn4 = torch.nn.Conv1d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.lanes_lstm = torch.nn.LSTM(input_size=96, hidden_size=2048, num_layers=1, bidirectional=False,
                                        batch_first=True)

        self.fc1 = torch.nn.Linear(in_features=3072, out_features=2048)
        self.fc2 = torch.nn.Linear(in_features=2048, out_features=2048)
        self.fc3 = torch.nn.Linear(in_features=2048, out_features=1024)
        self.fc4 = torch.nn.Linear(in_features=1024, out_features=1024)

        # Lane attention block (for the moment not implemented)
        # TODO: Implement

        # Trajectory Generator
        # There are K different FC, each connected to another FC, this
        #  one is shared, which will output the Vfk (k being the index
        #  between 1 and K)
        # Input for those are the concatenation between the output
        #  of the LSTM of Vp and the average (weighted using the
        #  lane attention block) of all the lane's LSTM outputs.
        self._k: int = ...  # There will be K outputs
        self._mtp_fc_k = list()
        for i in range(self._k):
            # input size, specification, output size
            # B X 1536,   u512,          B X 512
            # B X 512,    u512,          B X 512
            # B X 512,    u256,          B X 256
            self._mtp_fc_k.append([
                torch.nn.Linear(in_features=1536, out_features=512),
                torch.nn.Linear(in_features=512, out_features=512),
                torch.nn.Linear(in_features=512, out_features=256),
            ])

        # TODO: DÉFINIR 7 FC DU LA BLOC

    def forward(self, history, lanes, neighbors):
        # TODO: Implement the following
        # This if case is to avoid PyCharm noticing it
        if time_ns():
            raise NotImplementedError()
        # history = [batch_size, 2, tau]

        # two cnn layers for history and neighbors
        cnn_hist = self.hist_nbrs_cnn_2((self.hist_nbrs_cnn_1(history)))
        # cnn_hist = [batch_size, 64, tau - 2]
        cnn_hist = torch.reshape(cnn_hist, (self.batch_size, self.tau - 2, 64))
        lstm_hist, _ = self.hist_nbrs_lstm(cnn_hist)
        # lstm_hist = [batch_size, tau - 2, 512]

        eps = []
        for (nbr, lane) in zip(neighbors, lanes):
            # nbr = [batch_size, 2, tau]
            cnn_nbr = self.hist_nbrs_cnn_2((self.hist_nbrs_cnn_1(nbr)))
            # cnn_nbr = [batch_size, 64, tau - 2]
            cnn_nbr = torch.reshape(cnn_nbr, (self.batch_size, self.tau - 2, 64))
            lstm_nbr, _ = self.hist_nbrs_lstm(cnn_nbr)
            # lstm_nbr = [batch_size, tau - 2, 512]
            # lane = [batch_size, 2, M]
            cnn_lane = self.lanes_cnn4(self.lanes_cnn3(self.lanes_cnn2(self.lanes_cnn1(lane))))
            # cnn_lane = [batch_size, 64, M]
            cnn_lane = torch.reshape(cnn_lane, (self.batch_size, self.M, 64))
            lstm_lane, _ = self.lanes_lstm(cnn_lane)
            # lstm_lane = [batch_size, M, 2048]

            # concat
            out = torch.cat((lstm_hist, lstm_nbr, lstm_lane), 2)
            epsilon_i = self.fc4(self.fc3(self.fc2(self.fc1(out))))
            eps.append(epsilon_i)

        concat = torch.cat(eps, 1)
        out_la = self.fc7(self.fc6(self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(out)))))))

        return eps
        # trajectories = self.mtp_fc([fc(epsilon) for fc in self.mtp_fcs])
