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

        # Use gpu flag
        # TODO: use_cuda || cuda_available
        self.use_cuda = self._config['use_cuda']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = self._config['train_flag']

        self.k = self._config['number_of_predictions']
        self.batch_size = self._ns_config['batch_size']
        self.tau = self._ns_config['history_duration'] - 1
        self.M = (self._ns_config['forward_lane'] + self._ns_config['backward_lane']) // (self._ns_config['precision_lane'])
        self.in_size = self._ns_config['nb_columns']

        self.hidden_size = self._ns_config['hidden_size']


        # #######################################################################################

        self.hist_nbrs_cnn_1 = torch.nn.Conv1d(in_channels=2, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.hist_nbrs_cnn_2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.hist_nbrs_lstm = torch.nn.LSTM(input_size=64, hidden_size=512, num_layers=1, bidirectional=False, batch_first=True)

        self.lanes_cnn1 = torch.nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lanes_cnn2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lanes_cnn3 = torch.nn.Conv1d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.lanes_cnn4 = torch.nn.Conv1d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.lanes_lstm = torch.nn.LSTM(input_size=96, hidden_size=2048, num_layers=1, bidirectional=False, batch_first=True)

        self.fc1 = torch.nn.Linear(in_features=3072, out_features=2048)
        self.fc2 = torch.nn.Linear(in_features=2048, out_features=2048)
        self.fc3 = torch.nn.Linear(in_features=2048, out_features=1024)
        self.fc4 = torch.nn.Linear(in_features=1024, out_features=1024)
        """
        self.mtp_fcs = [torch.nn.Linear(
            self.tfe_lstm_out_size + self.tfe_fc_out_size, self.mtp_fcs_out_size) for i in range(self.k)]
        self.mtp_fc = torch.nn.Linear(self.mtp_fcs_out_size, self.mtp_fc_out_size)
        """

    def forward(self, history, lanes, neighbors ):
        # history = [batch_size, 2, tau]
        # neighbors = [batch_size, 2, tau]
        # two cnn layers for history and neighbors
        cnn_hist = self.hist_nbrs_cnn_2((self.hist_nbrs_cnn_1(history)))
        # cnn_hist = [batch_size, 64, tau - 2]
        cnn_nbrs = self.hist_nbrs_cnn_2((self.hist_nbrs_cnn_1(history)))
        # cnn_nbrs = [batch_size, 64, tau - 2]

        # lstm for history and neighbors
        cnn_hist = torch.reshape(cnn_hist, (self.batch_size, self.tau - 2, 64))
        lstm_hist,_ = self.hist_nbrs_lstm(cnn_hist)
        # lstm_hist = [batch_size, tau - 2, 512]
        cnn_nbrs = torch.reshape(cnn_nbrs, (self.batch_size, self.tau - 2, 64))
        lstm_nbrs,_ = self.hist_nbrs_lstm(cnn_nbrs)
        # lstm_nbrs = [batch_size, tau - 2, 512]

        # lanes = [batch_size, 2, M]
        # 4 cnn layers for lanes
        cnn_lanes = self.lanes_cnn4(self.lanes_cnn3(self.lanes_cnn2(self.lanes_cnn1(lanes))))
        # cnn_lanes = [batch_size, 64, M]

        cnn_lanes = torch.reshape(cnn_lanes, (self.batch_size, self.M, 64))


        #lstm for lanes
        lstm_lanes,_ = self.lanes_lstm(cnn_lanes)
        # lstm_lanes = [batch_size, M, 2048]

        #concat
        epsilon = torch.cat((lstm_hist, lstm_nbrs, lstm_lanes), 2)

        out = self.fc4(self.fc3(self.fc2(self.fc1(epsilon))))

        return out
        #trajectories = self.mtp_fc([fc(epsilon) for fc in self.mtp_fcs])

