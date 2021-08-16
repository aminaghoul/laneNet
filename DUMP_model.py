self.hist_nbrs_cnn_1 = torch.nn.Conv1d(in_channels=self.nb_coordinates, out_channels=64, kernel_size=2,
                                               stride=1, padding=0)
        self.batchnorm1 = torch.nn.BatchNorm1d(64)
        self.hist_nbrs_cnn_2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=0)
        self.hist_nbrs_lstm = torch.nn.LSTM(input_size=64, hidden_size=512, num_layers=1, bidirectional=False,
                                           batch_first=True)

        self.lanes_cnn1 = torch.nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lanes_cnn2 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lanes_cnn3 = torch.nn.Conv1d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.lanes_cnn4 = torch.nn.Conv1d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)

        # 96, 2018
        self.lanes_lstm = torch.nn.LSTM(input_size=96, hidden_size=2048, num_layers=1, bidirectional=False,
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


"""for i in range(self.k):
    # input size, specification, output size
    # B X 1536,   u512,          B X 512
    # B X 512,    u512,          B X 512
    # B X 512,    u256,          B X 256
    self._mtp_fc_k.append([
        torch.nn.Linear(in_features=1536, out_features=512),
        torch.nn.Linear(in_features=512, out_features=512),
        torch.nn.Linear(in_features=512, out_features=256)
    ])"""

"""self.new = dict(
         tfe_h_cnn_1=torch.nn.Conv1d(
             in_channels=self.nb_coordinates,
             out_channels=64, kernel_size=2,
             stride=1, padding=0),
         tfe_h_cnn_2=torch.nn.Conv1d(
             in_channels=64, out_channels=64,
             kernel_size=2, stride=1, padding=0),
         tfe_h_lstm=torch.nn.LSTM(
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
         tfe_n_lstm=torch.nn.LSTM(
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
         tfe_l_lstm=torch.nn.LSTM(
             input_size=self.M, hidden_size=2048,
             num_layers=1, bidirectional=False,
             batch_first=True),
         tfe_fc_1=torch.nn.Linear(in_features=3072, out_features=2048),
         tfe_fc_2=torch.nn.Linear(in_features=2048, out_features=2048),
         tfe_fc_3=torch.nn.Linear(in_features=2048, out_features=1024),
         tfe_fc_4=torch.nn.Linear(in_features=1024, out_features=1024),
         tfe_fc_5=torch.nn.Linear(in_features=6656, out_features=self.h * self.nb_coordinates),
         la_fc=[
             torch.nn.Linear(in_features=1024 * self.N, out_features=512),
             torch.nn.Linear(in_features=512, out_features=512),
             torch.nn.Linear(in_features=512, out_features=256),
             torch.nn.Linear(in_features=256, out_features=256),
             torch.nn.Linear(in_features=256, out_features=64),
             torch.nn.Linear(in_features=64, out_features=64),
             torch.nn.Linear(in_features=64, out_features=self.N),
         ]
     )"""