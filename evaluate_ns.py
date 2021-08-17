import torch
from torch.utils.data import DataLoader
import yaml
import utils as u
import numpy as np

import multiprocessing as mp
import json
from nuscenes.eval.prediction.config import PredictionConfig
from nuscenes.prediction.helper import convert_local_coords_to_global
from nuscenes.eval.prediction.data_classes import Prediction
from nuscenes.eval.prediction.compute_metrics import compute_metrics
from main import NS
import json
import math
import pickle
from random import shuffle

from matplotlib import pyplot as plt

from utils import *
from contextlib import contextmanager
from datetime import timedelta
from inspect import stack
from itertools import chain
from signal import signal, SIGALRM, alarm
from time import time, sleep

import numpy as np
import torch
import yaml
from nuscenes.eval.prediction.config import PredictionConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from main import NS
from model import LaneNet




    # We begin with the dataset,import json
import math
import pickle
from random import shuffle

from matplotlib import pyplot as plt

from utils import *
from contextlib import contextmanager
from datetime import timedelta
from inspect import stack
from itertools import chain
from signal import signal, SIGALRM, alarm
from time import time, sleep

import numpy as np
import torch
import yaml
from nuscenes.eval.prediction.config import PredictionConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from main import NS


class SuperNS(Dataset):
    # TODO: Add more parameters as well as more
    #  attributes to this.
    def __init__(self, split='mini_train'):
        self.__ns = list(NS.every_map('config.yml', split))
        self.__items = list(chain.from_iterable(self.__ns))
        shuffle(self.__items)
        self.helper = self.__ns[0].helper

    def __len__(self):
        # This is the list of every element of
        #  every datasets
        return len(self.__items)

    def __getitem__(self, item):
        # This is the list of every element of
        #  every datasets
        return self.__items[item]


# Read config file
config_file = 'config.yml'
with open(config_file, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize dataset
ts_set = SuperNS('test')

# Initialize data loader:
ts_dl = DataLoader(ts_set,
                   batch_size=4,
                   shuffle=True,
                   num_workers=8)

# Initialize Models:
net = LaneNet('config.yml').float().to(device)
# TODO: do checkpoints
net.load_state_dict(torch.load(config['opt_r']['checkpt_dir'] + '/' + 'best.tar')['model_state_dict'])
for param in net.parameters():
    param.requires_grad = False
net.eval()
# TODO: do this :
initial_state = config['args_mdp']['initial_state']


# Lists of predictions
eval_ade = np.array([])
eval_fde = np.array([])

with mp.Pool(8) as process_pool:

    # Load batch
    for i, (history, future, lanes, neighbors, reference_lane) in enumerate(ts_dl):

        # history : B x tau x 2
        # lanes : B x N x M x 2
        # neighbors : B x N x tau x 2
        # v : future : B x h x 2
        # reference_lane : B x M x 2

        # We start by preprocessing the inputs
        history = torch.permute(history.float().to(device), (0, 2, 1))
        lanes = torch.permute(lanes.float().to(device), (1, 0, 3, 2))
        neighbors = torch.permute(neighbors.float().to(device), (1, 0, 3, 2))
        # history : B x 2 x tau
        # lanes :  N x B x 2 x M
        # neighbors :  N x B x 2 x tau

        v = torch.flatten(future, start_dim=1)
        v = v.float().to(device)
        # v : future : B x (h x 2)

        # Then we do a forward pass
        predict, out_la = net(history, lanes, neighbors)

        # predict : K x B x (hx2)
        print('predshape : ', predict.shape)

        v_hat = predict.permute(1, 0, 2)
        # v_hat : B x K x (hx2)

        all_lanes = lanes.permute(1, 0, 3, 2)

        reference = []
        for index, reference_index in enumerate(reference_lane):
            reference.append(all_lanes[index][reference_index])
        reference = torch.stack(reference).permute(0, 2, 1)

        index = 0
        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            print(history.shape, neighbors.shape, v.shape, predict.permute(1, 0, 2).shape)
            for t_history, t_neighbors, t_future, t_predict, t_lanes, ref in zip(
                    history, neighbors.permute(1, 0, 2, 3),
                    future, predict.permute(1, 0, 2), lanes.permute(1, 0, 2, 3), reference):
                # t_neighbors : N x nb_cordinates x (tau + 1)

                target_x, target_y = [], []
                for x, y, *z in t_history.permute(1, 0):
                    target_y.append(y)
                    target_x.append(x)
                args = [target_x, target_y, '-']
                target_x, target_y, *z = t_future.permute(1, 0)
                args.extend([target_x, target_y, '*'])

                l_x, l_y, *z = ref
                args.extend([l_x, l_y, '*'])

                for neighbor in t_neighbors:
                    neighbors_x, neighbors_y = [], []
                    for n_x, n_y, *z in neighbor:
                        neighbors_x.append(n_x)
                        neighbors_y.append(n_y)
                    args.extend([neighbors_x, neighbors_y, '-'])

                for t_k_predict in t_predict:
                    # predictions_x, predictions_y = [], []
                    # print(t_k_predict.reshape((2, h)).shape)
                    t_k__predict = t_k_predict.reshape((2, h))
                    predictions_x, predictions_y, *z = t_k__predict
                    args.extend([predictions_x.detach().numpy(), predictions_y.detach().numpy(), '*'])
                    """for xx, yy in zip(x, y):
                        predictions_x.append(xx)
                        predictions_y.append(yy)
                    args.extend((predictions_x, predictions_y, '+'))"""

                for x, y in t_lanes:
                    args.extend([x, y, '-g'])

                open('/dev/shm/%d.pickle' % index, 'wb').write(pickle.dumps(args))
                index += 1

        batch_error_dict = metrics(predict.detach().numpy(), v.detach().numpy())
        eval_ade_batch_errors = np.hstack((eval_ade, batch_error_dict['ade']))
        eval_fde_batch_errors = np.hstack((eval_fde, batch_error_dict['fde']))
        print("ADE", np.min(batch_error_dict['ade']))  # sum(map(ade_t, range(B))) / B)
        print("FDE", np.min(batch_error_dict['fde']))  # sum(map(fde_t, range(B))) / B)

        print("Batch " + str(i) + " of " + str(len(ts_dl)))

results5 = compute_metrics(preds5, helper, pred_config5)
print('Results for K=5: \n' + str(results5))

results10 = compute_metrics(preds10, helper, pred_config10)
print('Results for K=10: \n' + str(results10))