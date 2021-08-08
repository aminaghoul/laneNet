# from model import ConvLSTM
import json
import math
from contextlib import contextmanager
from datetime import timedelta
from inspect import stack
from itertools import chain
from signal import signal, SIGALRM, alarm
from time import time

import torch
import yaml
from nuscenes.eval.prediction.config import PredictionConfig
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from main import NS
from model import LaneNet


@contextmanager
def do_something(what):
    print(what + '...', end='', flush=True)
    begin = time()
    yield
    print('\r' + what + '...done (in %s)' % timedelta(seconds=time() - begin))


class SuperNS(Dataset):
    # TODO: Add more parameters as well as more
    #  attributes to this.
    def __init__(self):
        self.__ns = list(NS.every_map('config.yml'))
        self.__items = list(chain.from_iterable(self.__ns))
        self.helper = self.__ns[0].helper

    def __len__(self):
        # This is the list of every element of
        #  every datasets
        return len(self.__items)

    def __getitem__(self, item):
        # This is the list of every element of
        #  every datasets
        return self.__items[item]


def show_trace(interval, limit=None):
    """
    Warning: uses alarms

    :param interval: How much seconds
    :param limit: Maximum amount of rows
    :return:
    """

    def handler(*args):
        print('Current state of the stack:')
        for args in enumerate(stack()[1:][:limit]):
            print(*args, sep=' -> ')
        print('...........................')
        alarm(interval)

    signal(SIGALRM, handler)
    alarm(interval)


# TODO: Cache the results of the loss function
def get_loss(*args, **kwargs):
    # Placeholder, the real function will come later
    return 0


def main_loop():
    """This will contain everything needed to train the network"""

    # We begin with the dataset, as it will
    #  be used to define other variables
    # This dataset is an aggregator we will
    #  use to get all the scenes from all
    #  the maps.
    with do_something('Initializing the DataSet'):
        tr_set = SuperNS()
    # TODO: Do something like the following, with a
    #  training set, a validation set and a testing
    #  set, for instance having super_ns.val,
    #  super_ns.train and super_ns.test.
    # val_set = NS(config['dataroot'], config['val'])
    # ts_set = NS(config['dataroot'],config['test'])

    # We start by reading the configuration files
    with do_something('Reading configuration files'):
        config_file = 'config.yml'
        prediction_config_file = 'prediction_cfg.json'
        with open(config_file, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)['ns_args']
        with open(prediction_config_file, 'r') as f:
            pred_config = json.load(f)
            # TODO: Typo?

    with do_something('Deserializing configuration'):
        pred_config = PredictionConfig.deserialize(pred_config, tr_set.helper)

    # We then extract parameters from them
    batch_size = config['batch_size']
    _K = config['number_of_predictions']
    h = config['prediction_duration']
    num_epochs = config['num_epochs']
    alpha = config['alpha']
    beta = config['beta']

    # We then define the helpers

    # This will be used to determine whether or not we can
    #  use a cuda enabled GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Tensorboard summary writer
    writer = SummaryWriter(log_dir=config['log_dir'])

    print('Initializing the data loader...', end='', flush=True)
    tr_dl = DataLoader(tr_set, batch_size=batch_size, shuffle=False, num_workers=8)
    print('done')
    # TODO: Do as said near the definition of tr_set
    # val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    # ts_dl = DataLoader(ts_set, batch_size=batch_size, shuffle=False, num_workers=8)

    # Initialize Models:
    print('Initializing models...', end='', flush=True)
    net = LaneNet('config.yml').float().to(device)
    print('done')

    # Initialize Optimizer:
    print('Initializing the optimizer...', end='', flush=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])
    print('done')

    with do_something('Initializing processors'):
        l1_loss = torch.nn.L1Loss().to(device)
        cel = torch.nn.CrossEntropyLoss().to(device)

    # Load checkpoint if specified in config:
    start_epoch = 1
    val_loss = math.inf
    min_val_loss = math.inf

    print('Initialized')

    # ======================================================================================================================
    # Main Loop
    # ======================================================================================================================

    # Forever increasing counter to keep track of iterations (for tensorboard log).
    current_iteration = (start_epoch - 1) * (len(tr_set) // batch_size)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        # __________________________________________________________________________________________________________________
        # Train
        # __________________________________________________________________________________________________________________

        # Set batch norm layers to train mode
        net.train()

        # Variables to track training performances
        # ...

        for i, (history, v, lanes, neighbors, reference_lane) in enumerate(tr_dl):
            # We start by preprocessing the inputs
            history = torch.permute(history.float().to(device), (0, 2, 1))
            lanes = torch.permute(lanes.float().to(device), (1, 0, 3, 2))
            neighbors = torch.permute(neighbors.float().to(device), (1, 0, 3, 2))
            print("history.shape :", history.shape)
            print("lanes.shape :", lanes.shape)
            print("neighbors.shape :", neighbors.shape)

            # The output should be flattened
            # TODO: Why? Can we remove the permute?
            v = torch.flatten(torch.permute(v.float().to(device), (0, 2, 1)), start_dim=1)
            print("v.shape :", v.shape)

            # Then we do a forward pass
            future_predictions = v_hat = net(history, lanes, neighbors).permute(1, 0, 2)
            print("v_hat.shape :", v_hat.shape)
            print("referencelane : ", reference_lane.shape)
            print("lanes.shape :", lanes.shape)
            # ref_lane_pred = get_ref_lane(lanes, v_hat)
            # Then we calculate the losses
            loss_total = get_loss(
                v_hat=v_hat,
                reference_lane=reference_lane,
                alpha=alpha,
                beta=beta,
                v=v, h=h, cel=cel,
                l1_loss=l1_loss
                # ref_lane_pred=ref_lane_pred
            )
            print('Loss', loss_total)

            # TODO: These
            # Copied from the old main
            # optimizer.zero_grad()
            # loss.backward()
            # torch.autograd.backward(r, svf_diff)
            # a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            # optimizer.step()


if __name__ == '__main__':
    main_loop()
