# from model import ConvLSTM

import json
import math
from contextlib import contextmanager
from datetime import timedelta
from inspect import stack
from itertools import chain
from signal import signal, SIGALRM, alarm
from time import time

import numpy as np
import torch
import yaml
from nuscenes.eval.prediction.config import PredictionConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
        # Note: 16 refers to the batch size
        return min(16, len(self.__items))

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

def get_loss(*, v_hat, reference_lane, alpha, beta, v, h, cel: torch.nn.CrossEntropyLoss, l1_loss: torch.nn.L1Loss):
    # TODO: Make sure we don't detach the wrong thingsAdding h
    # Internal functions
    def dist(point, lane):
        """the distance from the point `point` to the lane `lane`"""
        # TODO: Is it a good idea?
        x, y = point.detach().numpy()
        # The first step is to get the closest coordinates in the
        #  lane
        # TODO: See if we can finer than that
        # TODO: See if we can recover the fingerprint of the lane
        #  and use the helper instead.
        closest_x = -np.inf
        closest_y = -np.inf
        closest_distance = np.inf
        for x_lane, y_lane in lane.detach().numpy():
            distance = np.sqrt(
                np.power(x - x_lane, 2) +
                np.power(y - y_lane, 2))
            if distance < closest_distance:
                closest_distance = distance
                closest_x = x_lane
                closest_y = y_lane
        xc, yc = x - closest_x, y - closest_y
        return np.sqrt(xc * xc + yc * yc)

    def threshold_distance(v_hat_t_k_i, v_t_i, l_ref_t):
        if dist(v_hat_t_k_i, l_ref_t) > dist(v_t_i, l_ref_t):
            return dist(v_hat_t_k_i, l_ref_t)
        else:
            return 0

    # Note: the variables v and v_hat depends on the index t,
    #  therefore in the following they will be referred to as
    #  v[t] and v_hat[t], and will contain a list of M coordinates.
    # print('Entering get_loss')
    # print('v_hat:', v_hat.shape)
    # print('v:', v.shape)

    # . is the amount of coordinates in the future

    # lane_ref: B x M x 2
    # v_hat: B x K x . x 2
    # v: B x . x 2

    # Any prediction should be okay
    K: int = v_hat[0].shape[0]
    # Any variable should be okay
    B: int = len(v_hat)
    # The amount of coordinates in a lane
    M: int = reference_lane.shape[1]

    # Defined as a smooth L1 loss between v_hat_f_k and V_f
    def get_loss_pos_t_k(t, k):
        return l1_loss(v_hat[t][k], v[t][k])

    # Defined as the cross-entropy loss for selecting the reference
    #  lane from the lane candidates.
    # TODO: Implement a function to get the lane candidates
    # loss_cls = cel(reference_lane, torch.tensor(np.array((B, K, *reference_lane.shape[1:]))))
    # print(reference_lane.shape, reference_lane.shape)

    # If reference_lane is first
    target = torch.empty(B, 2, dtype=torch.long).random_(5)
    # loss_cls = cel(reference_lane, target)

    # If reference_lane is last
    target = torch.empty(B, 2, dtype=torch.long).random_(5)

    # print("referencelane : ", reference_lane.shape)

    # print("ref_lane_pred : ", ref_lane_pred.shape)
    # loss_cls = cel(reference_lane, ref_lane_pred)

    # t is defined as the index within the batch
    # k is defined as the index within the predictions

    def get_loss_lane_off(t, k):
        # Sum of the distances
        # - for a scene t
        # - for a simulation k
        # - for each point in time (in the future) l+i
        # ############################################
        v_hat_t_k = v_hat[t][k]
        v_t = v[t]
        l_ref_t = reference_lane[t]
        # ###############################
        # print("vhat : ", v_hat_t_k.shape)
        # print("Lref : ", l_ref_t.shape)
        # print("V  : ", v_t.shape)
        # ############################################
        # TODO: Does this preserve the coordinates or
        #  does it mess with everything?
        # We reshape them to have the dimension last
        #  so we can easily extract the coordinates
        v_hat_t_k = v_hat_t_k.reshape((h, 2))
        v_t = v_t.reshape((h, 2))
        l_ref_t = l_ref_t.reshape((M, 2))
        # ############################################
        # We return the sum for all the points.
        return sum(threshold_distance(*i, l_ref_t) for i in zip(v_hat_t_k, v_t)) / h

    def get_loss_pred_t_k(t, k):
        return beta * get_loss_pos_t_k(t, k) + (1 - beta) * get_loss_lane_off(t, k)

    loss_pred = sum(min(get_loss_pred_t_k(t, k) for k in range(K)) for t in tqdm(list(range(B))))
    return alpha * loss_pred  # + (1 - alpha) * loss_cls


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
        print('Starting epoch', epoch)
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
            # print("history.shape :", history.shape)
            # print("lanes.shape :", lanes.shape)
            # print("neighbors.shape :", neighbors.shape)

            # The output should be flattened
            # TODO: Why? Can we remove the permute?
            v = torch.flatten(torch.permute(v.float().to(device), (0, 2, 1)), start_dim=1)
            # print("v.shape :", v.shape)

            # Then we do a forward pass
            v_hat = net(history, lanes, neighbors).permute(1, 0, 2)
            # print("v_hat.shape :", v_hat.shape)
            # print("referencelane : ", reference_lane.shape)
            # print("lanes.shape :", lanes.shape)
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
            print('Loss:', loss_total)

            # TODO: These
            # Copied from the old main
            # optimizer.zero_grad()
            # loss.backward()
            # torch.autograd.backward(r, svf_diff)
            # a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            # optimizer.step()


if __name__ == '__main__':
    main_loop()
