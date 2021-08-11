# from model import ConvLSTM

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


@contextmanager
def do_something(what):
    print(what + '...', end='', flush=True)
    begin = time()
    yield
    print('\r' + what + '...done (in %s)' % timedelta(seconds=time() - begin))


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


def main_loop():
    """This will contain everything needed to train the network"""

    # We begin with the dataset, as it will
    #  be used to define other variables
    # This dataset is an aggregator we will
    #  use to get all the scenes from all
    #  the maps.
    with do_something('Initializing the DataSet'):
        tr_set = SuperNS('mini_train')

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
        # with open(prediction_config_file, 'r') as f:
        #     pred_config = json.load(f)
        #     # TODO: Typo?

    # with do_something('Deserializing configuration'):
    #     pred_config = PredictionConfig.deserialize(pred_config, tr_set.helper)

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
    tr_dl = DataLoader(tr_set, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)
    print('done')

    # Initialize Models:
    print('Initializing models...', end='', flush=True)
    net = LaneNet('config.yml').float().to(device)
    l1_loss = torch.nn.L1Loss().to(device)
    cel = torch.nn.CrossEntropyLoss().to(device)
    print('done')

    # Initialize Optimizer:
    print('Initializing the optimizer...', end='', flush=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])
    # if hyperparams['learning_rate_style'] == 'const':
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    """elif hyperparams['learning_rate_style'] == 'exp':
        lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type],
                                                                   gamma=hyperparams['learning_decay_rate'])"""
    print('done')

    with do_something('Initializing processors'):

        try:
            previous_state = torch.load('current_state.bin')
            start_epoch = previous_state['epoch']
            val_loss = previous_state['loss']
            min_val_loss = previous_state['min_val_loss']
            # optimizer.load_state_dict(previous_state['optimizer_state_dict'])
            # net.load_state_dict(previous_state['model_state_dict'])

        except FileNotFoundError:
            start_epoch = 1
            val_loss = math.inf
            min_val_loss = math.inf

    print('Initialized')

    # ======================================================================================================================
    # Main Loop
    # ======================================================================================================================

    # Forever increasing counter to keep track of iterations (for tensorboard log).
    current_iteration = (start_epoch - 1) * (len(tr_set) // batch_size)
    eval_ade = np.array([])
    eval_fde = np.array([])
    loss = []
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
            print('Step ', i, '/', len(tr_dl))

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
            v = torch.flatten(torch.permute(v.float().to(device), (0, 2, 1)), start_dim=1)
            # v : future : B x (h x 2)

            # Then we do a forward pass
            predict, out_la = net(history, lanes, neighbors, reference_lane)
            # predict : K x B x (hx2)

            v_hat = predict.permute(1, 0, 2)
            # v_hat : B x K x (hx2)

            # Then we calculate the losses
            optimizer.zero_grad()
            loss_total = get_loss(
                v_hat=v_hat,
                reference_indices=reference_lane,
                alpha=alpha,
                beta=beta,
                v=v, h=h,
                all_lanes=lanes,
                device=device,
                l1_loss=l1_loss,
                cel=cel, out_la=out_la
            )
            loss.append(loss_total)
            print('Loss:', loss_total)

            loss_total.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 1.0)
            optimizer.step()

            # Stepping forward the learning rate scheduler and annealers.
            lr_scheduler.step()

            index = 0
            try:
                raise KeyboardInterrupt
            except KeyboardInterrupt:
                print(history.shape, neighbors.shape, v.shape, predict.permute(1, 0, 2).shape)
                for t_history, t_neighbors, t_future, t_predict, t_lanes in zip(
                        history, neighbors.permute(1, 0, 2, 3),
                        v, predict.permute(1, 0, 2), lanes.permute(1, 0, 2, 3)):
                    print(t_lanes.shape)
                    target_x, target_y = [], []
                    for x, y, *z in t_history.permute(1, 0):
                        target_y.append(y)
                        target_x.append(x)
                    args = [target_x, target_y, '-']
                    target_x, target_y = t_future.reshape((2, h))
                    args.extend([target_x, target_y, '-'])
                    for t_k_predict in t_predict:
                        predictions_x, predictions_y = [], []
                        for x, y, *z in t_k_predict.reshape((h, 2)):
                            predictions_x.append(y)
                            predictions_y.append(x)
                        args.extend((predictions_x, predictions_y, '-'))

                    for x, y in t_lanes:
                        args.extend([x, y, '-g'])

                    open('/dev/shm/%d.pickle' % index, 'wb').write(pickle.dumps(args))
                    index += 1

            '''
            def e(t, i, k):
                x_a, y_a = reshaped_v[t][i]
                x_b, y_b = reshaped_v_hat[t][k][i]
                return np.sqrt(
                    (x_a - x_b) ** 2 +
                    (y_a - y_b) ** 2)
                    
            def ade_t(t):
                return (min(sum(e(t, i, k) for i in range(h)) for k in range(K))) / h

            def fde_t(t):
                return (min(e(t, h - 1, k) for k in range(K))) / h'''

            batch_error_dict = metrics(predict.detach().numpy(), v.detach().numpy())
            eval_ade_batch_errors = np.hstack((eval_ade, batch_error_dict['ade']))
            eval_fde_batch_errors = np.hstack((eval_fde, batch_error_dict['fde']))
            print("ADE", np.min(batch_error_dict['ade']))  # sum(map(ade_t, range(B))) / B)
            print("FDE", np.min(batch_error_dict['fde']))  # sum(map(fde_t, range(B))) / B)
        torch.save({
            'epoch': epoch,
            # 'model_state_dict': net.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'min_val_loss': min(val_loss, min_val_loss),
            'ade': eval_ade_batch_errors,
            'fde': eval_fde_batch_errors
        }, 'current_state.bin')

    plt.plot(eval_ade, label="ADE")
    plt.plot(eval_fde, label="FDE")
    plt.plot(loss, label="loss")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    show_trace(60, 1)
    main_loop()
