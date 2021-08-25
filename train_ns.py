import pickle
import sys

import yaml

sys.path.insert(0, '/home/aghoul/nuscenes-devkit/python-sdk')

import torch
from preprocess_nuscenes import NuscenesDataset #, maskedNLL
from torch.utils.data import DataLoader
from model import LaneNet

import random
import time
import math

import logging
import utils as utils
import argparse
import os
from utils import *


# We start by reading the configuration files
logging.info('Reading configuration files')

config_file = 'config.yml'
with open(config_file, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

args = {}
args['train_flag'] = True
args['model_dir'] = 'experiments/'
args['restore_file'] = 'best'

utils.set_logger(os.path.join(args['model_dir'], 'train.log'))



# use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config['ln_args']['train_flag'] = args['train_flag']
config['ln_args']['model_dir'] = args['model_dir']
K = config['ln_args']['number_of_predictions']

print("\nEXPERIMENT:", args['model_dir'], "\n")
alpha = config['ln_args']['alpha']
beta = config['ln_args']['beta']

batch_size = config['ln_args']['batch_size']

## Initialize data loaders
logging.info("Loading the datasets...")

trSet = NuscenesDataset(step='mini_train')

valSet = NuscenesDataset(step='validation')

trDataloader = DataLoader(trSet, batch_size=batch_size, shuffle=True, num_workers=8)
#print('samples = ', trSet.nb_samples)
valDataloader = DataLoader(valSet, batch_size=batch_size, shuffle=True, num_workers=8)

# Initialize network
net = LaneNet('config.yml').float().to(device)

# Set the random seed for reproducible experiments
random.seed(100)
torch.manual_seed(100)
if torch.cuda.is_available(): torch.cuda.manual_seed(100)

# This corrects for the differences in dropout, batch normalization during training and testing.
# No dropout, batch norm so far; but it is a good default practice anyways
net.train()

## Initialize optimizer
pretrainEpochs = 0
trainEpochs = 50

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

## Variables holding train and validation loss values:
l1_loss = torch.nn.SmoothL1Loss().to(device)
cel = torch.nn.CrossEntropyLoss().to(device)

train_loss = []
val_loss = []
prev_val_loss = math.inf
min_val_loss = math.inf
best_val_loss = math.inf

for epoch_num in range(pretrainEpochs + trainEpochs):
    logging.info("Epoch {}/{}".format(epoch_num + 1, pretrainEpochs + trainEpochs))

    ## Train:_____________________________________________________________________
    net.train_flag = True
    net.train()  # This is super important for Transformer ... as it uses dropouts...

    # Variables to track training performance:
    avg_tr_loss = 0
    avg_tr_time = 0
    avg_lat_acc = 0
    avg_lon_acc = 0

    for i, (history, future, mask, neighbors, lanes, reference_lane) in enumerate(trDataloader):

        st_time = time.time()

        print('history : ', history.shape)
        print('future : ', future.shape)
        print('neighbors : ', neighbors.shape)
        print('lanes : ', lanes.shape)
        print('reference_lane : ', reference_lane.shape)

        # history : B x (tau+1) x 2
        # lanes : B x N x M x 2
        # neighbors : B x N x (tau+1) x 2
        # v : future : B x h x 2
        # reference_lane : B x 1
        neighbors = torch.squeeze(neighbors, dim=3)
        # We start by preprocessing the inputs
        history = torch.permute(history.float().to(device), (0, 2, 1))
        lanes = torch.permute(lanes.float().to(device), (1, 0, 3, 2))
        neighbors = torch.permute(neighbors.float().to(device), (1, 0, 3, 2))
        # history : B x 2 x tau
        # lanes :  N x B x 2 x M
        # neighbors :  N x B x 2 x tau

        reference_lane = torch.squeeze(reference_lane.float().to(device), dim=1)

        v = torch.flatten(future, start_dim=1)
        v = v.float().to(device)
        # v : future : B x (h x 2)

        # Then we do a forward pass
        predict, out_la = net(history, lanes, neighbors)

        # predict : K x B x (hx2)
        print("predict : ", predict.shape)
        v_hat = predict.permute(1, 0, 2)
        print("v_hat : ", v_hat.shape)
        # v_hat : B x K x (hx2)

        # Then we calculate the losses_

        # loss_total = l1_loss(predict[0], v)
        loss_total = get_loss(
            v_hat=v_hat.type(torch.float32),
            reference_indices=reference_lane.type(torch.float32),
            alpha=alpha,
            beta=beta,
            v=future.type(torch.float32), h=h,
            all_lanes=lanes.type(torch.float32),
            device=device,
            l1_loss=l1_loss,
            cel=cel, out_la=out_la
        )
        print('Loss:', loss_total)
        # loss_total = loss_total.float()
        optimizer.zero_grad()
        loss_total.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()
        all_lanes = lanes.permute(1, 0, 3, 2).type(torch.long)

        reference = []
        for index, reference_index in enumerate(reference_lane.type(torch.long)):
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
                arguments = []

                # t_neighbors : N x nb_cordinates x (tau + 1)

                target_x, target_y = [], []
                for (x, y, _) in t_history.permute(1, 0):
                    target_y.append(y)
                    target_x.append(x)
                arguments.append(((target_x, target_y, 'ego_history'), {'label': 'Target history'}))

                target_x, target_y, *z = t_future.permute(1, 0)
                arguments.append(((target_x, target_y, 'ego_future'), {'label': 'Target future'}))

                l_x, l_y, _ = ref
                arguments.append(((l_x, l_y, 'reference_lane'), {'label': 'Reference lane'}))

                for n_index, neighbor in enumerate(t_neighbors):
                    neighbors_x, neighbors_y = [], []
                    for n_x, n_y, *z in neighbor:
                        neighbors_x.append(n_x)
                        neighbors_y.append(n_y)
                    arguments.append(((neighbors_x, neighbors_x, 'neighbor'), {'label': 'Neighbor %d' % n_index}))

                for p_index, t_k_predict in enumerate(t_predict):
                    # predictions_x, predictions_y = [], []
                    # print(t_k_predict.reshape((2, h)).shape)
                    t_k__predict = t_k_predict.reshape((2, h))
                    predictions_x, predictions_y, *z = t_k__predict
                    arguments.append(
                        ((predictions_x.detach().numpy(), predictions_y.detach().numpy(), 'prediction'),
                         {'label': 'Prediction %d' % p_index}))
                    """for xx, yy in zip(x, y):
                        predictions_x.append(xx)
                        predictions_y.append(yy)
                    args.extend((predictions_x, predictions_y, '+'))"""

                for p_index, (x, y) in enumerate(t_lanes):
                    arguments.append(((x, y, 'lane'), {'label': 'Lane %d' % p_index}))

                open('/dev/shm/%d.pickle' % index, 'wb').write(pickle.dumps(arguments))
                index += 1


        # Track average train loss and average train time:
        batch_time = time.time() - st_time
        avg_tr_loss += loss_total.item()
        avg_tr_time += batch_time

        if i % 100 == 99:
            eta = avg_tr_time / 100 * (len(trSet) / batch_size - i)
            logging.info(
                "Epoch no: {} | Epoch progress(%): {:0.2f} | Avg train loss: {:0.4f} | Validation loss prev epoch {:0.4f} | ETA(s): {}".format(
                    epoch_num + 1, i / (len(trSet) / batch_size) * 100, avg_tr_loss / 100, prev_val_loss, int(eta)))

            train_loss.append(avg_tr_loss / 100)
            avg_tr_loss = 0
            avg_lat_acc = 0
            avg_lon_acc = 0
            avg_tr_time = 0
    ## Validate:______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

    net.train_flag = False
    net.eval()

    logging.info("Epoch {} complete. Calculating validation loss...".format(epoch_num + 1))
    avg_val_loss = 0
    avg_cl_loss = 0
    avg_rg_loss = 0
    val_batch_count = 0
    total_points = 0

    with torch.no_grad():
        for i, (history, future, mask, neighbors, lanes, reference_lane) in enumerate(valDataloader):
            st_time = time.time()

            print('history : ', history.shape)
            print('future : ', future.shape)
            print('neighbors : ', neighbors.shape)
            print('lanes : ', lanes.shape)
            print('reference_lane : ', reference_lane.shape)

            # history : B x (tau+1) x 2
            # lanes : B x N x M x 2
            # neighbors : B x N x (tau+1) x 2
            # v : future : B x h x 2
            # reference_lane : B x 1
            neighbors = torch.squeeze(neighbors, dim=3)
            # We start by preprocessing the inputs
            history = torch.permute(history.float().to(device), (0, 2, 1))
            lanes = torch.permute(lanes.float().to(device), (1, 0, 3, 2))
            neighbors = torch.permute(neighbors.float().to(device), (1, 0, 3, 2))
            # history : B x 2 x tau
            # lanes :  N x B x 2 x M
            # neighbors :  N x B x 2 x tau

            reference_lane = torch.squeeze(reference_lane.float().to(device), dim=1)

            v = torch.flatten(future, start_dim=1)
            v = v.float().to(device)
            # v : future : B x (h x 2)

            # Then we do a forward pass
            predict, out_la = net(history, lanes, neighbors)

            # predict : K x B x (hx2)
            print("predict : ", predict.shape)
            v_hat = predict.permute(1, 0, 2)
            print("v_hat : ", v_hat.shape)
            # v_hat : B x K x (hx2)

            # Then we calculate the losses

            # loss_total = l1_loss(predict[0], v)
            loss_total = get_loss(
                v_hat=v_hat.type(torch.float32),
                reference_indices=reference_lane.type(torch.float32),
                alpha=alpha,
                beta=beta,
                v=future.type(torch.float32), h=h,
                all_lanes=lanes.type(torch.float32),
                device=device,
                l1_loss=l1_loss,
                cel=cel, out_la=out_la
            )
            print('Loss:', loss_total)

            avg_val_loss += loss_total.item()
            val_batch_count += 1

    # Print validation loss and update display variables
    logging.info("Validation loss : {:0.4f}".format(
        avg_val_loss / val_batch_count))
    val_loss.append(avg_val_loss / val_batch_count)

    if (min_val_loss > avg_val_loss / val_batch_count ) :
        logging.info("- Found new best val_loss")
        if not os.path.exists('./trained_models'):
            print("trained_models Directory does not exist! Making directory {}".format('./trained_models'))
            os.mkdir('./trained_models')
        else:
            print("./trained_models Directory exists! ")
        torch.save(net.state_dict(),
                   'trained_models/LaneNet_' + '_epoch_' + str(epoch_num + 1) + '.tar')
        min_val_loss = avg_val_loss / val_batch_count

    prev_val_loss = avg_val_loss / val_batch_count


