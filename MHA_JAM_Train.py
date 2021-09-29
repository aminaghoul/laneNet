import sys

#sys.path.insert(0, '/home/aghoul/nuscenes-devkit/python-sdk')
sys.path.insert(0, '/mnt/data/datasets/nuscenes')

import torch
from MHA_JAM_dl import NuscenesDataset, maskedNLL
from torch.utils.data import DataLoader
from MHA_JAM_Model import highwayNet

import random
import time
import math

import logging
import utils_nn as utils
import argparse
import os



parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default='baselineX')
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in experiments/experiment containing weights to reload before \
                    training")  # 'best' or 'train'

cmd_args = parser.parse_args()

args = {}
args['train_flag'] = True
args['model_dir'] = 'experiments/' + cmd_args.experiment
args['restore_file'] = cmd_args.restore_file  # or 'last' or 'best'

utils.set_logger(os.path.join(args['model_dir'], 'train.log'))

json_path = os.path.join(args['model_dir'], 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)

# use GPU if available
params.use_cuda = torch.cuda.is_available()
params.train_flag = args['train_flag']
params.model_dir = args['model_dir']

print("\nEXPERIMENT:", args['model_dir'], "\n")

batch_size = params.batch_size

## Initialize data loaders
logging.info("Loading the datasets...")

trSet = NuscenesDataset(im_w=params.im_w, step='train')
valSet = NuscenesDataset(im_w=params.im_w, step='validation')

trDataloader = DataLoader(trSet, batch_size=batch_size, shuffle=True, num_workers=8)

valDataloader = DataLoader(valSet, batch_size=batch_size, shuffle=True, num_workers=8)

# Initialize network
net = highwayNet(params)
net = net.float()
if params.use_cuda:
    net = net.cuda()

# Set the random seed for reproducible experiments
random.seed(100)
torch.manual_seed(100)
if params.use_cuda: torch.cuda.manual_seed(100)

# This corrects for the differences in dropout, batch normalization during training and testing.
# No dropout, batch norm so far; but it is a good default practice anyways
net.train()

## Initialize optimizer
pretrainEpochs = 0
trainEpochs = 50

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

## Variables holding train and validation loss values:
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

    for i, data in enumerate(trDataloader):
        st_time = time.time()
        t_hist, hist, fut, mask, s_mask, t_map , final_map = data
        fut = fut.permute(1, 0, 2)
        mask = mask.permute(1, 0).bool()
        s_mask = s_mask.bool()
        final_map = final_map.bool()

        if params.use_cuda:
            t_hist = t_hist.float().cuda()
            hist = hist.float().cuda()
            fut = fut.float().cuda()
            mask = mask.float().cuda()
            s_mask = (s_mask > 0.5).cuda()
            t_map = t_map.float().cuda()
            final_map = (final_map > 0.5).cuda()

        fut_pred, class_proba = net(t_hist.float(), hist.float(), s_mask, t_map.float(), final_map)
        b_size = class_proba.shape[0]
        fut_pred = fut_pred.contiguous().view(params.out_length, params.n_head, b_size, -1)
        l, _, _ = maskedNLL(fut_pred, class_proba, fut, params.n_head, mask)

        # Backprop and update weights
        optimizer.zero_grad()
        l.backward()
        a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        # Track average train loss and average train time:
        batch_time = time.time() - st_time
        avg_tr_loss += l.item()
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
        for i, data in enumerate(valDataloader):
            st_time = time.time()

            t_hist, hist, fut, mask, s_mask, t_map, final_map = data
            fut = fut.permute(1, 0, 2)
            mask = mask.permute(1, 0)

            if params.use_cuda:
                t_hist = t_hist.float().cuda()
                hist = hist.float().cuda()
                fut = fut.float().cuda()
                mask = mask.float().cuda()
                s_mask = (s_mask > 0.5).cuda()
                t_map = t_map.float().cuda()
                final_map = (final_map > 0.5).cuda()

            fut_pred, class_proba = net(t_hist, hist, s_mask, t_map, final_map)
            b_size = class_proba.shape[0]
            fut_pred = fut_pred.contiguous().view(params.out_length, params.n_head, b_size, -1)
            l, l0, l1 = maskedNLL(fut_pred, class_proba, fut, params.n_head, mask)

            avg_val_loss += l.item()
            avg_rg_loss += l0.item()
            avg_cl_loss += l1.item()
            val_batch_count += 1

    # Print validation loss and update display variables
    logging.info("Validation loss : {:0.4f} | Regression loss : {:0.4f} | Classification loss : {:0.4f}".format(
        avg_val_loss / val_batch_count, avg_rg_loss / val_batch_count, avg_cl_loss / val_batch_count))
    val_loss.append(avg_val_loss / val_batch_count)

    if (min_val_loss > avg_val_loss / val_batch_count ) :
        logging.info("- Found new best val_loss")
        if not os.path.exists('./trained_models'):
            print("trained_models Directory does not exist! Making directory {}".format('./trained_models'))
            os.mkdir('./trained_models')
        else:
            print("./trained_models Directory exists! ")
        torch.save(net.state_dict(),
                   'trained_models/MHA_JAM_' + str(params.n_head) + '_epoch_' + str(epoch_num + 1) + '.tar')
        min_val_loss = avg_val_loss / val_batch_count

    prev_val_loss = avg_val_loss / val_batch_count


