# from model import ConvLSTM
import json
from inspect import stack
from itertools import chain
from signal import signal, SIGALRM, alarm

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import time
import math
import yaml
from torch.utils.tensorboard import SummaryWriter
from nuscenes.eval.prediction.compute_metrics import compute_metrics
from nuscenes.eval.prediction.config import PredictionConfig
from model import LaneNet
from main import NS

config_file = 'config.yml'
# Read config file
with open(config_file, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)['ns_args']


class SuperNS(Dataset):
    def __init__(self):
        self.__ns = list(NS.every_map('config.yml'))
        self.__items = list(chain.from_iterable(self.__ns))
        self.helper = self.__ns[0].helper

    def __len__(self):
        return len(self.__items)

    def __getitem__(self, item):
        return self.__items[item]


def handler(*args):
    print('Current state of the stack:')
    for args in enumerate(stack()):
        print(*args, sep=' -> ')
    print('...........................')
    alarm(10)


signal(SIGALRM, handler)
alarm(1)

# Tensorboard summary writer:
writer = SummaryWriter(log_dir=config['log_dir'])

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Initialize datasets and dataloaders:
batch_size = config['batch_size']
tr_set = SuperNS()

# Prediction helper and configs:
helper = tr_set.helper
with open('prediction_cfg.json', 'r') as f:
    pred_config = json.load(f)
pred_config5 = PredictionConfig.deserialize(pred_config, helper)

# val_set = NS(config['dataroot'], config['val'])
# ts_set = NS(config['dataroot'],config['test'])

tr_dl = DataLoader(tr_set, batch_size=batch_size, shuffle=False, num_workers=8)

# val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
# ts_dl = DataLoader(ts_set, batch_size=batch_size, shuffle=False, num_workers=8)

if __name__ == '__main__':

    # tr_set = SuperNS()
    # tr_dl = DataLoader(tr_set, batch_size=batch_size, shuffle=False, num_workers=8)

    # Useful functions

    def dist(v, lane):
        xa, ya = v
        xb, yb = lane
        xc, yc = xa - xb, ya - yb
        return np.sqrt(xc * xc + yc * yc)

    def l(V_hat, V, L_ref):
        if dist(V_hat, L_ref) > dist(V, L_ref):
            return dist(V_hat, L_ref)
        else:
            return 0

    # Parameters
    K = config['number_of_predictions']
    h = config['prediction_duration']
    # Initialize Models:
    net = LaneNet('config.yml').float().to(device)

    # Initialize Optimizer:
    num_epochs = config['num_epochs']
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])

    # Loss
    L1Loss = torch.nn.L1Loss()
    L1Loss = L1Loss.to(device)

    # Load checkpoint if specified in config:
    start_epoch = 1
    val_loss = math.inf
    min_val_loss = math.inf

    # ======================================================================================================================
    # Main Loop
    # ======================================================================================================================

    # Forever increasing counter to keep track of iterations (for tensorboard log).
    iters_epoch = len(tr_set) // batch_size
    iters = (start_epoch - 1) * iters_epoch

    for epoch in range(start_epoch, start_epoch + num_epochs):

        # __________________________________________________________________________________________________________________
        # Train
        # __________________________________________________________________________________________________________________

        # Set batchnorm layers to train mode
        net.train()

        # Variables to track training performance
        tr_svf_diff_path = 0
        tr_svf_diff_goal = 0
        tr_time = 0
        pred = []
        # For tracking training time
        st_time = time.time()

        # Load batch
        for i, data in enumerate(tr_dl):

            # Process inputs
            history, future, lanes, neighbors, lane_reference = data

            history = history.float().to(device)
            future = future.float().to(device)
            lanes = lanes.float().to(device)
            neighbors = neighbors.float().to(device)

            history = torch.permute(history, (0, 2, 1))
            future = torch.permute(future, (0, 2, 1))
            lanes = torch.permute(lanes, (1, 0, 3, 2))
            neighbors = torch.permute(neighbors, (1, 0, 3, 2))
            print("history.shape :", history.shape)
            print("future.shape :", future.shape)
            print("lanes.shape :", lanes.shape)
            print("neighbors.shape :", neighbors.shape)

            # Forward pass
            fut_pred = net(history, lanes, neighbors).permute(1, 0, 2)
            print("fut_pred.shape :", fut_pred.shape)
            # Loss



            # Reshape future
            future = torch.flatten(future, start_dim=1)
            lpred = 0
            for (b_pred, b) in zip(fut_pred, future):
                print("bpred shape : ", b_pred.shape)
                print("b shape : ", b.shape)
                for pred_k in b_pred:
                    Lpos = L1Loss(pred_k, b)
                    Llane_off = 0
                    print("pred_k shape : ", pred_k.shape)
                    for index in range(h):
                        print(pred_k[index])
                        print(b[index])
                        exit()
                        Llane_off += l(pred_k[i], b[i],lane_ref)





            print("future.shape :", future.shape)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            # torch.autograd.backward(r, svf_diff)
            # a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            optimizer.step()

            # Track train time
            batch_time = time.time() - st_time
            tr_time += batch_time
            st_time = time.time()

            # Tensorboard train metrics
            # writer.add_scalar('train/SVF diff (goals)', torch.mean(torch.abs(svf_diff[:, 1, :, :])).item(), iters)
            # writer.add_scalar('train/SVF diff (paths)', torch.mean(torch.abs(svf_diff[:, 0, :, :])).item(), iters)

            # Increment global iteration counter for tensorboard
            iters += 1

            # Print/log train loss  for epoch after pre-defined steps
            iters_log = config['steps_to_log_train_loss']
            if i % iters_log == iters_log - 1:
                eta = tr_time / iters_log * (len(tr_set) / config['batch_size'] - i)
                print("Epoch no:", epoch,
                      "| Epoch progress(%):", format(i / (len(tr_set) / config['batch_size']) * 100, '0.2f'),
                      "| Train loss:", format(loss, '0.5f'),
                      "| Val loss prev epoch", format(val_loss, '0.7f'),
                      "| Min val loss", format(min_val_loss, '0.5f'))

                # Log images from train batch into tensorboard:
                # writer.add_figure('train/SVFs_and_rewards', tb_fig_train, iters)

                # Reset variables to track training performance
                tr_time = 0
        results5 = compute_metrics(pred, helper, pred_config5)
        print('Results for K=5: \n' + str(results5))
        # __________________________________________________________________________________________________________________
        # Validate
        # __________________________________________________________________________________________________________________
        print('Calculating validation loss...')

        # Set batchnorm layers to eval mode, stop tracking gradients
        net.eval()
        with torch.no_grad():

            # Variables to track validation performance
            val_batch_count = 0

            # Load batch
            for k, data_val in enumerate(val_dl):

                # Process inputs
                _, _, img, svf_e, motion_feats, _, _, _, _, img_vis, _, _, _ = data_val
                svf_e = svf_e.float().to(device)
                img = img.float().to(device)
                motion_feats = motion_feats.float().to(device)

                # Calculate reward over grid using model
                r, _ = net(motion_feats, img)

                # Forward RL (solve for maxent policy and SVF)
                r_detached = r.detach()
                svf, pi = rl.solve(mdp, r_detached, initial_state=initial_state)

                # Calculate difference in state visitation frequencies
                svf = svf.to(device)
                svf_diff = svf - svf_e
                val_svf_diff_path += torch.mean(torch.abs(svf_diff[:, 0, :, :])).item()
                val_svf_diff_goal += torch.mean(torch.abs(svf_diff[:, 1, :, :])).item()
                val_batch_count += 1

                # Log images from first val batch into tensorboard
                if k == 0:
                    tb_fig_val = u.tb_reward_plots(img_vis[0:8],
                                                   r[0:8].detach().cpu(),
                                                   svf[0:8].detach().cpu(),
                                                   svf_e[0:8].detach().cpu())
                    writer.add_figure('val/SVFs_and_rewards', tb_fig_val, iters)

        # Print validation losses
        print('Val SVF diff (paths) :', format(val_svf_diff_path / val_batch_count, '0.5f'),
              ', Val SVF diff (goals) :', format(val_svf_diff_goal / val_batch_count, '0.7f'))
        val_loss = val_svf_diff_path / val_batch_count

        # Tensorboard val metrics
        writer.add_scalar('val/SVF_diff_goals', val_svf_diff_goal / val_batch_count, iters)
        writer.add_scalar('val/SVF_diff_paths', val_svf_diff_path / val_batch_count, iters)
        writer.flush()

        # Save checkpoint
        if config['opt_r']['save_checkpoints']:
            model_path = config['opt_r']['checkpt_dir'] + '/' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'min_val_loss': min(val_loss, min_val_loss)
            }, model_path)

        # Save best model if applicable
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            model_path = config['opt_r']['checkpt_dir'] + '/best.tar'
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'min_val_loss': min_val_loss
            }, model_path)
    results5 = compute_metrics(preds5, helper, pred_config5)
    print('Results for K=5: \n' + str(results5))
    # Close tensorboard writer
    writer.close()
