# from model import ConvLSTM
from itertools import chain

import torch
from torch.utils.data import DataLoader, Dataset
import time
import math
import yaml
from torch.utils.tensorboard import SummaryWriter

from model import LaneNet
from main import NS

config_file = 'config.yml'
# Read config file
with open(config_file, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

# Initialize datasets and dataloaders:
batch_size = 32
tr_set = NS.every_map('config.yml')

#val_set = NS(config['dataroot'], config['val'])
#ts_set = NS(config['dataroot'],config['test'])

tr_dl = DataLoader(tr_set, batch_size=batch_size, shuffle=False, num_workers=8)
#val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
#ts_dl = DataLoader(ts_set, batch_size=batch_size, shuffle=False, num_workers=8)

class SuperNS(Dataset):
    def __init__(self):
        self.__ns = list(NS.every_map('config.yml'))
        self.__items = list(chain.from_iterable(self.__ns))

    def __len__(self):
        return len(self.__items)

    def __getitem__(self, item):
        return self.__items[item]


if __name__ == '__main__':
    tr_set = SuperNS()
    tr_dl = DataLoader(
        tr_set,
        batch_size=32,
        shuffle=True,
        num_workers=8)

    print(len(next(iter(tr_dl))))


    # Initialize Models:
    net = LaneNet('config.yml').float().to(
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    exit()
    # Initialize Optimizer:
    num_epochs = 25
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    # Load checkpoint if specified in config:
    start_epoch = 1
    val_loss = math.inf
    min_val_loss = math.inf

    # ======================================================================================================================
    # Main Loop
    # ======================================================================================================================

    # Forever increasing counter to keep track of iterations (for tensorboard log).
    iters_epoch = len(tr_set) // 32
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

        # For tracking training time
        st_time = time.time()

        # Load batch
        for i, data in enumerate(tr_dl):

            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

            parameters = hist, lanes, voisins

            # Forward pass
            if config['args_r']['use_maneuvers']:
                fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                # Pre-train with MSE loss to speed up training
                if epoch_num < pretrainEpochs:
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    # Train with NLL loss
                    l = maskedNLL(fut_pred, fut, op_mask) + crossEnt(lat_pred, lat_enc) + crossEnt(lon_pred, lon_enc)
                    avg_lat_acc += (torch.sum(torch.max(lat_pred.data, 1)[1] == torch.max(lat_enc.data, 1)[1])).item() / \
                                   lat_enc.size()[0]
                    avg_lon_acc += (torch.sum(torch.max(lon_pred.data, 1)[1] == torch.max(lon_enc.data, 1)[1])).item() / \
                                   lon_enc.size()[0]
            else:
                fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                if epoch_num < pretrainEpochs:
                    l = maskedMSE(fut_pred, fut, op_mask)
                else:
                    l = maskedNLL(fut_pred, fut, op_mask)
            # Process inputs
            _, _, img, svf_e, motion_feats, _, _, _, _, img_vis, _, _, _ = data
            svf_e = svf_e.float().to(device)
            img = img.float().to(device)
            motion_feats = motion_feats.float().to(device)

            # Calculate reward over grid using model
            r, _ = net(motion_feats, img)

            # Forward RL (solve for maxent policy and SVF)
            r_detached = r.detach()
            svf, _ = rl.solve(mdp, r_detached, initial_state=initial_state)

            # Calculate difference in state visitation frequencies
            svf = svf.to(device)
            svf_diff = svf - svf_e

            # Backprop
            optimizer.zero_grad()
            torch.autograd.backward(r, svf_diff)
            a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            optimizer.step()

            # Track difference in state visitation frequencies and train time
            batch_time = time.time() - st_time
            tr_svf_diff_path += torch.mean(torch.abs(svf_diff[:, 0, :, :])).item()
            tr_svf_diff_goal += torch.mean(torch.abs(svf_diff[:, 1, :, :])).item()
            tr_time += batch_time
            st_time = time.time()

            # Tensorboard train metrics
            writer.add_scalar('train/SVF diff (goals)', torch.mean(torch.abs(svf_diff[:, 1, :, :])).item(), iters)
            writer.add_scalar('train/SVF diff (paths)', torch.mean(torch.abs(svf_diff[:, 0, :, :])).item(), iters)

            # Increment global iteration counter for tensorboard
            iters += 1

            # Print/log train loss (path SVFs) and ETA for epoch after pre-defined steps
            iters_log = config['opt_r']['steps_to_log_train_loss']
            if i % iters_log == iters_log - 1:
                eta = tr_time / iters_log * (len(tr_set) / config['opt_r']['batch_size'] - i)
                print("Epoch no:", epoch,
                      "| Epoch progress(%):", format(i / (len(tr_set) / config['opt_r']['batch_size']) * 100, '0.2f'),
                      "| Train SVF diff (paths):", format(tr_svf_diff_path / iters_log, '0.5f'),
                      "| Train SVF diff (goals):", format(tr_svf_diff_goal / iters_log, '0.7f'),
                      "| Val loss prev epoch", format(val_loss, '0.7f'),
                      "| Min val loss", format(min_val_loss, '0.5f'),
                      "| ETA(s):", int(eta))

                # Log images from train batch into tensorboard:
                tb_fig_train = u.tb_reward_plots(img_vis[0:8],
                                                 r[0:8].detach().cpu(),
                                                 svf[0:8].detach().cpu(),
                                                 svf_e[0:8].detach().cpu())
                writer.add_figure('train/SVFs_and_rewards', tb_fig_train, iters)

                # Reset variables to track training performance
                tr_svf_diff_path = 0
                tr_svf_diff_goal = 0
                tr_time = 0

        # __________________________________________________________________________________________________________________
        # Validate
        # __________________________________________________________________________________________________________________
        print('Calculating validation loss...')

        # Set batchnorm layers to eval mode, stop tracking gradients
        net.eval()
        with torch.no_grad():

            # Variables to track validation performance
            val_svf_diff_path = 0
            val_svf_diff_goal = 0
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

    # Close tensorboard writer
    writer.close()
