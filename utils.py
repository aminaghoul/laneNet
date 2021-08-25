import math
import numpy as np
import yaml
from tqdm import tqdm
import json
import logging
import os
import shutil

import torch

# Get arguments
config_file = 'config.yml'
with open(config_file, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

batch_size = config['ln_args']['batch_size']


# Loss function
def dist(point, lane):
    # the distance from the point `point` to the lane `lane`
    # TODO: Is it a good idea?
    x, y, *z = point  # .detach().numpy()
    # The first step is to get the closest coordinates in the
    #  lane
    # TODO: See if we can finer than that
    # TODO: See if we can recover the fingerprint of the lane
    #  and use the helper instead.
    closest_x = -np.inf
    closest_y = -np.inf
    closest_distance = np.inf
    for x_lane, y_lane, *z in lane:  # .detach().numpy():
        distance = math.sqrt(
            math.pow(x - x_lane, 2) +
            math.pow(y - y_lane, 2))
        if distance < closest_distance:
            closest_distance = distance
            closest_x = x_lane
            closest_y = y_lane
    xc, yc = x - closest_x, y - closest_y
    return math.sqrt(xc * xc + yc * yc)


def threshold_distance(v_hat_t_k_i, v_t_i, l_ref_t):
    if dist(v_hat_t_k_i, l_ref_t) > dist(v_t_i, l_ref_t):
        return dist(v_hat_t_k_i, l_ref_t)
    else:
        return 0


def get_loss(v_hat, reference_indices, alpha, beta, v, h, all_lanes, device, cel, l1_loss, out_la):
    # TODO: Make sure we don't detach the wrong thingsAdding h
    # Internal functions

    all_lanes = all_lanes.permute(1, 0, 3, 2)

    reference_lane = []
    for index, reference_index in enumerate(reference_indices):
        reference_lane.append(all_lanes[index][int(reference_index)])
    reference_lane = torch.stack(reference_lane)

    # Note: the variables v and v_hat depends on the index t,
    #  therefore in the following they will be referred to as
    #  v[t] and v_hat[t], and will contain a list of M coordinates.
    # Any prediction should be okay
    K: int = v_hat[0].shape[0]
    # Any variable should be okay
    B: int = len(v_hat)
    # The amount of coordinates in a lane
    M: int = all_lanes.shape[2]

    v_hat = v_hat.reshape(B, K, h, 2)  # 2 : nb coordinates
    print('Entering get_loss')

    # lane_ref: B x M x 2
    # v_hat: B x K x h x 2
    # v: B x h x 2
    # all_lanes: B x N x M x 2
    # TODO: See all places where I used reshape instead of permute

    # Defined as the cross-entropy loss for selecting the reference
    #  lane from the lane candidates.
    # TODO: Implement a function to get the lane candidates
    # loss_cls = cel(reference_lane, torch.tensor(np.array((B, K, *reference_lane.shape[1:]))))
    # print(reference_lane.shape, reference_lane.shape)

    # If reference_lane is first
    target = torch.tensor(reference_indices.clone().detach().requires_grad_(True), dtype=torch.long)

    loss_cls = cel(out_la, target)  # .float()

    # If reference_lane is last
    # target = torch.empty(B, 2, dtype=torch.long).random_(5)

    # print("referencelane : ", reference_lane.shape)

    # print("ref_lane_pred : ", ref_lane_pred.shape)

    # #########################
    # Begin loss_cls ##########
    # First step: find the associated lanes for each t, k alongside the
    #  distance from the reference lane
    def false():
        associated_lanes = []
        distances_to_real = []
        loss_cls_b = []
        for t in range(B):
            associated_lanes_t = []
            distances_to_real_t = []
            loss_cls_k = []
            for k in range(K):
                distances = []
                # WARNING: detach
                future_row = torch.permute(v_hat[t][k].reshape((h, 2)), (1, 0)).detach().numpy()
                for l_n in all_lanes[t].detach().numpy():
                    # l_n : M x 2
                    nu = (lambda x: x)
                    distance = 0
                    for i, v_i in enumerate(future_row, 1):
                        _distances = []
                        for l_m in l_n:
                            _distances.append(np.linalg.norm(
                                np.array([v_i[:2], l_m[:2]]),
                            ))
                        distance += min(_distances) * nu(i)
                    distances.append(distance)

                associated_lane = np.argmin(distances)
                # min(range(len(distances)), key=distances.__getitem__)
                associated_lane = all_lanes[:, associated_lane]  # B x M x 2
                loss_cls_k.append(cel(associated_lane, reference_lane))

                """ associated_lanes_t.append(associated_lane)
    
                nu = (lambda x: x)
    
                # we now have to get the distance from the lane to the real one
                distance_to_real = []
                for i, (a, b) in enumerate(zip(reference_lane.numpy(), associated_lane.numpy())):
                    # TODO: Is nu useful
                    distance_to_real.append(nu(i) * np.linalg.norm(np.array((a, b))))
                distances_to_real_t.append(sum(distance_to_real))
            print(len(associated_lanes_t))
            distances_to_real.append(distances_to_real_t)
            associated_lanes.append(associated_lanes_t)
    
        # TODO: Make sure we have the same understanding of optimisation,
        #  which is the minimization of the score
        # TODO: I'm not sure about that, maybe we should have B x N instead of B x K
            target = torch.tensor(np.zeros((B,))).long()
        print("associated_lane :", np.array(distances_to_real).shape)
        print("distances_to_real :", np.array(distances_to_real).shape)
        print('ref_lane : ', reference_lane.shape)"""

            loss_cls_b.append(np.min(loss_cls_k))
        loss_cls = sum(loss_cls_b) / B

    # End loss_cls ############
    # #########################

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

        # v_hat_t_k = torch.permute(v_hat_t_k.reshape((2, h,)), (1, 0))

        # v_t = torch.permute(v_t.reshape((2, h,)), (1, 0))

        # l_ref_t = torch.permute(l_ref_t.reshape((2, M,)), (1, 0))

        # ############################################
        # We return the sum for all the points.
        res = sum(threshold_distance(*i, l_ref_t) for i in zip(v_hat_t_k, v_t)) / h

        return res

    def get_loss_pred_t_k(t, k):
        res = l1_loss(v_hat[t][k], v[t]) * beta + (1 - beta) * get_loss_lane_off(t, k)

        return res

    loss_pred_t = []
    loss_pred_k = []
    for t in tqdm(list(range(B))):
        for k in range(K):
            loss_pred_k.append(get_loss_pred_t_k(t, k))
        loss_pred_t.append(np.min(loss_pred_k))

    loss_pred = sum(
        loss_pred_t) / B  # sum(min(get_loss_pred_t_k(t, k) for k in range(K)) for t in tqdm(list(range(B))))
    loss = alpha * loss_pred + (1 - alpha) * loss_cls

    return loss.type(torch.float32)


# Metrics
def compute_ade(predicted_trajs, gt_traj):
    ade_k = []
    for predicted_traj in predicted_trajs:
        error = np.linalg.norm(predicted_traj - gt_traj)
        ade = np.mean(error)
        ade_k.append(ade)
    return np.min(ade_k).flatten()  # ade.flatten()


def compute_fde(predicted_trajs, gt_traj):
    fde_k = []
    for predicted_traj in predicted_trajs:
        final_error = np.linalg.norm(predicted_traj[-1] - gt_traj[-1])
        fde_k.append(final_error)
    return np.min(fde_k).flatten()  # final_error.flatten()


def metrics(predictions, futures, best_of=True):
    batch_error_dict = {'ade': list(), 'fde': list()}
    for pred, fut in zip(predictions, futures):
        ade_errors = compute_ade(pred, fut)
        fde_errors = compute_fde(pred, fut)

        if best_of:
            ade_errors = np.min(ade_errors, keepdims=True)
            fde_errors = np.min(fde_errors, keepdims=True)
        batch_error_dict['ade'].extend(list(ade_errors))
        batch_error_dict['fde'].extend(list(fde_errors))

    return batch_error_dict


def collate_fn(batch):
    """Append zeroes when the batch is too small"""

    def iterate():
        # If the batch is the correct size, continue
        if len(batch) != batch_size:
            # Otherwise start with determining the filler size and the shapes
            shapes, tail = [row.shape for row in batch[0]], batch_size - len(batch)
            # Then generate as many filler rows as I should to get the correct batch size
            batch.extend([tuple(np.zeros(shape) for shape in shapes) for _ in range(tail)])
        print('===================================')
        for index in range(len(batch[0])):
            print('Index', index)
            # for row in batch:
            #     print(index, row[index].shape)
            """print([row[index].shape for row in batch])
            # We forgot to remove the extra dimension
            if index == 2 and len(batch[0][index].shape) == 4:
                # [6, 20, 1, 2] -> [6, 20, 2]
                for row_index, row in enumerate(batch):
                    row[index] = np.array([[j[0] for j in i] for i in row[index]])"""
            try:
                print('Index', index)
                if len(set(row[index].shape for row in batch)) != 1:
                    mini, = [row[index] for row in batch if len(row[index].shape) < len(batch[0][index].shape)]
                    print(mini, mini.shape, [i.shape for i in mini])
                    raise RuntimeError()
            except ValueError:
                print('Index', index)
                print(batch[0][index])
                print('Index', index)
                print(batch[0][index][0])
                print('Index', index)
                print(batch[0][index][0][0])
                print('Index', index)
                print(batch[0][index][0][0][0])
                print('Index', index)
                raise RuntimeError()
            yield torch.tensor(np.array([row[index] for row in batch]))

    return list(iterate())


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

	In general, it is useful to have a logger so that every output to the terminal is saved
	in a permanent file. Here we save it to `model_dir/train.log`.

	Example:
	```
	logging.info("Starting training...")
	```

	Args:
		log_path: (string) where to log
	"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


# TODO: See where I used to reshape and better reshape
def reshape_v(v, h):
    return torch.permute(
        v.reshape((v.shape[0], 2, h)),
        (0, 2, 1))


def reshape_v_hat(v, h):
    return torch.permute(
        v.reshape((v.shape[0], v.shape[1], 2, h)),
        (0, 1, 3, 2))
