from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from nuscenes import NuScenes
from pyquaternion import Quaternion
from PIL import Image, ImageOps
from torchvision import transforms
from typing import Dict, Tuple, Any, List, Callable, Union
import pandas as pd
import json
from help import PredHelper
# This is the path where you stored your copy of the nuScenes dataset.
DATAROOT = '/mnt/data/datasets/nuscenes'
DATAROOT = '/home/aghoul/nuscenes-devkit/python-sdk/nuscenes'
import math

nuscenes = NuScenes('v1.0-trainval', dataroot=DATAROOT)
from nuscenes.prediction import PredictHelper

helper = PredictHelper(nuscenes)
help = PredHelper(nuscenes)
class NuscenesDataset(Dataset):
    def __init__(self, im_w, t_h=4, t_f=12, max_occ=17, step='train'):
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.step = step
        self.indx = self.generate_sample_indx(step)
        self.inds = pd.read_csv('index.csv')
        self.inds = pd.DataFrame(self.inds.iloc[:, 2]).set_index([self.inds.iloc[:, 0], self.inds.iloc[:, 1]])

        self.nb_samples = len(self.indx.index)
        self.d = {}
        self.im_w = im_w
        self.max_occ = max_occ
        for arg in range(len(nuscenes.category)):
            self.d[nuscenes.category[arg]['name']] = arg
        self.max_ann = self.max_ann_nb()
        print('d : ', self.d)

    def __len__(self):
        return self.nb_samples

    def __getitem__(self, idx):
        my_sample = nuscenes.get('sample', self.indx.iloc[idx, 0])
        target_hist = np.zeros((self.t_h, 5))
        t_hist = helper.get_past_for_agent(self.indx.iloc[idx, 1], self.indx.iloc[idx, 0], seconds=self.t_h // 2,
                                           in_agent_frame=True, just_xy=True)
        t_hist = np.concatenate((t_hist, help.get_past_vel_for_agent(self.indx.iloc[idx, 1], self.indx.iloc[idx, 0],
                                                                       seconds=self.t_h // 2, in_agent_frame=False,
                                                                       just_xy=True)), axis=1)
        target_hist[:min(self.t_h, t_hist.shape[0]), :] = t_hist[:min(self.t_h, t_hist.shape[0]), :]
        fut = np.zeros((self.t_f, 2))
        f = helper.get_future_for_agent(self.indx.iloc[idx, 1], self.indx.iloc[idx, 0], seconds=6, in_agent_frame=True,
                                        just_xy=True)
        fut[:f.shape[0], :] = f
        hist = np.zeros((self.max_ann, self.t_h, 5))
        annotation_record = helper.get_sample_annotation(self.indx.iloc[idx, 1], self.indx.iloc[idx, 0])
        neighbors = helper.get_past_for_sample(self.indx.iloc[idx, 0], seconds=self.t_h // 2, in_agent_frame=False,
                                               just_xy=True)

        map_pos = self.indx.iloc[idx, 2:]
        map_pos = map_pos.map(json.loads)
        mask = np.concatenate((np.ones(f.shape[0]), np.zeros((self.t_f - f.shape[0]))))

        l = map_pos.sum()
        map_pos = map_pos.map(
            lambda x: np.c_[np.ones((1, len(x))), np.zeros((1, self.max_occ - len(x)))])
        final_map = np.concatenate((map_pos[:]), axis=1).reshape(-1)
        t_map = self.get_map(self.indx.iloc[idx, 0], self.indx.iloc[idx, 1])
        for k in range(0, len(l)):
            _, ann_data = return_instance(my_sample, l[k] - 1)
            if (neighbors[ann_data['instance_token']].size > 0):
                hist[k, :min(neighbors[ann_data['instance_token']].shape[0], self.t_h),
                :2] = convert_global_coords_to_local(neighbors[ann_data['instance_token']][
                                                     :min(neighbors[ann_data['instance_token']].shape[0], self.t_h), :],
                                                     annotation_record['translation'],
                                                     annotation_record['rotation'])
                ext_f = help.get_past_vel_for_agent(ann_data['instance_token'], ann_data['sample_token'],
                                                      seconds=self.t_h // 2,
                                                      in_agent_frame=False, just_xy=True)

                hist[k, :min(self.t_h, ext_f.shape[0]), 2:] = ext_f[:min(self.t_h, ext_f.shape[0]), :]

        return target_hist, hist, fut, mask, np.concatenate(
            (np.ones(len(l)), np.zeros((self.max_ann - len(l))))), t_map, final_map
        '''
        # use for test
        return self.indx.iloc[idx, 1], self.indx.iloc[idx, 0], annotation_record['translation'], annotation_record[
            'rotation'], target_hist, hist, fut, mask, np.concatenate(
            (np.ones(len(l)), np.zeros((self.max_ann - len(l))))), t_map, final_map
        '''

    def update_index(self):
        self.inds = pd.read_csv('index.csv')
        self.inds = pd.DataFrame(self.inds.iloc[:, 2]).set_index([self.inds.iloc[:, 0], self.inds.iloc[:, 1]])

    def update_epoch(self, epoch):
        self.epoch = epoch

    def generate_sample_indx(self, step):
        if step == 'train':
            data = pd.read_csv('train_data_m.csv')
        elif step == 'validation':
            data = pd.read_csv('val_data_m.csv')
        elif step == 'selected':
            data = pd.read_csv('test.csv')
        return data

    def get_map(self, k, ann_token):
        im = Image.open('maps/map_' + k + '_' + ann_token + '.png')
        # im = self.crop_center(im.rotate(a), 250, 250)
        input_image = ImageOps.flip(im.convert('RGB'))
        # input_image = im.convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        return input_tensor

    def max_ann_nb(self):
        nb_max = 0
        nb_samples = len(nuscenes.sample)
        for s in range(nb_samples):
            my_sample = nuscenes.sample[s]
            nb_ann = len(my_sample['anns'])
            if nb_max < nb_ann:
                nb_max = nb_ann
        return nb_max


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def angle_of_rotation(yaw: float) -> float:
    """
    Given a yaw angle (measured from x axis), find the angle needed to rotate by so that
    the yaw is aligned with the y axis (pi / 2).
    :param yaw: Radians. Output of quaternion_yaw function.
    :return: Angle in radians.
    """
    return (np.pi / 2) + np.sign(-yaw) * np.abs(yaw)


def make_2d_rotation_matrix(angle_in_radians: float) -> np.ndarray:
    """
    Makes rotation matrix to rotate point in x-y plane counterclockwise
    by angle_in_radians.
    """

    return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                     [np.sin(angle_in_radians), np.cos(angle_in_radians)]])


def convert_global_coords_to_local(coordinates: np.ndarray,
                                   translation: Tuple[float, float, float],
                                   rotation: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Converts global coordinates to coordinates in the frame given by the rotation quaternion and
    centered at the translation vector. The rotation is meant to be a z-axis rotation.
    :param coordinates: x,y locations. array of shape [n_steps, 2].
    :param translation: Tuple of (x, y, z) location that is the center of the new frame.
    :param rotation: Tuple representation of quaternion of new frame.
        Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
    :return: x,y locations in frame stored in array of share [n_times, 2].
    """
    yaw = angle_of_rotation(quaternion_yaw(Quaternion(rotation)))

    transform = make_2d_rotation_matrix(angle_in_radians=yaw)

    coords = (coordinates - np.atleast_2d(np.array(translation)[:2])).T

    return np.dot(transform, coords).T[:, :2]


def convert_local_coords_to_global(coordinates: np.ndarray,
                                   translation: Tuple[float, float, float],
                                   rotation: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Converts local coordinates to global coordinates.
    :param coordinates: x,y locations. array of shape [n_steps, 2]
    :param translation: Tuple of (x, y, z) location that is the center of the new frame
    :param rotation: Tuple representation of quaternion of new frame.
        Representation - cos(theta / 2) + (xi + yi + zi)sin(theta / 2).
    :return: x,y locations stored in array of share [n_times, 2].
    """
    yaw = angle_of_rotation(quaternion_yaw(Quaternion(rotation)))

    transform = make_2d_rotation_matrix(angle_in_radians=-yaw)

    return np.dot(transform, coordinates.T).T[:, :2] + np.atleast_2d(np.array(translation)[:2])


def return_instance(my_sample, k):
    my_annotation_token = my_sample['anns'][k]
    my_annotation_metadata = nuscenes.get('sample_annotation', my_annotation_token)
    my_instance = nuscenes.get('instance', my_annotation_metadata['instance_token'])
    return my_instance, my_annotation_metadata


## Multipath loss function (Chai et al, CoRL 2019)
# Inputs:
# y_traj: Predicted values of muX, muY, sigX, sigY and rho. Shape: [sequence_length, num_anchors, batch_size, 5]
# y_prob: Probabilities assigned by mode to each anchor. Shape: [batch_size, num_anchors]
# y_gt: Ground truth trajectory. Shape: [sequence_length, batch_size, 2]
# a: anchors. Shape: [sequence_length, num_anchors, 2]
# mask: mask of 0/1 values depending on length of ground_truth sequence. Shape: [sequence_length, batch_size]
def maskedNLL(y_traj, y_prob, y_gt, n_h, mask):
    sequence_length = y_traj.shape[0]
    batch_size = y_prob.shape[0]
    y_traj = y_traj.contiguous().view(sequence_length, n_h, batch_size, -1)
    y_gt_rpt = y_gt.unsqueeze(1).repeat(1, n_h, 1, 1)

    # Calculate NLL for trajectories corresponding to selected anchors:
    muX = (y_traj[:, :, :, 0])
    muY = (y_traj[:, :, :, 1])
    x = y_gt_rpt[:, :, :, 0]
    y = y_gt_rpt[:, :, :, 1]
    out = (torch.pow(x - muX, 2) + torch.pow(y - muY, 2))
    out = out.permute(0, 2, 1)
    m_err, inds = torch.min(out.sum(0), -1)
    _, cls = torch.max(y_prob,-1)
    l0 = m_err / torch.max(torch.sum(mask, dim=0), torch.ones(batch_size).cuda())
    l1 = - torch.log(torch.squeeze(y_prob.gather(1, inds.unsqueeze(1))))
    l = l0 + l1
    l = torch.mean(l, 0)

    return l, torch.mean(l0, 0), torch.mean(l1, 0)

def outputActivation(x):
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out

