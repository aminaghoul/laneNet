import pickle
from collections import defaultdict

from matplotlib import pyplot as plt
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from nuscenes import NuScenes
from pyquaternion import Quaternion
from PIL import Image, ImageOps
from torchvision import transforms
from typing import Dict, Tuple, Any, List, Callable, Union
import pandas as pd

from help import PredHelper#, get_closest_lanes
import json

# This is the path where you stored your copy of the nuScenes dataset.
# DATAROOT = '/mnt/data/home/kamessao/python-sdk/nuscenes'
DATAROOT = '/home/aghoul/nuscenes-devkit/python-sdk/nuscenes'
import math
import yaml
nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
from nuscenes.prediction import PredictHelper

helper_ns = PredictHelper(nuscenes)
help = PredHelper(nuscenes)
config_file = 'config.yml'
with open(config_file, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
class NuscenesDataset(Dataset):
    def __init__(self, t_h=4, t_f=12, max_occ=17, step='mini_train'):
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted
        self.m = int((config['ns_args']['forward_lane'] + config['ns_args']['backward_lane']) / config['ns_args']['precision_lane'])
        self.FORWARD_DISTANCE, self.BACKWARD_DISTANCE = config['ns_args']['forward_lane'], config['ns_args']['backward_lane']
        self.PRECISION_LANE = config['ns_args']['precision_lane']
        self.length_lane = config['ns_args']['length_lane']
        self.step = step
        #if step == 'mini_train':
        self.nb_samples = 404
        self.n = config['ln_args']['nb_lane_candidates']
        #self.indx = self.generate_sample_indx(step)
        #self.inds = pd.read_csv('index.csv')
        #self.inds = pd.DataFrame(self.inds.iloc[:, 2]).set_index([self.inds.iloc[:, 0], self.inds.iloc[:, 1]])
        #print("self.inds : ", self.inds)
        #self.nb_samples = len(self.indx.index)
        self.d = {}
        #self.im_w = im_w
        self.max_occ = max_occ
        for arg in range(len(nuscenes.category)):
            self.d[nuscenes.category[arg]['name']] = arg
        self.max_ann = self.max_ann_nb()
        self.token_list = get_prediction_challenge_split('mini_train', dataroot=DATAROOT)
        """try:
            # raise FileNotFoundError
            with open('/dev/shm/cached_v11_%s_%d_%d_agents.bin' % (
                    self.step,
                    self.t_h, self.t_f), 'rb') as f:
                self.target_hist, self.fut, self.mask, self.neighbors_coordinates, self.lanes_coordinates, self.reference_lane_indx = pickle.load(f)
        except FileNotFoundError:
            self.target_hist, self.fut, self.mask, self.neighbors_coordinates, self.lanes_coordinates, self.reference_lane_indx = self.__getitem__()
            content = pickle.dumps([self.target_hist, self.fut, self.mask, self.neighbors_coordinates, self.lanes_coordinates, self.reference_lane_indx])
            with open('/dev/shm/cached_v11_%s_%d_%d_agents.bin' % (
                    self.step,
                    self.t_h, self.t_f), 'wb') as f:
                f.write(content)"""

    def __len__(self):
        return self.nb_samples

    def __getitem__(self, idx):
        # Nuscenes instance and sample token for prediction data point
        instance_token, sample_token = self.token_list[idx].split("_")
        my_sample = nuscenes.get('sample', sample_token)
        my_scene = nuscenes.get('scene', my_sample['scene_token'])
        log = nuscenes.get('log', my_scene['log_token'])
        map_name = log['location']

        # This class is used to handle all the map related inquiries.
        nusc_map = NuScenesMap(map_name=map_name, dataroot=DATAROOT)
        target_hist = np.zeros((self.t_h + 1, 5))
        t_hist = helper_ns.get_past_for_agent(instance_token, sample_token, seconds=self.t_h // 2,
                                           in_agent_frame=True, just_xy=True)
        t_hist_global = helper_ns.get_past_for_agent(instance_token, sample_token, seconds=self.t_h // 2,
                                           in_agent_frame=False, just_xy=True)
        t_hist = np.concatenate((t_hist, help.get_past_vel_for_agent(instance_token=instance_token, sample_token= sample_token,
                                                                       seconds=self.t_h // 2, in_agent_frame=False,
                                                                       just_xy=True)), axis=1)
        target_hist[:min(self.t_h, t_hist.shape[0]), :] = t_hist[:min(self.t_h, t_hist.shape[0]), :]
        fut = np.zeros((self.t_f, 2))
        f = helper_ns.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True,
                                        just_xy=True)
        f_global = helper_ns.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=False,
                                        just_xy=True)

        fut[:f.shape[0], :] = f
        hist = np.zeros((self.max_ann, self.t_h, 5))
        hist_global = np.zeros((self.max_ann, self.t_h, 2))
        annotation_record = helper_ns.get_sample_annotation(instance_token, sample_token)
        neighbors = helper_ns.get_past_for_sample(sample_token, seconds=self.t_h // 2, in_agent_frame=False,
                                               just_xy=True)
        """timestamps = []
        for i in neighbors:
            v = neighbors.get(i)
            print('NNNNNN : ', v)
            print('LEEEEEEEENN : ', len(v))
            print("------------------------------------------")
            t = list(nuscenes.get('sample', neighbors.get(i)[idex]['sample_token'])['timestamp'] for idex in range(len(v)))
            print(t)
            timestamps.append(t)"""

        l = my_sample['anns']
        for k in range(0, len(l)):
            ann_data = return_instance(my_sample, k)
            if (neighbors[ann_data['instance_token']].size > 0):
                hist[k, :min(neighbors[ann_data['instance_token']].shape[0], self.t_h),
                :2] = convert_global_coords_to_local(neighbors[ann_data['instance_token']][
                                                     :min(neighbors[ann_data['instance_token']].shape[0], self.t_h), :],
                                                     annotation_record['translation'],
                                                     annotation_record['rotation'])
                hist_global[k, :min(neighbors[ann_data['instance_token']].shape[0], self.t_h), :] = neighbors[ann_data['instance_token']][
                                                     :min(neighbors[ann_data['instance_token']].shape[0], self.t_h), :]
                ext_f = help.get_past_vel_for_agent(ann_data['instance_token'], ann_data['sample_token'],
                                                      seconds=self.t_h // 2,
                                                      in_agent_frame=False, just_xy=True)
                hist[k, :min(self.t_h, ext_f.shape[0]), 2:] = ext_f[:min(self.t_h, ext_f.shape[0]), :]

        mask = np.concatenate((np.ones(f.shape[0]), np.zeros((self.t_f - f.shape[0]))))
        closest_lanes = self.get_closest_lanes(nusc_map, t_hist_global[-1][0], t_hist_global[-1][1], 50)
        lanes_records = [(closest_lane, nusc_map.get_arcline_path(closest_lane)) for (closest_lane, _) in closest_lanes]

        current_pos_nbrs = []
        for i in neighbors:
            v = neighbors.get(i)
            if len(v) > 0:
                current_pos_nbrs.append((i, v[0])) # je suis pas sûre...
        nb_lane = {}
        lanes_res = np.zeros((self.n, self.m, 2))
        neighbors_hist = np.zeros((self.n, self.t_h + 1, 2))
        l = []
        neighb_hist = []
        for id, lane in lanes_records:
            distance = 10
            nb_id = None
            xy = None
            poses = arcline_path_utils.discretize_lane(lane, resolution_meters=0.5)
            for nb, xy in current_pos_nbrs:
                lanes_distances = np.linalg.norm(np.array(poses)[:, :2] - [xy[0], xy[1]], axis=1).min()
                if lanes_distances < distance:
                    distance = lanes_distances
                    nb_id = nb
            leng = arcline_path_utils.length_of_lane(lane)
            if leng == self.length_lane:
                lane_elongated = poses
            if leng > self.length_lane:
                lane_elongated = poses[0:self.length_lane*2]
            if leng < self.length_lane:
                fwd = 0
                fwd_dist = self.FORWARD_DISTANCE
                bckwd = 0
                bckwd_dist = self.BACKWARD_DISTANCE
                poses_out = []
                poses_in = []
                id_out, id_in = id, id
                while fwd < fwd_dist or bckwd < bckwd_dist:
                    if fwd< fwd_dist:
                        outgo = nusc_map.get_outgoing_lane_ids(id_out)
                        if len(outgo) == 0:
                            bckwd_dist = bckwd_dist*2
                            continue

                        outgo = outgo[0]
                        lane_record_out = nusc_map.get_arcline_path(outgo)
                        poses_out_ = arcline_path_utils.discretize_lane(lane_record_out, resolution_meters=0.5)
                        poses_out = poses_out + poses_out_
                        fwd = len(poses_out)//2
                        id_out = outgo
                    if bckwd < bckwd_dist:
                        incom = nusc_map.get_incoming_lane_ids(id_in)
                        if len(incom) == 0:
                            fwd_dist = fwd_dist*2
                            continue
                        incom = incom[0]
                        lane_record_in = nusc_map.get_arcline_path(incom)
                        poses_in_ = arcline_path_utils.discretize_lane(lane_record_in, resolution_meters=0.5)
                        poses_in = poses_in + poses_in_[::-1]
                        bckwd = len(poses_in) // 2
                        id_in = incom

                lane_elongated = poses_in[::-1] + poses + poses_out
            leng_elon = len(lane_elongated) // 2
            if leng_elon > self.length_lane:
                lane_elongated = lane_elongated[0:self.length_lane*2]
            if nb_id is not None:
                neigh_v = neighbors.get(nb_id)
                if len(neigh_v) < self.t_h + 1:
                    neigh_v = np.concatenate((np.array(neigh_v), np.zeros(((self.t_h + 1)-len(neigh_v), 2))), axis=0)
            if nb_id is None:
                neigh_v = np.zeros((self.t_h + 1, 2))

            l.append(lane_elongated)
            neighb_hist.append(neigh_v)
            nb_lane[id] = [nb_id, lane, poses, lane_elongated, neigh_v, distance]

        l = np.array(l)

        l = np.delete(l, 2, 2)
        neighb_hist = np.array(neighb_hist)
        lanes_res[:l.shape[0], :] = l
        neighbors_hist[:, :min(neighb_hist.shape[1], self.t_h+1), :] = neighb_hist

        # Convert global to local for neighbors and lanes
        lanes_coordinates = np.array(list(convert_global_coords_to_local(
            np.array(i), annotation_record['translation'], annotation_record['rotation']) for i in lanes_res))
        neighbors_coordinates = np.array(list(convert_global_coords_to_local(
            np.array(i), annotation_record['translation'], annotation_record['rotation']) for i in neighbors_hist))

        ann_datas = (return_instance(my_sample, k) for k in range(len(my_sample['anns'])))
        #instance_tokens, sample_tokens = list((ann_data['instance_token'], ann_data['sample_token']) for ann_data in ann_datas)
        nbs = []
        for l_id in nb_lane.keys():
            nbs.append(nb_lane.get(l_id)[0])
            lane_elo = nb_lane.get(l_id)[3]
            xy = nb_lane.get(l_id)[4]
            """x_p = []
            y_p = []
            ta = np.concatenate((t_hist_global[::-1], f_global), axis=0)
            x_t = []
            y_t = []
            for x, y in ta:
                x_t.append(x)
                y_t.append(y)
            plt.plot(x_t, y_t, '-b')
            for x, y, _ in lane_elo:
                x_p.append(x)
                y_p.append(y)
            plt.plot(x_p, y_p, '-g')
            x_n = []
            y_n = []
            for x, y in xy:
                x_n.append(x)
                y_n.append(y)
            plt.plot(x_n, y_n, '-r')
            plt.show()"""

        ext_f = np.array(list(help.get_past_vel_for_agent(ann_data['instance_token'], ann_data['sample_token'],
                                            seconds=self.t_h // 2,
                                            in_agent_frame=False, just_xy=True) for ann_data in ann_datas
                              if ann_data['instance_token'] in nbs))


        # Reference lane
        nu = (lambda j: j + 1)
        d = (lambda l: sum(min(np.linalg.norm(c - m) for m in l[1]) * nu(i) for i, c in enumerate(fut)))
        reference_lane_indx = min(enumerate(lanes_coordinates), key=d)[0]

        target_hist = target_hist[:, :2]

        return target_hist, fut, mask, neighbors_coordinates, lanes_coordinates,reference_lane_indx
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

    def get_closest_lanes(self, nusc_map, x: float, y: float, radius: float = 50):
        """
            Get closest lane id within a radius of query point. The distance from a point (x, y) to a lane is
                the minimum l2 distance from (x, y) to a point on the lane.
                :param x: X coordinate in global coordinate frame.
                :param y: Y Coordinate in global coordinate frame.
                :param radius: Radius around point to consider.
                :return: Lane id of closest lane within radius.
        """

        lanes = nusc_map.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])
        lanes = lanes['lane'] + lanes['lane_connector']

        discrete_points = nusc_map.discretize_lanes(lanes, 0.5)

        min_id = []
        for lane_id, points in discrete_points.items():
            distance = np.linalg.norm(np.array(points)[:, :2] - [x, y], axis=1).min()
            if distance <= radius:
                min_id.append((lane_id, distance))
        closest = sorted(min_id, key=lambda item: (item[1]))[0:self.n]

        return closest
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
    return my_annotation_metadata