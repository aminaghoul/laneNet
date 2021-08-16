import itertools
import math
import pickle
from collections import defaultdict
from functools import partial
from queue import Queue
from threading import Thread
from typing import List, Any, Tuple

import numpy as np
import yaml
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.arcline_path_utils import length_of_lane
from nuscenes.map_expansion.map_api import NuScenesMap, locations
from nuscenes.prediction import PredictHelper, convert_global_coords_to_local
from torch.utils.data import Dataset

from nuscenes import NuScenes

from utils_nuscenes import *


class NS(Dataset):
    @classmethod
    def every_map(cls, config_path, split="mini_train"):
        for map_name in locations:
            yield NS(config_path, map_name, split)

    def __repr__(self):
        return 'NS(%r, %r, %r)' % (
            self._config_path,
            self._map_name,
            self._config['split'])

    def __init__(self, config_path, map_name, split):
        self._config_path = config_path

        with open(config_path, 'r') as yaml_file:
            self._config = yaml.safe_load(yaml_file)['ns_args']
            self._config['split'] = split

        self._map_name = map_name

        self._data_root = data_root = self._config['data_root']
        version, verbose = self._config['version'], self._config['verbose']

        # Useful objects
        self._ns = NuScenes(version, dataroot=data_root, verbose=verbose)
        self.helper = self._helper = PredictHelper(self._ns)

        #
        self._n = self._config['nb_lane_candidates']
        self._forward_lane = self._config['forward_lane']
        self._backward_lane = self._config['backward_lane']
        self._precision_lane = self._config['precision_lane']

        # Parameter useful for later
        self._history_duration = self._config['history_duration']
        self._lane_coordinates = int((self._forward_lane + self._backward_lane) / self._precision_lane)
        self._prediction_duration = self._config['prediction_duration']

        self._nusc_map = NuScenesMap(map_name=self._map_name, dataroot=self._data_root)

        self._token_list = get_prediction_challenge_split(self._config['split'], dataroot=self._data_root)

        try:
            # raise FileNotFoundError
            with open('/dev/shm/cached_v9_%s_%s_%d_%d_agents.bin' % (
                    self._map_name, self._config['split'],
                    self._history_duration, self._prediction_duration), 'rb') as f:
                self._history, self._future, self._lanes, self._neighbors, self._reference_lanes = pickle.load(f)
        except FileNotFoundError:
            self._history, self._future, self._lanes, self._neighbors, self._reference_lanes = self._load()
            content = pickle.dumps([self._history, self._future, self._lanes, self._neighbors, self._reference_lanes])
            with open('/dev/shm/cached_v9_%s_%s_%d_%d_agents.bin' % (
                    self._map_name, self._config['split'],
                    self._history_duration, self._prediction_duration), 'wb') as f:
                f.write(content)
        if self._config['nb_coordinates'] == 2:
            self._history = [np.array([[x, y] for x, y, _, __, ___ in sim]) for sim in self._history]
            self._future = [np.array([[x, y] for x, y, _, __, ___ in sim]) for sim in self._future]
            self._neighbors = [np.array([[[x, y] for x, y, _, __, ___ in n] for n in sim]) for sim in self._neighbors]

    def __len__(self):
        # TODO: Remove this
        # print('Length', len(self._history))
        return len(self._history)

    def _load(self):
        h_d = self._history_duration
        p_d = self._prediction_duration
        nusc_map = self._nusc_map

        expanded_list = [self._helper.get_sample_annotation(*i.split("_")) for i in self._token_list]

        instances = set(i['instance_token'] for i in expanded_list)

        grand_history, grand_future, grand_lanes, grand_neighbors = [], [], [], []

        # Choice of the agent: take the one with the most available samples
        availability = defaultdict(int)
        lanes = defaultdict(list)
        coords_a = defaultdict(list)
        samples = defaultdict(list)
        for attributes in expanded_list:
            if not attributes['category_name'].startswith('vehicle.'):
                continue
            closest_lane = nusc_map.get_closest_lane(*attributes['translation'][:2], 1)
            if closest_lane:
                # samples[attributes['instance_token']]
                samples[attributes['instance_token']].append(attributes['sample_token'])
                coords_a[attributes['instance_token']].append([*attributes['translation'][:2]])
                availability[attributes['instance_token']] += 1
                lanes[attributes['instance_token']].append(closest_lane)

        agents = list(filter((lambda x: availability.get(x, -1) > (h_d + p_d + 1)), instances))
        list_coords_l = []
        list_coords_a = []

        for agent in agents:
            if agent in samples.keys():
                sample_token = samples[agent][h_d + 1]
                list_coords_a.append(
                    self.helper.get_past_for_agent(agent, sample_token, seconds=12, in_agent_frame=False))
                close_lane = lanes[agent]
                for l in close_lane:
                    lane_record = nusc_map.get_arcline_path(l)
                    poses = arcline_path_utils.discretize_lane(lane_record, resolution_meters=1)
                    list_coords_l.append(poses)

        def void():
            import matplotlib.pyplot

            for index, i in enumerate(list_coords_a):
                __x = [j[0] for j in i]
                __y = [j[1] for j in i]
                if -np.inf not in __x:
                    matplotlib.pyplot.plot(list(__x), list(__y), label='Target %d' % index)
            for index, i in enumerate(list_coords_l):
                __x = [j[0] for j in i]
                __y = [j[1] for j in i]
                if -np.inf not in __x:
                    matplotlib.pyplot.plot(list(__x), list(__y), label='Lane %d' % index)

            matplotlib.pyplot.legend()
            matplotlib.pyplot.show()


if __name__ == '__main__':
    for ns in NS.every_map('config.yml'):
        pass
