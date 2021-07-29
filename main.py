from collections import defaultdict
from functools import partial
from typing import List, Any

import numpy as np
import yaml
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction import PredictHelper, convert_global_coords_to_local
from torch.utils.data import Dataset

from nuscenes import NuScenes


# set of N lanes candidates


# #############################################
def is_neighbor(reference, item_attrs, radius=10) -> bool:
    """

    :param reference:
    :param item_attrs:
    :param radius: In meters.
    :return:
    """
    xa, ya, za = reference['translation']
    xb, yb, zb = item_attrs['translation']
    xc, yc = xa - xb, ya - yb
    return xc * xc + yc * yc < radius * radius


def distance(a, b):
    xa, ya, za = a['translation']
    xb, yb, zb = b['translation']
    xc, yc = xa - xb, ya - yb
    return np.sqrt(xc * xc + yc * yc)


def distance_lane(reference_map, reference, lane):
    x_r, y_r, z_r = reference['translation']
    (x_l, y_l, z_l), _ = arcline_path_utils.project_pose_to_lane(
        reference['translation'], reference_map.get_arcline_path(lane))
    return np.sqrt(((x_r - x_l) ** 2) + ((y_r - y_l) ** 2))


class NS(Dataset):
    def __init__(self, config_path):
        with open(config_path, 'r') as yaml_file:
            self._config = yaml.safe_load(yaml_file)['ns_args']
        self._data_root = data_root = self._config['data_root']
        version, verbose = self._config['version'], self._config['verbose']

        # Useful objects
        self._ns = NuScenes(version, dataroot=data_root, verbose=verbose)
        self._helper = PredictHelper(self._ns)

        # Parameter useful for later
        self._history_duration = self._config['history_duration']
        self._lane_coordinates = self._config['lane_amount_of_coordinates']
        self._prediction_duration = self._config['prediction_duration']

        self._token_list = get_prediction_challenge_split(self._config['split'], dataroot=self._data_root)
        self._history, self._future, self._lanes, self._neighbors = self._load()

    def __len__(self):
        return len(self._token_list)

    def _load(self):
        h_d = self._history_duration
        p_d = self._prediction_duration
        # singapore-onenorth
        # singapore-hollandvillage
        # singapore-queenstown
        # boston-seaport
        nusc_map = NuScenesMap(map_name=self._config['map_name'], dataroot=self._data_root)

        expanded_list = [self._helper.get_sample_annotation(*i.split("_")) for i in self._token_list]
        instances = set(i['instance_token'] for i in expanded_list)

        grand_history, grand_future, grand_lanes, grand_neighbors = [], [], [], []

        # Choice of the agent: take the one with the most available samples
        try:
            agents = open('/dev/shm/cached_%s_%s_%d_%d_agents.bin' % (
                self._config['map_name'], self._config['split'], h_d, p_d
            ), 'r').read().strip().split(',')
        except FileNotFoundError:
            availability = defaultdict(int)
            for attributes in expanded_list:
                if attributes['category_name'] != 'vehicle.car':
                    continue
                if nusc_map.get_closest_lane(*attributes['translation'][:2], 5):
                    availability[attributes['instance_token']] += 1
            agents = list(filter((lambda x: availability.get(x, -1) > (h_d + p_d)), instances))
            open('/dev/shm/cached_%s_%s_%d_%d_agents.bin' % (
                self._config['map_name'], self._config['split'], h_d, p_d
            ), 'w').write(','.join(agents))

        print('Found %d candidates' % len(agents))

        for agent in agents:
            agent_attributes = [i for i in expanded_list if i['instance_token'] == agent]
            _timestamp = (lambda x: self._helper._timestamp_for_sample(x))
            agent_attributes = sorted(agent_attributes, key=(lambda x: _timestamp(x['sample_token'])))
            present = agent_attributes[len(agent_attributes) // 2]
            past = (self._helper.get_past_for_agent(agent, present['sample_token'], 10, False, False) + [present])[:h_d]
            future = self._helper.get_future_for_agent(agent, present['sample_token'], 10, True, True)[:p_d]
            assert len(past) == h_d, len(past)
            assert len(future) == p_d, len(future)

            history = np.array([r['translation'][:2] for r in past])
            history = convert_global_coords_to_local(history, present['translation'], present['rotation'])

            all_lanes = set()
            cars_on_lanes = dict()

            for neighbor in self._helper.get_annotations_for_sample(present['sample_token']):
                if not is_neighbor(present, neighbor, ):
                    continue
                lane = nusc_map.get_closest_lane(*neighbor['translation'][:2], 1)
                if not lane:
                    continue
                all_lanes.add(lane)
                cars_on_lanes.setdefault(lane, {})[neighbor['instance_token']] = neighbor

            distance_lanes = dict((i, distance_lane(nusc_map, present, i)) for i in all_lanes)
            all_lanes = sorted(all_lanes, key=distance_lanes.get)
            lanes_coordinates = []
            for lane in all_lanes:
                def split_in_n(record, n):
                    alpha = 0.1
                    first = arcline_path_utils.discretize_lane(record, alpha)
                    while len(first) != n:
                        if len(first) > n:
                            alpha += 0.0001
                        else:
                            alpha -= 0.0001
                        first = arcline_path_utils.discretize_lane(record, alpha)
                    return first

                coordinates = np.array(
                    [i[:2] for i in split_in_n(nusc_map.get_arcline_path(lane), self._lane_coordinates)])
                coordinates = convert_global_coords_to_local(coordinates, present['translation'], present['rotation'])
                lanes_coordinates.append(coordinates)

            car_coordinates: List[Any] = [None] * len(lanes_coordinates)
            for lane, _cars in cars_on_lanes.items():
                car = min(_cars.values(), key=partial(distance, present))
                _coordinates = self._helper.get_past_for_agent(car['instance_token'], car['sample_token'], 10, False,
                                                               False) + [car]
                _coordinates = [r['translation'][:2] for r in _coordinates[:h_d]]
                _coordinates = [(-np.inf, -np.inf)] * (h_d - len(_coordinates)) + _coordinates
                _coordinates = np.array(_coordinates)
                _coordinates = convert_global_coords_to_local(_coordinates, present['translation'], present['rotation'])
                car_coordinates[int(all_lanes.index(lane))] = _coordinates
                assert car['sample_token'] == present['sample_token'], (
                    car['instance_token'], car['sample_token'])

            print(car_coordinates)
            print(history[:1])
            print(lanes_coordinates[:1])
            print(car_coordinates[:1])

            grand_history.append(np.array(history))
            grand_future.append(np.array(future))
            grand_lanes.append(np.array(lanes_coordinates))
            grand_neighbors.append(np.array(car_coordinates))

        # V^(p), V^(f), L^n, V^n
        return grand_history, grand_future, grand_lanes, grand_neighbors


if __name__ == '__main__':
    NS('config.yml')
