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
    '''if item_attrs['category_name'] != 'vehicle.car':
        return False'''
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
            config = yaml.safe_load(yaml_file)
        ns_args = config['ns_args']
        self._data_root = data_root = ns_args['data_root']
        version, verbose = ns_args['version'], ns_args['verbose']

        # Useful objects
        self._ns = NuScenes(version, dataroot=data_root, verbose=verbose)
        self._helper = PredictHelper(self._ns)

    def train(self):
        token_list = get_prediction_challenge_split("mini_train", dataroot=self._data_root)
        nusc_map = NuScenesMap(map_name='singapore-onenorth', dataroot=self._data_root)

        expanded_list = [self._helper.get_sample_annotation(*i.split("_")) for i in token_list]
        instances = set(i['instance_token'] for i in expanded_list)

        # Choice of the agent: take the one with the most available samples
        try:
            agent = open('/dev/shm/_____agent.bin', 'r').read().strip()
        except FileNotFoundError:
            availability = defaultdict(int)
            for attributes in expanded_list:
                if nusc_map.get_closest_lane(*attributes['translation'][:2], 5):
                    availability[attributes['instance_token']] += 1
            agent = max(instances, key=(lambda x: availability.get(x, -1)))
            open('/dev/shm/_____agent.bin', 'w').write(agent)

        agent_attributes = [i for i in expanded_list if i['instance_token'] == agent]
        _timestamp = (lambda x: self._helper._timestamp_for_sample(x))
        agent_attributes = sorted(agent_attributes, key=(lambda x: _timestamp(x['sample_token'])))
        present = agent_attributes[len(agent_attributes) // 2]
        past = (self._helper.get_past_for_agent(agent, present['sample_token'], 10, False, False) + [present])[:6]
        future = self._helper.get_future_for_agent(agent, present['sample_token'], 10, True, True)[:6]
        assert len(past) == 6, len(past)
        assert len(future) == 6, len(future)

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

            coordinates = np.array([i[:2] for i in split_in_n(nusc_map.get_arcline_path(lane), 100)])
            coordinates = convert_global_coords_to_local(coordinates, present['translation'], present['rotation'])
            lanes_coordinates.append(coordinates)

        car_coordinates: List[Any] = [None] * len(lanes_coordinates)
        for lane, _cars in cars_on_lanes.items():
            car = min(_cars.values(), key=partial(distance, present))
            _coordinates = self._helper.get_past_for_agent(car['instance_token'], car['sample_token'], 10, False,
                                                           False) + [car]
            _coordinates = [r['translation'][:2] for r in _coordinates[:6]]
            _coordinates = [(-np.inf, -np.inf)] * (6 - len(_coordinates)) + _coordinates
            _coordinates = np.array(_coordinates)
            _coordinates = convert_global_coords_to_local(_coordinates, present['translation'], present['rotation'])
            car_coordinates[int(all_lanes.index(lane))] = _coordinates
            assert car['sample_token'] == present['sample_token'], (
                car['instance_token'], car['sample_token'])

        print(car_coordinates)
        # V^(p)
        _ = np.array(history)
        # V^(f)
        _ = np.array(future)
        # L^n
        _ = np.array(lanes_coordinates)
        # V^n
        _ = np.array(car_coordinates)
        print(history[:1])
        print(lanes_coordinates[:1])
        print(car_coordinates[:1])


if __name__ == '__main__':
    NS('config.yml').train()
