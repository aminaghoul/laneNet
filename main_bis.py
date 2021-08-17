from itertools import chain
from typing import List, Iterator, Generator

import numpy as np
import yaml
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap

from nuscenes.nuscenes import NuScenes

from matplotlib import pyplot as plt
from pandas._libs.internals import defaultdict

config_file = 'config.yml'
with open(config_file, 'r') as yaml_file:
    global_config = yaml.safe_load(yaml_file)
    ns_config = global_config['ns_args']
    ln_config = global_config['ln_args']

tau = ns_config['history_duration']
h = ns_config['prediction_duration']
n = ln_config['nb_lane_candidates']

M = float(ns_config['forward_lane'] + ns_config['backward_lane'] / ns_config['precision_lane'])
FORWARD_DISTANCE, BACKWARD_DISTANCE = ns_config['forward_lane'], ns_config['backward_lane']

assert M.is_integer()
M = int(M)

nusc = NuScenes(version=ns_config['version'], dataroot=ns_config['data_root'], verbose=True)


def show_trajectory(*coordinates):
    """

    :param coordinates: coordinates[i]: [nb_coord, 2]
    :return:
    """
    args = []
    for c in coordinates:
        args.extend((*zip(*c), '-'))
    plt.plot(*args)
    plt.show()


def unpack_scene(scene):
    timestamps_sample = []
    sample_tokens = []
    instance_tokens = []

    st = scene['first_sample_token']
    sample = nusc.get('sample', st)
    sample_tokens.append(sample['token'])
    instances = {}
    for token in sample['anns']:
        instance = nusc.get('sample_annotation', token)
        if instance['category_name'].startswith('vehicle.'):
            instances[instance['instance_token']] = instance

    instance_tokens.append(instances)
    timestamps_sample.append(sample['timestamp'])
    while not sample['next'] == "":
        sample = nusc.get('sample', sample['next'])
        sample_tokens.append(sample['token'])
        instances = {}
        for token in sample['anns']:
            instance = nusc.get('sample_annotation', token)
            if instance['category_name'].startswith('vehicle.'):
                instances[instance['instance_token']] = instance

        instance_tokens.append(instances)
        timestamps_sample.append(sample['timestamp'])
    return timestamps_sample, sample_tokens, instance_tokens


# TODO: Cache this function?
def get_length_of_lane(nusc_map: NuScenesMap, lane_id):
    return arcline_path_utils.length_of_lane(
        nusc_map.get_arcline_path(lane_id))


# TODO: Add the possibility for more than one choice of precision
_discredized_lanes = dict()


def discretize_lanes(nusc_map: NuScenesMap, lanes):
    """
    Warning: The returned object is not a copy, so avoid changing it.
    TODO: Make a copy of the object to avoid possible errors
    TODO: Thread safety
    :param nusc_map:
    :param lanes:
    :return:
    """
    new = [i for i in lanes if i not in _discredized_lanes]
    _discredized_lanes.update(nusc_map.discretize_lanes(new, 0.5))
    return [[j[:2] for j in _discredized_lanes[i]] for i in lanes]


# nusc.render_sample(st)
def iter_lanes(nusc_map, x, y, radius):
    # TODO: Set the radius here to be able to increase it at will
    """Iterate through lanes based on a reference point, a radius,
        and sorted according to its distance to the reference."""
    lanes = nusc_map.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])
    lanes = lanes['lane'] + lanes['lane_connector']
    lanes_distances = dict()

    for lane_id, points in zip(lanes, discretize_lanes(nusc_map, lanes)):
        lanes_distances[lane_id] = np.linalg.norm(np.array(points)[:, :2] - [x, y], axis=1).min()

    yield from sorted(lanes, key=lanes_distances.get)


def cut_iterator(iterator, n):
    for i, v in enumerate(iterator, 2):
        yield v
        if i > n:
            return
    raise RuntimeError('Not enough data (excepted %d, got less)' % n)


def preprocess_scene(scene):
    timestamps_samples, sample_tokens, instance_tokens = unpack_scene(scene)
    # EGO VEHICLE
    history_ego = ego[:tau]
    future_ego = ego[tau:h + tau]
    history_ego_coords = [e['translation'][:2] for e in history_ego]
    future_ego_coords = [e['translation'][:2] for e in future_ego]

    # Reference point
    x, y = history_ego_coords[tau - 1]
    lanes = list(cut_iterator(elongate_lanes(nusc_map, iter_lanes(nusc_map, x, y, 10), ), 6))

    # LANES
    # TODO: What is a lane_connector
    discrete_points = nusc_map.discretize_lanes(lanes, 0.5)
    lanes_coordinates = [[i[:2] for i in j] for j in discrete_points.values()]

    print(lanes_coordinates)
    show_trajectory([i['translation'][:2] for i in ego], *lanes_coordinates)

    raise ValueError(lanes)

    def void():
        lanes = nusc_map.get_closest_lane(*history_ego_coords[tau - 1], 10)
        print(lanes)
        # REFERENCE LANE :
        nu = (lambda j: j + 1)
        d = (lambda l: sum(min(np.linalg.norm(c - m) for m in l) * nu(i) for i, c in enumerate(future_ego_coords)))

        raise SystemExit


class Scene:
    def __init__(self, scene_data):
        # We start by storing the information given
        self.__scene_data = scene_data

        # Along with some information used throughout the code

        # Since the ego vehicle does not have an instance id, we will have
        #  to sort it, then access it via the timestamp given by the sample
        # Note: in this file, agent will be synonym of neighbor, as the target
        #  agent will be referred to as "ego".
        self._ego = dict((i['timestamp'], i) for i in nusc.ego_pose)

        # This is really useful to make sure we take the
        #  correct map.
        # TODO: What can we do with this?
        self.__log = nusc.get('log', scene_data['log_token'])
        self.__map_name = self.__log['location']

        # This class is used to handle all the map related inquiries.
        self._nusc_map = NuScenesMap(map_name=self.__map_name, dataroot=ns_config['data_root'])

        # TODO: Is the following a good idea?
        # In this class, we will sort every sample based on its timestamp, to avoid having
        #  to keep track of all the sample tokens, i.e: the item 0 of a list will correspond
        #  to the earliest sample in the scent. Note: This will often mean we will have to
        #  add fillers to make sure everything has a homogeneous shape
        self._samples_timestamps, self._samples_tokens, self._samples_instances = unpack_scene(scene_data)
        assert self._samples_timestamps == sorted(self._samples_timestamps)
        self._instances = sorted(set(chain.from_iterable(self._samples_instances)))
        self._reference_sample_index, self._reference_sample_token = self._determinate_reference_sample()
        self._reference_timestamp = self._samples_timestamps[self._reference_sample_index]

        self._split_history = (lambda _list: _list[:self._reference_sample_index + 1][-tau:])

        self._ego = [self._ego[timestamp]['translation'][:2] for timestamp in self._samples_timestamps]
        self._ego_history = self._split_history(self._ego)
        self._ego_future = self._ego[self._reference_sample_index + 1:][:h]

        x, y = self._ego_history[-1]

        self._close_lanes = list(self._filter_uniques(iter_lanes(self._nusc_map, x, y, 10)))
        show_trajectory(*discretize_lanes(self._nusc_map, self._close_lanes))
        raise ValueError()

        neighbors = self._neighbors = self._get_neighbors()
        # First step is to get all the lanes associated with each neighbors
        # self._neighbors_lanes = list()
        # for neighbor in neighbors:
        #     self._neighbors_lanes.append(self._get_full_lanes(neighbor))

        lanes = iter_lanes(self._nusc_map, x, y, 10)
        lanes = list(self._elongate_lanes(lanes))
        # lanes = list(cut_iterator(lanes, n))
        coordinates = [discretize_lanes(self._nusc_map, lane) for lane in lanes]
        coordinates = [[[j[:2] for j in i] for i in c.values()] for c in coordinates]
        show_trajectory(*chain.from_iterable(coordinates), self._ego_history, *neighbors, )
        # In order to simplify the lane finding, we will associate each
        #  agent to a lane.

        raise ValueError

    def _filter_uniques(self, lanes_iterator: Iterator[str]) -> Generator[str, None, None]:
        connected = set()
        for lane in lanes_iterator:
            if lane not in connected:
                yield lane
            connected.update(self._nusc_map.get_incoming_lane_ids(lane))
            connected.update(self._nusc_map.get_outgoing_lane_ids(lane))

    """def _get_full_lanes(self, history: np.array) -> List[np.array]:
        try:
            lanes = iter_lanes(self._nusc_map, *history[-1], 10)
        except ValueError:
            return []
        lanes = self._elongate_lanes(lanes)
        raise ValueError(list(lanes))"""

    def _put_together(self, lane_id: str, callback: callable, remaining_length: float, *already):
        for lane in callback(lane_id):
            remaining = remaining_length - get_length_of_lane(self._nusc_map, lane)
            if remaining < 0:
                yield (*already, lane)
            else:
                yield from self._put_together(lane, callback, remaining, *already, lane)

    def _elongate_lanes(self, lanes_iterator):
        already_found = set()
        for lane in lanes_iterator:
            # If the lane was found,
            #  then we don't use it again
            if lane in already_found:
                continue

            # First step: incoming lanes
            incoming_possibilities = self._put_together(
                lane, self._nusc_map.get_incoming_lane_ids,
                BACKWARD_DISTANCE)
            outgoing_possibilities = self._put_together(
                lane, self._nusc_map.get_outgoing_lane_ids,
                FORWARD_DISTANCE)
            for incoming_possibility in incoming_possibilities:
                for outgoing_possibility in outgoing_possibilities:
                    yield (*reversed(incoming_possibility), lane, *outgoing_possibility)

            already_found.add(lane)

            # raise ValueError(lane)

    def _get_neighbors(self):
        neighbors = []  # np.array((n, tau, 2))
        for instance in self._instances:
            instance_coordinates = np.full((tau + 1, 2), np.nan)
            first_value, first_index = None, -1
            for sample_index, sample in enumerate(self._split_history(self._samples_instances)):
                annotations = sample.get(instance)
                if annotations is not None:
                    instance_coordinates[sample_index] = annotations['translation'][:2]
                    if first_value is None:
                        first_value = np.array(annotations['translation'][:2])
                        first_index = sample_index
                else:
                    # We should not get a car disappearing
                    assert first_value is None

            if first_index > -1:
                if first_index:
                    for sample_index in range(first_index):
                        instance_coordinates[sample_index] = first_value
                neighbors.append(instance_coordinates)
        return neighbors

    def _determinate_reference_sample(self):
        """Determine the reference sample, as a tuple sample_id, sample_token.
        This will be chosen to have the maximum amount of neighbor coordinates
            defined in the tau previous samples."""
        scores = defaultdict(int)
        for s in range(tau, len(self._samples_timestamps)):
            absent_at_least_once = set()
            for index in range(s - tau, s + 1):
                for instance in self._instances:
                    if instance not in self._samples_instances[index]:
                        absent_at_least_once.add(instance)
            scores[s] += len(absent_at_least_once)
        s = min(scores, key=scores.get)
        return s, self._samples_tokens[s]

    # TODO: Is this useful?
    def _unpack_scene(self):
        timestamps_sample = []
        sample_tokens = []
        instance_tokens = []

        st = self.__scene_data['first_sample_token']
        sample = nusc.get('sample', st)
        while True:
            sample_tokens.append(sample['token'])
            instance_tokens.append(sample['anns'])
            timestamps_sample.append(sample['timestamp'])
            if not sample['next']:
                return timestamps_sample, sample_tokens, instance_tokens
            sample = nusc.get('sample', sample['next'])

    def _associate_neighbors_to_lanes(self):
        # We will determine all of the data
        # The first step will be to determine which agents are in lanes
        for i in self._samples_instances:
            print(''.join('X' if j in i else ' ' for j in self._instances))
        raise ValueError()


def main():
    timestamps = sorted([pose['timestamp'] * 1e-6 for pose in nusc.ego_pose])
    differences = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    # print(differences)
    """for scene in nusc.scene:
        print(scene)"""

    preprocessed_data = [[] for i in range(5)]
    for scene in nusc.scene:
        Scene(scene)
        # for index, arg in enumerate(preprocess_scene(scene)):
        #     preprocessed_data[index].append(arg)


if __name__ == '__main__':
    main()
