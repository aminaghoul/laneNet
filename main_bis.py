from itertools import chain
from traceback import print_exception, print_exc
from typing import List, Iterator, Generator

import numpy as np
import yaml
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap

from nuscenes.nuscenes import NuScenes

from matplotlib import pyplot as plt
from sqlitedict import SqliteDict
from tqdm import tqdm

from nuscenes.prediction import convert_global_coords_to_local
from pandas._libs.internals import defaultdict

config_file = 'config.yml'
with open(config_file, 'r') as yaml_file:
    global_config = yaml.safe_load(yaml_file)
    ns_config = global_config['ns_args']
    ln_config = global_config['ln_args']

tau = ns_config['history_duration']
h = ns_config['prediction_duration']
n = ln_config['nb_lane_candidates']

M = float((ns_config['forward_lane'] + ns_config['backward_lane']) / ns_config['precision_lane'])
FORWARD_DISTANCE, BACKWARD_DISTANCE = ns_config['forward_lane'], ns_config['backward_lane']
PRECISION_LANE = ns_config['precision_lane']

assert M.is_integer()
M = int(M)

nusc = None


def get_nusc():
    global nusc
    if nusc is None:
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


def discretize_lanes(nusc_map: NuScenesMap, lanes, precision_centi_meters: int = 50):
    """
    Warning: The returned object is not a copy, so avoid changing it.
    TODO: Make a copy of the object to avoid possible errors
    TODO: Thread safety
    :param nusc_map:
    :param lanes:
    :param precision_centi_meters:
    :return:
    """
    if isinstance(lanes, str):
        return discretize_lanes(nusc_map, [lanes], precision_centi_meters)[0]
    new = [i for i in lanes if i not in _discredized_lanes]
    for k, v in nusc_map.discretize_lanes(new, precision_centi_meters / 100).items():
        _discredized_lanes[k, precision_centi_meters] = v
    return [[np.array(j[:2]) for j in _discredized_lanes[i, precision_centi_meters]] for i in lanes]


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

        self._split_history = (lambda _list: _list[:self._reference_sample_index + 1][-tau - 1:])

        self._extended_ego_history = self._split_history([self._ego[t] for t in self._samples_timestamps])
        self._ego = [np.array(self._ego[timestamp]['translation'][:2]) for timestamp in self._samples_timestamps]
        self._ego_history = self._split_history(self._ego)
        self._ego_future = np.array(self._ego[self._reference_sample_index + 1:][:h])
        translation = self._extended_ego_history[-1]['translation']
        rotation = self._extended_ego_history[-1]['rotation']

        x, y = self._ego_history[-1]

        self._close_lanes = list(self._filter_uniques(iter_lanes(self._nusc_map, x, y, 100)))

        self._show_neighbors()
        self._neighbors, self._lanes = self._get_neighbors_and_lanes()
        self._lanes_coordinates = []
        for neighbor, lane in zip(self._neighbors, self._lanes):
            lane = np.array(list(chain.from_iterable(discretize_lanes(self._nusc_map, lane, 1))))
            distances = [np.linalg.norm(lane[i] - lane[i - 1]) for i in range(1, len(lane))]
            lane = [coordinates for coordinates, distance in zip(lane, distances) if distance > 0.01 * 0.1]
            reference_point, _ = min(enumerate(lane), key=(lambda arg: np.linalg.norm(neighbor[-1] - arg[1][:2])))
            self._lanes_coordinates.append(lane[slice(
                reference_point - int(BACKWARD_DISTANCE * 100),
                reference_point + int(FORWARD_DISTANCE * 100),
                int(PRECISION_LANE * 100))])

        nu = (lambda j: j + 1)
        d = (lambda l: sum(min(np.linalg.norm(c - m) for m in l[1]) * nu(i) for i, c in enumerate(self._ego_future)))
        reference_lane = min(enumerate(self._lanes_coordinates), key=d)[0]

        self._lanes_coordinates = np.array(list(convert_global_coords_to_local(
            np.array(i), translation, rotation) for i in self._lanes_coordinates))
        self._neighbors = np.array(list(convert_global_coords_to_local(
            i, translation, rotation) for i in self._neighbors))
        self._ego_history = convert_global_coords_to_local(self._ego_history, translation, rotation)
        self._ego_future = convert_global_coords_to_local(self._ego_future, translation, rotation)
        self.columns = \
            self._ego_history, self._ego_future, \
            self._neighbors, self._lanes_coordinates, \
            np.array([reference_lane]), translation, rotation

    def _show_neighbors(self):
        print(self.__scene_data)
        print(self._ego_history)
        """show_trajectory(self._ego_history, *discretize_lanes(self._nusc_map, self._close_lanes))
        show_trajectory(*self._neighbors, *[discretize_lanes(
            self._nusc_map, next(iter_lanes(
                self._nusc_map, *neighbor[-1], 10))) for neighbor in self._neighbors])"""

    def _get_neighbors_and_lanes(self):
        neighbors = self._get_neighbors()
        x, y = self._ego_history[-1]

        # discretize_lanes(self._nusc_map, possible_lane)

        # First step is to get all the lanes associated with each neighbors
        # This will be a mapping of all the close lanes to possible neighbor
        #  lanes
        close_possibilities = dict((i, {}) for i in self._close_lanes)
        close_possibilities[None] = {}

        neighbors_distances_to_ego = [np.linalg.norm(neighbor - self._ego_history[-1]) for neighbor in neighbors]
        neighbors_distances_to_lane = [0 for _ in neighbors]

        for neighbor_index, neighbor in enumerate(neighbors):
            closest_lane = next(iter_lanes(self._nusc_map, *neighbor[-1], 10))
            neighbors_distances_to_lane[neighbor_index] = np.linalg.norm(
                discretize_lanes(self._nusc_map, [closest_lane])[0] - neighbor[-1], axis=1).min()
            possible_lanes = list(self._elongate_lanes([closest_lane]))
            for possible_lane in possible_lanes:
                close_lane = [i for i in possible_lane if i in self._close_lanes]
                if not close_lane:
                    close_lane = [None]
                for close_lane in close_lane:
                    close_possibilities[close_lane][neighbor_index] = possible_lane

        station = [np.linalg.norm(neighbor[0] - neighbor[-1]) for neighbor in neighbors]
        score = [i + j - k for i, j, k in zip(neighbors_distances_to_ego, neighbors_distances_to_lane, station)]

        associated_lanes = set()
        while close_possibilities:
            # I use a list to avoid the error that arise when changing
            #  the size of the dict while going through it.
            for key in list(close_possibilities):

                value = close_possibilities.pop(key)
                # TODO: Find ways to use those
                if not value:
                    continue
                if len(value) == 1:
                    (val, val2), = value.items()
                    associated_lanes.add((key, val, val2))
                    # We remove the neighbor index from other possibilities
                    for k, v in close_possibilities.items():
                        close_possibilities[k] = dict((i, j) for i, j in v.items() if i != val)
                else:
                    # We put ambiguous values back
                    close_possibilities[key] = value

            if not close_possibilities or set(close_possibilities) == {None}:
                break
            closest_undecided = next(i for i in self._close_lanes if i in close_possibilities)
            accepted = min(close_possibilities[closest_undecided], key=score.__getitem__)
            accepted, *rejected = sorted(close_possibilities[closest_undecided], key=score.__getitem__)
            # show_trajectory(*(neighbors[r] for r in (accepted, *rejected)))
            associated_lanes.add((closest_undecided, accepted, close_possibilities.pop(closest_undecided)[accepted]))
            for k, v in close_possibilities.items():
                close_possibilities[k] = dict((i, j) for i, j in v.items() if i != accepted)

        def iterate():
            for i in self._close_lanes:
                for j, k, l in associated_lanes:
                    if j == i:
                        yield j, k, l

        def get_coordinates_of_lanes(full_lane):
            return list(chain.from_iterable(discretize_lanes(self._nusc_map, full_lane)))

        associated_lanes = list(iterate())[:n]
        p = [i[-1] for i in associated_lanes]
        n_ = [i[1] for i in associated_lanes]
        args = [neighbors[i] for i in n_]
        # print(list(arg[-1] for arg in args))
        # print(self.__scene_data)
        for lane in iter_lanes(self._nusc_map, x, y, 50):
            args.append(discretize_lanes(self._nusc_map, [lane])[0])
        # show_trajectory(*args, *map(get_coordinates_of_lanes, p))
        # show_trajectory(self._ego_history, *neighbors, )
        return [neighbors[i] for i in n_], [i[-1] for i in associated_lanes]

    def _filter_uniques(self, lanes_iterator: Iterator[str]) -> Generator[str, None, None]:
        connected = set()
        for lane in lanes_iterator:
            if lane not in connected:
                yield lane
            connected.update(self._nusc_map.get_incoming_lane_ids(lane))
            connected.update(self._nusc_map.get_outgoing_lane_ids(lane))
            connected.add(lane)

    def _get_full_lanes(self, history: np.array) -> List[np.array]:
        try:
            lanes = iter_lanes(self._nusc_map, *history[-1], 10)
        except ValueError:
            return []
        lanes = self._elongate_lanes(lanes)
        raise ValueError(list(lanes))

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
        """Get a list of histories of neighbors"""
        neighbors = []  # [., tau, 2]
        for instance in self._instances:
            instance_coordinates = np.full((tau + 1, 2), np.nan)
            first_value, first_index = None, -1
            disappeared = False
            last_disappeared = None
            for sample_index, sample in enumerate(self._split_history(self._samples_instances)):
                annotations = sample.get(instance)
                if annotations is not None:
                    assert not disappeared, disappeared
                    instance_coordinates[sample_index] = annotations['translation'][:2]
                    if first_value is None:
                        first_value = np.array(annotations['translation'][:2])
                        first_index = sample_index
                    last_disappeared = False
                elif first_value is not None:
                    disappeared = True
                    last_disappeared = True

            if first_index > -1 and not last_disappeared:
                if first_index:
                    for sample_index in range(first_index):
                        instance_coordinates[sample_index] = first_value
                try:
                    next(iter_lanes(self._nusc_map, *instance_coordinates[-1], 10))
                except StopIteration:
                    pass
                else:
                    neighbors.append(instance_coordinates)
        return sorted(neighbors, key=(lambda neighbor: np.linalg.norm(neighbor[-1] - self._ego_history[-1])))

    def _determinate_reference_sample(self):
        """Determine the reference sample, as a tuple sample_id, sample_token.
        This will be chosen to have the maximum amount of neighbor coordinates
            defined in the tau previous samples."""
        # We will have two scores, one to determine the amount of neighbors in the
        #  current sample and one to determine the amount of neighbor in the whole
        #  history.
        whole_scores = defaultdict(int)
        for s in range(tau, len(self._samples_timestamps) - h):
            absent_at_least_once = set()
            for index in range(s - tau, s + 1):
                for instance in self._instances:
                    if instance not in self._samples_instances[index]:
                        absent_at_least_once.add(instance)
            whole_scores[s] += len(absent_at_least_once)
        current_score = dict()
        for s in range(tau, len(self._samples_timestamps)):
            current_score[s] = - len(self._samples_instances[s])
        full_scores = dict((i, (current_score[i], whole_scores[i])) for i in current_score)
        s = min(full_scores, key=full_scores.__getitem__)
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


def get_dataset():
    def generate():
        # print(differences)
        """for scene in nusc.scene:
            print(scene)"""
        all_columns = SqliteDict('/var/tmp/ns_cache.db')
        # preprocessed_data = [[] for i in range(5)]
        if nusc is not None:
            timestamps = sorted([pose['timestamp'] * 1e-6 for pose in nusc.ego_pose])
            differences = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
            for scene in tqdm(list(sorted(nusc.scene, key=(lambda x: x['name'])))):
                try:
                    all_columns[scene['name']]
                except KeyError:
                    try:
                        all_columns[scene['name']] = Scene(scene).columns
                        all_columns.commit()
                    except KeyboardInterrupt:
                        raise
                    except BaseException as e:
                        print('Error while processing %r: %r' % (scene['name'], e))
                        print_exc()
                # for index, arg in enumerate(preprocess_scene(scene)):
                #     preprocessed_data[index].append(arg)
        items = list(all_columns.values())
        for history, future, neighbors, lanes, reference_lane, translation, rotation in items:
            if len(lanes) < n:
                try:
                    lanes = np.concatenate((lanes, np.zeros((n - len(lanes), M, 2))))
                except ValueError:
                    print(lanes)
                    raise

            if len(neighbors) < n:
                neighbors = np.concatenate((neighbors, np.zeros((n - len(neighbors), tau + 1, 2))))
            history = np.zeros((tau + 1, 2))
            future = np.zeros((h, 2))
            neighbors = np.zeros((n, tau + 1, 2))
            lanes = np.zeros((n, M, 2))

            translation = np.zeros((3,))
            rotation = np.zeros((4,))
            yield history, future, neighbors, lanes, reference_lane, translation, rotation
            '''\
                np.array(history, dtype=np.float64),
                
            a = \
                np.array(future, dtype=np.float64), \
                np.array(neighbors, dtype=np.float64), \
                np.array(lanes, dtype=np.float64), \
                np.array(reference_lane, dtype=np.float64)
        # , \
        # np.array(translation), np.array(rotation)'''

    return list(generate())


if __name__ == '__main__':
    #get_nusc()
    get_dataset()
