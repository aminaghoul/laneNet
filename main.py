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

# Define the radius we have when looking for a lane
# TODO: Use this or delete this
IS_ON_LANE = 1


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
    @classmethod
    def every_map(cls, config_path, multi_thread=False):
        # TODO: Test the method with multi_thread on
        def worker(_map_name: str):
            try:
                output.put((None, NS(config_path, _map_name)))
            except BaseException as exception:
                output.put((exception, None))

        if multi_thread:
            output = Queue()
            for map_name in locations:
                Thread(
                    target=worker,
                    args=[map_name],
                    daemon=True).start()
            for index in range(len(locations)):
                error, result = output.get()
                # TODO: Stop the other workers
                if error is not None:
                    raise error
                yield result
        else:
            for map_name in locations:
                yield NS(config_path, map_name)

    def __repr__(self):
        return 'NS(%r, %r)' % (
            self._config_path,
            self._map_name)

    def __init__(self, config_path, map_name):
        self._config_path = config_path

        with open(config_path, 'r') as yaml_file:
            self._config = yaml.safe_load(yaml_file)['ns_args']

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
            with open('/dev/shm/cached_v5_%s_%s_%d_%d_agents.bin' % (
                    self._map_name, self._config['split'],
                    self._history_duration, self._prediction_duration), 'rb') as f:
                self._history, self._future, self._lanes, self._neighbors, self._reference_lanes = pickle.load(f)
        except FileNotFoundError:
            self._history, self._future, self._lanes, self._neighbors, self._reference_lanes = self._load()
            content = pickle.dumps([self._history, self._future, self._lanes, self._neighbors, self._reference_lanes])
            with open('/dev/shm/cached_v5_%s_%s_%d_%d_agents.bin' % (
                    self._map_name, self._config['split'],
                    self._history_duration, self._prediction_duration), 'wb') as f:
                f.write(content)

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
        try:
            agents = open('/dev/shm/cached_agent_v4_%s_%s_%d_%d_agents.bin' % (
                self._map_name, self._config['split'], h_d, p_d
            ), 'r').read().strip().split(',')
        except FileNotFoundError:
            availability = defaultdict(int)
            for attributes in expanded_list:
                if not attributes['category_name'].startswith('vehicle.'):
                    continue
                if nusc_map.get_closest_lane(*attributes['translation'][:2], 3):
                    availability[attributes['instance_token']] += 1
            agents = list(filter((lambda x: availability.get(x, -1) > (h_d + p_d + 1)), instances))
            open('/dev/shm/cached_agent_v4_%s_%s_%d_%d_agents.bin' % (
                self._map_name, self._config['split'], h_d, p_d
            ), 'w').write(','.join(agents))

        print('Found %d candidates' % len(agents))

        for agent in agents:
            # The first step is to get the list of all the attributes associated with the agent
            agent_attributes = [i for i in expanded_list if i['instance_token'] == agent]
            # TODO: Replace this with an assert, because this should (I hope) already be the case
            _timestamp = (lambda x: self._helper._timestamp_for_sample(x))
            agent_attributes = sorted(agent_attributes, key=(lambda x: _timestamp(x['sample_token'])))

            # This is important, as this will be our reference:
            #  we will take our neighbors based on the distance to
            #  at this point in time.
            # This is also important, as this will be treated
            #  as the origin of the map.
            # TODO: Can we iterate through all the different possibilities we have?
            # We will select the current one to be the first step that allow us
            #  to have h_d points in the past and p_d points in the future.
            present = agent_attributes[h_d]

            # We take more than needed, then truncate to take only what we need.
            # Note: We take the global coordinates as we need them to determine our neighbors
            # Second note: It's important to make sure values are sanitized
            assert p_d > 0, p_d
            assert h_d > 0, h_d

            # Third note: it's also important to truncate well:
            # - truncate in the future means take the p_d first
            future = self._helper.get_future_for_agent(agent, present['sample_token'], p_d * 1.5, False, False)[:p_d]

            # - truncate in the past means take the h_d first, then include the current one at the beginning.
            # - we don't limit the past as much as the future in order to make sure we are able to recover
            #   the past locations of the neighbors using `extended_past`
            past = [present] + self._helper.get_past_for_agent(agent, present['sample_token'], h_d * 15, False, False)
            extended_past, past = past, past[:h_d + 1]

            # This is important to have consistency in our data
            assert len(past) == h_d + 1, len(past)
            assert len(future) == p_d, len(future)

            # This is a part of the first input of our network,
            #  the last locations of the target agent, a.k.a Vp
            past_translations = np.array([r['translation'][:2] for r in past])

            # This will be used when training our network, as the expected result, a.k.a Vf
            future_translations = np.array([r['translation'][:2] for r in future])

            # We will convert the translations from global to local
            future_translations = convert_global_coords_to_local(
                future_translations, present['translation'], present['rotation'])
            past_translations = convert_global_coords_to_local(
                past_translations, present['translation'], present['rotation'])

            # We will then add the other parameters (v, a, theta)
            # TODO: Create a private method to do this, as we will
            #  be also doing this later for the neighbors
            for side, records, translations in (
                    ('past', past, past_translations),
                    ('future', future, future_translations)):
                # I'm not used to numpy, therefore I'll be using Python
                #  objects (lists) for this, then convert the whole
                #  list into a numpy array at the end.
                coordinates = []

                # We get the translations separately, as requesting the whole
                #  records cause the helper methods to not preprocess the
                #  coordinates.
                for record, (x, y) in zip(records, translations):
                    # I'll be using those two often below
                    token = record['instance_token'], record['sample_token']

                    # We simply use the methods exposed by the helper
                    v = self._helper.get_velocity_for_agent(*token)
                    a = self._helper.get_acceleration_for_agent(*token)
                    theta = self._helper.get_heading_change_rate_for_agent(*token)

                    # Sometimes values cannot be given, so nan is chosen as a replacement
                    _f = (lambda _arg: 0 if np.isnan(_arg) else _arg)
                    coordinates.append((x, y, *map(_f, (v, a, theta))))

                # We don't need any padding, as we have assertions before
                if side == 'past':
                    # We then reverse the array to have the correct order
                    past_translations = np.array(list(reversed(coordinates)))
                else:
                    # Or do nothing if the order is already correct
                    future_translations = np.array(list(coordinates))

            # We use an extended amount of points to find out the previous lanes in which
            #  the cars were recorded.
            # TODO: Decide whether or not remove this, as it would probably be easier to
            #  just request the full past of the instance from the helper.
            target_agent_extended_past_timeline = [i['sample_token'] for i in extended_past]

            all_lanes = set()
            cars_on_lanes = dict()

            # TODO: Make sure the neighbor is actually on the road
            # TODO: Make sure the neighbor is not running against the direction
            # TODO: Instead of using only the current state, see if I can use the
            #  whole scene, for instance by using weighted sums.
            # We take all the vehicles in the scene, find out if they are on a lane, then
            #  add them to their assigned lanes. Later we will select one actor per lane
            #  but for now all of the vehicles are added.
            for neighbor in self._helper.get_annotations_for_sample(present['sample_token']):
                # We should not care about road signs, pedestrians, etc, as we won't be
                #  able to associate a lane to them, and as their behaviour will not match
                #  a car's behaviour.
                # TODO: Is it possible to still be able to take them into account?
                if neighbor["category_name"].split('.')[0] != 'vehicle':
                    continue
                # We take the closest lane and assign the vehicle to it
                lane = nusc_map.get_closest_lane(*neighbor['translation'][:2], IS_ON_LANE)
                # If the vehicle is too far from any lanes, then it does not matter
                if not lane:
                    continue
                all_lanes.add(lane)
                # We store the vehicle's attributes, as well as the projected point, a useful
                #  information used later to determine the lane coordinates.
                cars_on_lanes.setdefault(lane, {})[neighbor['instance_token']] = dict(
                    neighbor=neighbor, projected_point=arcline_path_utils.project_pose_to_lane(
                        pose=neighbor['translation'], lane=nusc_map.get_arcline_path(lane)))

            # Make sure that we can split it in 100
            assert (self._precision_lane * 100).is_integer()

            # We define this to be able to have lanes and neighbors sorted based on their
            #  distances from the target agent at the present moment.
            distance_lanes = dict((i, distance_lane(nusc_map, present, i)) for i in all_lanes)
            # We removed the limit for debugging purposes, we moved the limit at the end, right
            #  before returning the neighbors and lanes
            all_lanes = sorted(all_lanes, key=distance_lanes.get)  # [:N]
            cars_on_lanes = dict((i, j) for i, j in cars_on_lanes.items() if i in all_lanes)
            lanes_coordinates = []

            # This will be used in the network as the Vn inputs
            car_coordinates: List[Any] = [None] * len(all_lanes)
            # This does not seem to be used, however I'll wait before removing it
            # TODO: See if I can remove this safely
            car_projected_transform: List[Any] = [None] * len(all_lanes)
            # Unlike its name might lead to believe, this will only contain the instance_token
            # TODO: Rename variables
            # TODO: Find all Any in variable descriptions and replace with expected types
            car_projected_attributes: List[Any] = [None] * len(all_lanes)

            # Populate the `car_coordinates` list (which will be used in the network) as well as the
            # `car_projected_attributes` (used to determine lanes coordinates).
            for lane, _cars in cars_on_lanes.items():
                # We select the neighbor based on the distance (in the present) between it
                #  and the target agent.
                # TODO: See if there are better ways to do it, for instance using the distance
                #  to the center of the lane, or by using multiple time steps instead of only
                #  the current one.
                full_car = min(_cars.values(), key=(lambda x: distance(present, x['neighbor'])))
                # We define this to hold the attributes
                car = full_car['neighbor']
                # The duration is in seconds, `self._history_duration` is in coordinates,
                #  self._history_duration is equal to tau, therefore:
                #   duration = self._history_duration / 2
                #  However I don't care as I will simply remove any excess later
                duration = self._history_duration
                # I use the present after the past because it should be included in the list,
                #  and I do this now to avoid having to do the work twice, once for the past
                #  and once for the present
                # TODO: Make sure that the current attributes are not present in the list
                _past_records = [car] + self._helper.get_past_for_agent(
                    car['instance_token'], car['sample_token'], duration, False, False)
                _past_records = _past_records[:h_d + 1]

                # We start by defining the x, y values in the correct referential
                _xy = np.array([r['translation'][:2] for r in _past_records])
                _xy = convert_global_coords_to_local(_xy, present['translation'], present['rotation'])

                # Each coordinates will be (x, y, v, a, theta)
                # For more info, see above when we do the same for the target agent.
                _coordinates = []
                for index, (record, (x, y)) in enumerate(zip(_past_records, _xy)):
                    token = record['instance_token'], record['sample_token']

                    v = self._helper.get_velocity_for_agent(*token)
                    a = self._helper.get_acceleration_for_agent(*token)
                    theta = self._helper.get_heading_change_rate_for_agent(*token)
                    _f = (lambda _arg: 0 if np.isnan(_arg) else _arg)
                    _coordinates.append((x, y, *map(_f, (v, a, theta))))

                # We pad by adding extra zeroes then trimming the list
                _coordinates.extend((0, 0, 0, 0, 0) for i in range(h_d + 1))
                # We then reverse the array to have the correct order (past to present)
                _coordinates = np.array(list(reversed(_coordinates[:h_d + 1])))

                # TODO: Replace every instance of -np.inf into 0

                car_coordinates[int(all_lanes.index(lane))] = _coordinates
                car_projected_transform[int(all_lanes.index(lane))] = full_car['projected_point']
                car_projected_attributes[int(all_lanes.index(lane))] = car['instance_token']
                assert car['sample_token'] == present['sample_token'], (
                    car['instance_token'], car['sample_token'])

            def get_full_lanes(past_timeline: List[str], instance_token: str) -> List[np.ndarray]:
                """
                :param past_timeline: A list of sample tokens to be used to retrieve all of the coordinates.
                :param instance_token: The token used to define the agent that we want to follow
                :return:
                """

                # TODO: De indent this
                if True:
                    # The first step will be to retrieve all the attributes of the instance
                    # Note: it does not matter if we get too much points, as this function
                    #  will not be responsible for the neighbors coordinates
                    # TODO: Reduce the length of this function by only calling the helper to
                    #  request the whole past of the instance.
                    try:
                        current_agent_past_attributes = [self._helper.get_sample_annotation(
                            instance_token, sample_token) for sample_token in past_timeline]
                    except KeyError:
                        return []
                    current_agent_current_attributes = current_agent_past_attributes[0]

                    # TODO: Dispose of this
                    """bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, 'basemap')
                    fig, ax = nusc_map.render_layers(['lane'], figsize=1, bitmap=bitmap)
                    for x in current_agent_past_attributes:
                        nusc_map.render_next_roads(*x['translation'][:2], figsize=1, bitmap=bitmap)
                    raise ValueError(sleep(3600))"""

                    def retrieve_associated_lane(current_attributes: dict):
                        return nusc_map.get_closest_lane(*current_attributes['translation'][:2], IS_ON_LANE)

                    # We will start with the past, as we have more data about it, and because we have to use it to have
                    #  coherent data (as we know the locations).
                    past_coordinates, expected_coordinates = [], self._backward_lane * 100
                    expected_coordinates_in_future = self._forward_lane * 100

                    def unique(iterator):
                        already = set()
                        for i in iterator:
                            if i not in already:
                                yield i
                                already.add(i)

                    current_agent_past_lanes = list(
                        unique(map(retrieve_associated_lane, current_agent_past_attributes)))
                    # If the car leaves the road, then we don't want to deal with it.
                    # TODO: Is this necessary?
                    if '' in current_agent_past_lanes:
                        return []
                    # Concatenate all the roads that are unambiguous (if there is only
                    #  one way in, it is most likely the way used by the car)
                    # TODO: Is this still necessary?
                    while True:
                        _tmp = nusc_map.get_incoming_lane_ids(current_agent_past_lanes[-1])
                        if len(_tmp) != 1:
                            break
                        # If the length is 1, it's like adding the element 0
                        current_agent_past_lanes.extend(_tmp)

                    # TODO: Think about removing the unused commented out variables
                    # reference_translation_beforehand = current_agent_past_attributes[1]['translation']
                    # This will be used as the reference point to know where to start in the road
                    reference_translation = current_agent_current_attributes['translation']
                    # See below: reference_yaw =
                    #  quaternion_yaw(Quaternion(current_agent_current_attributes['rotation']))

                    # The first variable is named after its only use in this loop
                    # TODO: Instead of this, what about having the first case outside of the loop?
                    for is_not_the_first_lane, current_agent_past_lane in enumerate(current_agent_past_lanes):
                        # Extract the record, then use it to extract the coordinates.
                        current_agent_past_lane_record = nusc_map.get_arcline_path(current_agent_past_lane)
                        current_agent_past_lane_coordinates = arcline_path_utils.discretize_lane(
                            current_agent_past_lane_record, 0.01)

                        # The next step will be to see the closest point on the lane in the present.
                        # We will consider it to be the location of the car.
                        projected_coordinates, _ = arcline_path_utils.project_pose_to_lane(
                            reference_translation, lane=current_agent_past_lane_record)

                        # TODO: Find in this code all the instance where I used the third coordinate as the yaw
                        # We consider the car to go with the flow
                        # TODO: Is this useful?
                        # assert abs(reference_yaw - projected_coordinates[-1]) < 0.1, (
                        #     reference_yaw, projected_coordinates, current_agent_current_attributes)

                        if is_not_the_first_lane:
                            # I have to make sure we are near the exit of the previous lane.
                            # TODO: I made a typo, but don't want to change it yet by fear of unexpected errors
                            assert np.linalg.norm(
                                np.array(projected_coordinates[:2]) - np.array(projected_coordinates[:2])) < 1, (
                                projected_coordinates, current_agent_past_lane_coordinates[-1])
                            # The coordinates are (I assume) ordered from entrance to exit
                            # TODO: Make sure this is the case.
                            # TODO: Since I will only use the lane tokens, is this really necessary?
                            past_coordinates = current_agent_past_lane_coordinates + past_coordinates
                        else:
                            # This is useful to see what to keep and what to throw away.
                            indexed_coordinates = list(i for i in enumerate(current_agent_past_lane_coordinates))
                            first_point: Tuple[int, Any] = min(indexed_coordinates, key=(lambda arg: np.linalg.norm(
                                np.array(reference_translation[:2]) - np.array(arg[1][:2]))))
                            # second_point: Tuple[int, Any] = min(indexed_coordinates, key=(lambda arg: np.linalg.norm(
                            #     np.array(reference_translation_beforehand[:2]) - np.array(arg[1][:2]))))

                            # We consider it to be true, otherwise we will have to change things in the code
                            # assert first_point[0] > second_point[0], (first_point[0], second_point[0])

                            # We will cut all the points after the current one (we include the current one)
                            past_coordinates = current_agent_past_lane_coordinates[:first_point[0] + 1]
                            expected_coordinates_in_future -= len(current_agent_past_lane_coordinates)
                            expected_coordinates_in_future += first_point[0] + 1

                        reference_translation = past_coordinates[0]
                        # reference_translation_beforehand = past_coordinates[1]
                        # reference_yaw = reference_translation[-1]

                    # We therefore have to reverse the list here to avoid any complications later in the code
                    # We also rename the variable, as in the future the current_agent_past_lanes name will be
                    #  used to describe the list of all the possible past timelines
                    # From this point, all the lanes should be in the correct order (past to future)
                    possible_past_timeline = list(reversed(current_agent_past_lanes))
                    # This is here to make sure we don't accidentally use the old name
                    del current_agent_past_lanes

                    # See if we have enough road in the past to get the amount of coordinates we want.
                    # Note: before this point, the list of lanes is ordered from present to past, as the
                    #  present lane is the first lane, and because we added to the lanes in the normal order
                    if len(past_coordinates) < expected_coordinates:
                        # In that case we will have a list of different possibles histories
                        current_agent_past_lanes = []
                        # For each of them we will look at the first item, which corresponds to the furthest road
                        #  in the past, then we concatenate the resulting possibilities with the itinerary we have.
                        for i in self._get_possibilities(
                                possible_past_timeline[0], expected_coordinates - len(past_coordinates), 'incoming'):
                            full_road = list((*i, *possible_past_timeline))
                            # We make sure the history is consistent.
                            # TODO: Should we reverse the effect of cut_index?
                            #  for instance if we see the past the uncertain data is
                            #  about the last coordinates (by uncertain data, we mean
                            #  the points that we got from the history, as they are coming
                            #  from the closest lanes instead of connecting lanes).
                            if self._verify_possibility(*full_road, cut_index=len(i)):
                                current_agent_past_lanes.append(full_road)
                    else:
                        # In that case this is much easier, as we only have one choice
                        current_agent_past_lanes = [possible_past_timeline]

                # This will hold the list of all the possible itineraries
                final_list = []
                # We consider the past timeline to be a list of possibilities, so
                #  we can have a single bit of code treating every cases.
                for possible_past_timeline in current_agent_past_lanes:
                    # According to the current timeline, we attempt to add roads until we have
                    #  enough coordinates to reconstruct the whole path.
                    # We can do as previous and only use this loop to create a list of roads, then
                    #  later in the function create another loop to populate the list of coordinates.
                    # TODO: Consider the possibility above
                    for i in self._get_possibilities(
                            possible_past_timeline[-1], expected_coordinates_in_future, 'outgoing'):
                        # The first step is to merge the two halves of the lanes
                        # Note: Whether it is the past or the future, the
                        full_road = list((*possible_past_timeline, *i))
                        full_records = list(map(nusc_map.get_arcline_path, full_road))
                        full_coordinates = list(itertools.chain.from_iterable(map(
                            partial(arcline_path_utils.discretize_lane, resolution_meters=0.01), full_records)))
                        full_indexed_coordinates = list(enumerate(full_coordinates))
                        reference_point, _ = min(full_indexed_coordinates, key=(lambda arg: np.linalg.norm(
                            np.array(current_agent_current_attributes['translation'][:2]) - np.array(arg[1][:2]))))

                        # TODO: assert
                        if self._verify_possibility(*full_road, cut_index=len(possible_past_timeline)):
                            # TODO: Make sure we don't get any one-off errors.
                            full_coordinates = full_coordinates[slice(
                                reference_point - int(self._backward_lane * 100),
                                reference_point + int(self._forward_lane * 100),
                                int(self._precision_lane * 100))]
                            full_coordinates = np.array(list(i[:2] for i in full_coordinates))
                            final_list.append(convert_global_coords_to_local(
                                full_coordinates, present['translation'], present['rotation']))

                return final_list

            # TODO: Decide whether or not we should be using the extended timeline
            # raise SystemExit(next(get_full_lanes(target_agent_extended_past_timeline, target_agent_token)))

            for lane_index, lane in enumerate(all_lanes):
                lanes_coordinates.append(
                    list(get_full_lanes(target_agent_extended_past_timeline, car_projected_attributes[lane_index])))

            grand_history.append(np.array(past_translations))
            grand_future.append(np.array(future_translations))
            grand_neighbors.append(car_coordinates)
            grand_lanes.append(lanes_coordinates)

        new_history, new_future, new_lanes, new_neighbors = [], [], [], []
        all_reference_lanes = []

        for history_row, future_row, lanes_row, neighbors_row in zip(
                grand_history, grand_future, grand_lanes, grand_neighbors):

            # The first step is to remove all neighbors couples that will not have any path
            # We do this that way to keep a consistent order
            total_possibilities = list(map(len, lanes_row))
            impossibles = [i for i, j in enumerate(total_possibilities) if not j]
            lanes_row = [j for i, j in enumerate(lanes_row) if i not in impossibles]
            neighbors_row = [j for i, j in enumerate(neighbors_row) if i not in impossibles]

            # Then we will have to separate the possibilities, some are trivial, some are not
            total_possibilities = list(map(len, lanes_row))
            trivial_instances = [i for i, j in enumerate(total_possibilities) if j == 1]
            multiples_indices = [i for i, j in enumerate(total_possibilities) if j > 1]
            multiples = [list(range(j)) for i, j in enumerate(total_possibilities) if j > 1]

            # I wrote this code so I have to do more checks to make sure if it works th way I want
            # TODO: What I just said
            possibilities = [[]]
            for after in multiples:
                _new_possibilities = []
                for j in after:
                    for possibility in possibilities:
                        _new_possibilities.append((*possibility, j))
                possibilities = _new_possibilities

            # TODO: Is it good to have zeroes in the inputs?

            if self._n > len(neighbors_row):
                # Dimension: N, n_coordinates, 2
                neighbors_row = neighbors_row + [np.zeros((h_d + 1, 5)) for _ in range(self._n - len(neighbors_row))]

            print('%s possibilities' % len(possibilities))
            for possibility in possibilities:
                lanes_coordinates: List[Any] = [None] * len(lanes_row)
                # The trivial ones only have one item
                for a in trivial_instances:
                    lanes_coordinates[a], = lanes_row[a]
                # The other ones will have multiples instances
                for a, b in zip(multiples_indices, possibility):
                    lanes_coordinates[a] = lanes_row[a][b]
                try:
                    lanes_coordinates.index(None)
                except ValueError:
                    pass
                else:
                    raise RuntimeError

                if self._n > len(lanes_coordinates):
                    # Dimension: N, n_coordinates, 2
                    lanes_coordinates = lanes_coordinates + [
                        np.zeros((self._lane_coordinates, 2)) for _ in range(self._n - len(lanes_coordinates))]
                new_history.append(history_row)
                new_future.append(future_row)
                # We put back the limit in case we got more than N
                new_lanes.append(np.array(lanes_coordinates[:self._n]))
                # print(neighbors_row)
                new_neighbors.append(np.array(neighbors_row[:self._n]))

                distances = []
                for l_n in new_lanes[-1]:
                    nu = (lambda x: x)
                    distances.append(sum(min(np.linalg.norm(
                        np.array([v_i[:2], l_m[:2]]),
                    ) for l_m in l_n) * nu(i) for i, v_i in enumerate(future_row, 1)))
                # TODO: Use this
                reference_lane = min(range(len(distances)), key=distances.__getitem__)
                all_reference_lanes.append(lanes_coordinates[reference_lane])

                # _, distance_index = min((j, i) for i, j in enumerate(distances))

                # TODO: Dispose of this or create a method
                def void():
                    import matplotlib.pyplot
                    __x = [j[0] for j in history_row]
                    __y = [j[1] for j in history_row]
                    matplotlib.pyplot.plot(list(__x), list(__y), label='Target Agent')
                    for index, i in enumerate(neighbors_row):
                        __x = [j[0] for j in i]
                        __y = [j[1] for j in i]
                        if -np.inf not in __x:
                            matplotlib.pyplot.plot(list(__x), list(__y), label='Neighbor %d' % index)
                    for index, i in enumerate(lanes_coordinates):
                        __x = [j[0] for j in i]
                        __y = [j[1] for j in i]
                        if -np.inf not in __x:
                            matplotlib.pyplot.plot(list(__x), list(__y), label='Lane %d' % index)

                    matplotlib.pyplot.legend()
                    matplotlib.pyplot.show()

                # TODO: Dispose of this or create a method
                """
                    matplotlib.pyplot.plot(list(__x), list(__y))
                __x = [j['translation'][0] for j in current_agent_past_attributes]
                __y = [j['translation'][1] for j in current_agent_past_attributes]
                matplotlib.pyplot.plot(list(__x), list(__y))
                matplotlib.pyplot.show()
                return False"""

        # V^(p), V^(f), L^n, V^n, L_ref
        return new_history, new_future, new_lanes, new_neighbors, list(
            map(np.array, all_reference_lanes))

    def __getitem__(self, item):
        # TODO: Dispose of this
        # print
        ('Item', item, len(self._history[item]), len(self._future[item]), len(self._lanes[item]),
         len(self._neighbors[item]), (self._history[item]).shape, (self._future[item]).shape,
         (self._lanes[item]).shape,
         (self._neighbors[item]), self._reference_lanes[item])
        return \
            self._history[item], self._future[item], \
            self._lanes[item], self._neighbors[item], \
            self._reference_lanes[item]

    def _get_possibilities(self, starting_lane: str, remaining_size: int, side='outgoing', *previous_lanes):
        # TODO: See if we can do something about the case in which we get not roads
        # TODO: Tests and documentation
        for possible_lane in getattr(self._nusc_map, 'get_%s_lane_ids' % side)(starting_lane):
            possible_record = self._nusc_map.get_arcline_path(possible_lane)
            possible_length = math.floor(length_of_lane(possible_record) * 100)
            size = remaining_size - possible_length
            if size > 0:
                yield from self._get_possibilities(possible_lane, size, side, *previous_lanes, possible_lane)
            else:
                if side == 'incoming':
                    yield tuple((possible_lane, *previous_lanes))
                else:
                    yield tuple((*previous_lanes, possible_lane))

    def _verify_possibility(self, *lanes, cut_index):
        # TODO: Tests, documentation and refactoring
        import matplotlib.pyplot
        lanes = list(map(self._nusc_map.get_arcline_path, lanes))
        lanes = list(map(partial(arcline_path_utils.discretize_lane, resolution_meters=0.01), lanes))
        # full_lane = list(itertools.chain.from_iterable(lanes))

        for loop in range(len(lanes) - 1):
            a, b = lanes[loop][-1], lanes[loop + 1][0]
            c, d = lanes[loop][0], lanes[loop + 1][-1]
            # print(a, len(lanes))
            # print(b, len(lanes))
            try:
                # assert abs(a[-1] - b[-1]) < 0.1, ()
                assert np.linalg.norm(np.array(a[:2]) - np.array(b[:2])) < 1, (a, b, c, d)
            except AssertionError:
                # If there is a discontinuity between past and future, we drop it
                if loop < cut_index:
                    return False

                def cones(x_list, y_list, radius=10, precision=100):
                    x_last = x_list[-1]
                    y_last = y_list[-1]
                    for i in range(precision):
                        angle = 2 * np.pi * i / precision
                        x_offset = x_last + (radius * np.cos(angle))
                        y_offset = y_last + (radius * np.sin(angle))
                        x_list.append(x_offset)
                        y_list.append(y_offset)
                    return x_list, y_list

                """for i in lanes:
                    __x = [j[0] for j in i]
                    __y = [j[1] for j in i]
                    import matplotlib.pyplot
                    matplotlib.pyplot.plot(list(__x), list(__y))"""
                import matplotlib.pyplot
                for index, item in enumerate(lanes):
                    matplotlib.pyplot.plot(*cones(
                        [j[0] for j in item],
                        [j[1] for j in item]), label=repr(index))
                '''matplotlib.pyplot.plot(*(
                    [j[0] for j in full_lane],
                    [j[1] for j in full_lane]), label='Full')'''
                matplotlib.pyplot.legend()

                """__x = [j[0] for j in lanes[loop + 1]]
                __y = [j[1] for j in lanes[loop + 1]]
                import matplotlib.pyplot
                matplotlib.pyplot.plot(list(__x), list(__y))"""
                """__x = [j['translation'][0] for j in current_agent_past_attributes]
                __y = [j['translation'][1] for j in current_agent_past_attributes]"""
                # matplotlib.pyplot.plot(list(__x), list(__y))

                """__x = [j[0] for j in lanes[loop + 1]]
                __y = [j[1] for j in lanes[loop + 1]]
                import matplotlib.pyplot
                matplotlib.pyplot.plot(list(__x), list(__y))"""
                """__x = [j['translation'][0] for j in current_agent_past_attributes]
                __y = [j['translation'][1] for j in current_agent_past_attributes]"""
                # matplotlib.pyplot.plot(list(__x), list(__y))

                matplotlib.pyplot.show()
                raise
        """# __x = itertools.chain.from_iterable([(j[0] for j in i) for i in lanes])
        # __y = itertools.chain.from_iterable([(j[1] for j in i) for i in lanes])
        for i in lanes:
            __x = [j[0] for j in i]
            __y = [j[1] for j in i]
            import matplotlib.pyplot
            matplotlib.pyplot.plot(list(__x), list(__y))
        __x = [j['translation'][0] for j in current_agent_past_attributes]
        __y = [j['translation'][1] for j in current_agent_past_attributes]
        matplotlib.pyplot.plot(list(__x), list(__y))
        matplotlib.pyplot.show()
        # sleep(3600)"""
        return True


if __name__ == '__main__':
    for ns in NS.every_map('config.yml'):
        pass
