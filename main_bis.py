import numpy as np
import yaml
from nuscenes.map_expansion.map_api import NuScenesMap

from nuscenes.nuscenes import NuScenes

from matplotlib import pyplot as plt

config_file = 'config.yml'
with open(config_file, 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)['ns_args']

tau = config['history_duration']
h = config['prediction_duration']

nusc = NuScenes(version=config['version'], dataroot=config['data_root'], verbose=True)


def show_trajectory(*coordinates):
    """

    :param coordinates: coordinates[i]: [nb_coord, 2]
    :return:
    """
    import matplotlib.pyplot as plt
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
    instance_tokens.append(sample['anns'])
    timestamps_sample.append(sample['timestamp'])
    while not sample['next'] == "":
        sample = nusc.get('sample', sample['next'])
        sample_tokens.append(sample['token'])
        instance_tokens.append(sample['anns'])
        timestamps_sample.append(sample['timestamp'])
    return timestamps_sample, sample_tokens, instance_tokens


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
    return dict((i, _discredized_lanes[i]) for i in lanes)


# nusc.render_sample(st)
def iter_lanes(nusc_map, x, y, radius):
    # TODO: Set the radius here to be able to increase it at will
    """Iterate through lanes based on a reference point, a radius,
        and sorted according to its distance to the reference."""
    lanes = nusc_map.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])
    lanes = lanes['lane'] + lanes['lane_connector']
    discrete = discretize_lanes(nusc_map, lanes)
    lanes_distances = dict()

    for lane_id, points in discrete.items():
        lanes_distances[lane_id] = np.linalg.norm(np.array(points)[:, :2] - [x, y], axis=1).min()

    yield from sorted(lanes, key=lanes_distances.get)


def cut_iterator(iterator, n):
    for i, v in enumerate(iterator, 2):
        yield v
        if i > n:
            return
    raise RuntimeError('Not enough data (excepted %d, got less)' % n)


def elongate_lanes(nusc_map, lanes_iterator):
    already_found = set()
    for lane in lanes_iterator:
        # If the lane was found,
        #  then we don't use it again
        if lane in already_found:
            continue

        yield lane
        already_found.add(lane)

        # raise ValueError(lane)


def preprocess_scene(scene):
    sorted_ego = sorted(nusc.ego_pose, key=(lambda i: i['timestamp']))
    log = nusc.get('log', scene['log_token'])
    map_name = log['location']

    nusc_map = NuScenesMap(map_name=map_name, dataroot=config['data_root'])
    timestamps_samples, sample_tokens, instance_tokens = unpack_scene(scene)
    # EGO VEHICLE
    ego = [i for i in sorted_ego if i['timestamp'] in timestamps_samples]
    assert len(ego) == len(timestamps_samples)
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

    def void():

        raise ValueError(lanes)

        lanes = nusc_map.get_closest_lane(*history_ego_coords[tau - 1], 10)
        print(lanes)
        # REFERENCE LANE :
        nu = (lambda j: j + 1)
        d = (lambda l: sum(min(np.linalg.norm(c - m) for m in l) * nu(i) for i, c in enumerate(future_ego_coords)))

        raise SystemExit


def main():
    timestamps = sorted([pose['timestamp'] * 1e-6 for pose in nusc.ego_pose])
    differences = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    # print(differences)
    """for scene in nusc.scene:
        print(scene)"""

    preprocessed_data = [[] for i in range(5)]
    for scene in nusc.scene:
        for index, arg in enumerate(preprocess_scene(scene)):
            preprocessed_data[index].append(arg)


if __name__ == '__main__':
    main()
