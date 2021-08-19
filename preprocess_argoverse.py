import math
from os.path import expanduser
from random import randrange
from typing import Tuple

from matplotlib import pyplot as plt
from tqdm import tqdm

from argoverse_api.argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

from argoverse_api.argoverse.map_representation.map_api import ArgoverseMap
# from argoverse_forecasting.utils.map_features_utils import MapFeaturesUtils

from csv import reader as csv_reader
import numpy as np
import pandas as pd

show = True

if not show:
    from argoverse_forecasting.utils.map_features_utils import MapFeaturesUtils

    am = ArgoverseMap()
else:
    MapFeaturesUtils = None
    am = None

##set root_dir to the correct path to your dataset folder
# root_dir = 'argoverse_api/forecasting_sample/data/'
root_dir = expanduser('~/argoverse-forecasting/venv/forecasting_train_v1.1/train/data/')

N = 6
HISTORY_SIZE = 20
PREDICTION_SIZE = 30
FRONT_OR_BACK_OFFSET_THRESHOLD = 5.0
NEARBY_DISTANCE_THRESHOLD = 50.0  # Distance threshold to call a track as neighbor
DEFAULT_MIN_DIST_FRONT_AND_BACK = 100.0

RAW_DATA_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
}


def read_csv_into_dict(location):
    converted_data = dict()
    csv_file = csv_reader(open(location))
    headers = next(csv_file)
    for row in csv_file:
        row_dict = dict(zip(headers, row))
        converted_data.setdefault(
            row_dict['TIMESTAMP'], {}
        )[row_dict['TRACK_ID']] = row_dict
    return converted_data


def get_full_coordinates(data, track_ids):
    """

    :param data:
    :param track_ids:
    :return: List of dimensions [nb_timestamps x nb_items x 2]
    """
    full_coordinates = []
    for key, value in sorted(data.items()):
        these_coordinates = []
        for current_track_id in track_ids:
            if current_track_id in value:
                val = value[current_track_id]
                these_coordinates.append([
                    float(val['X']),
                    float(val['Y'])])
            else:
                these_coordinates.append(None)
        full_coordinates.append(these_coordinates)
    return full_coordinates


def get_presence_and_distance(coordinates, tau=HISTORY_SIZE):
    presence = [0 for i in range(len(coordinates[0]))]
    for i in coordinates[:tau]:
        for index, j in enumerate(i):
            presence[index] += j is not None

    dst = (lambda a, b: math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) if b is not None else math.inf)
    distances = [0.0] + [dst(coordinates[tau - 1][0], i) for i in coordinates[tau - 1][1:]]
    return presence, distances


def pad_coordinates(coordinates):
    out_coordinates = np.transpose(np.array(coordinates))
    for row in out_coordinates:
        size_begin = next(index for index, value in enumerate(row) if value is not None)
        for i in range(size_begin):
            row[i] = row[size_begin]
        if None in row:
            size_end = next(index for index, value in enumerate(row) if value is None)
            for i in range(size_end, len(row)):
                row[i] = row[size_end - 1]
    return np.transpose(np.array(out_coordinates))


def filter_nearest_neighbors(coordinates, distances, stationary, n=N, tau=HISTORY_SIZE):
    # - 1e6 * presence[x[0]]
    indexed_rows = sorted(list(enumerate(distances)), key=(lambda x: x[1] + 1000 * stationary[x[0]]))
    indices = list(list(zip(*indexed_rows))[0][:n + 1])
    out_coordinates = []
    for index, time_step in enumerate(coordinates[:tau]):
        out_coordinates.append([time_step[i] for i in indices])
    return indices, list(map(list, out_coordinates))


def get_velocities(coordinates, timesteps):
    timesteps = sorted(timesteps, key=float)
    coordinates = np.array(coordinates)
    if len(coordinates.shape) > 2:
        raise RuntimeError(coordinates.shape)
    tau, n = coordinates.shape
    output = np.zeros((tau - 1, n))
    for n_index, neighbor in enumerate(np.transpose(coordinates)):
        for index in range(neighbor.shape[0] - 1):
            t_a = float(timesteps[index])
            t_b = float(timesteps[index + 1])
            x_a, y_a = neighbor[index]
            x_b, y_b = neighbor[index + 1]
            v_x = (x_b - x_a) / (t_b - t_a)
            v_y = (y_b - y_a) / (t_b - t_a)
            v = np.sqrt(v_x ** 2 + v_y ** 2)
            output[index, n_index] = v
    return output


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


def convert_global_coords_to_local(coordinates: list,
                                   translation: Tuple[float, float],
                                   transform) -> np.ndarray:
    coords = (np.array(coordinates) - np.atleast_2d(np.array(translation))).T
    return np.dot(transform, coords).T[:, :2]


def get_rot(first, second):
    """
    Retrieve the rot from two points
    :param first: A tuple of x, y, defined one step before first
    :param second: Another tuple of x, y, defined one step after step.
    :return:
    """
    pre = np.array(second) - np.array(first)
    theta = np.pi - np.arctan2(pre[1], pre[0])
    return np.asarray([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]], np.float32)


def compute_velocity(track_df):
    x_coord = track_df["X"].values

    y_coord = track_df["Y"].values
    timestamp = track_df["TIMESTAMP"].values
    vel_x, vel_y = zip(*[(
        x_coord[i] - x_coord[i - 1] /
        (float(timestamp[i]) - float(timestamp[i - 1])),
        y_coord[i] - y_coord[i - 1] /
        (float(timestamp[i]) - float(timestamp[i - 1])),
    ) for i in range(1, len(timestamp))])
    vel = [np.sqrt(x ** 2 + y ** 2) for x, y in zip(vel_x, vel_y)]
    return vel


STATIONARY_THRESHOLD = 13
VELOCITY_THRESHOLD = 1.0
EXIST_THRESHOLD = (
    15
)


def get_is_track_stationary(track_df) -> bool:
    """Check if the track is stationary.

    Args:
        track_df (pandas Dataframe): Data for the track
    Return:
        _ (bool): True if track is stationary, else False

    """
    vel = compute_velocity(track_df)
    # print("vel : ", vel)
    sorted_vel = sorted(vel)
    threshold_vel = sorted_vel[STATIONARY_THRESHOLD]
    return True if threshold_vel < VELOCITY_THRESHOLD else False


def fill_track_lost_in_middle(
        track_array: np.ndarray,
        seq_timestamps: np.ndarray,
        raw_data_format,
) -> np.ndarray:
    """Handle the case where the object exited and then entered the frame but still retains the same track id. It'll be a rare case.

        Args:
            track_array (numpy array): Padded data for the track
            seq_timestamps (numpy array): All timestamps in the sequence
            raw_data_format (Dict): Format of the sequence
        Returns:
            filled_track (numpy array): Track data filled with missing timestamps

    """
    curr_idx = 0
    filled_track = np.empty((0, track_array.shape[1]))
    for timestamp in seq_timestamps:
        filled_track = np.vstack((filled_track, track_array[curr_idx]))
        if timestamp in track_array[:, raw_data_format["TIMESTAMP"]]:
            curr_idx += 1
    return filled_track


def pad_track(
        track_df,
        seq_timestamps: np.ndarray,
        obs_len: int,
        raw_data_format,
) -> np.ndarray:
    """Pad incomplete tracks.

        Args:
            track_df (Dataframe): Dataframe for the track
            seq_timestamps (numpy array): All timestamps in the sequence
            obs_len (int): Length of observed trajectory
            raw_data_format (Dict): Format of the sequence
        Returns:
                padded_track_array (numpy array): Track data padded in front and back

    """
    track_vals = track_df.values
    track_timestamps = track_df["TIMESTAMP"].values

    # start and index of the track in the sequence
    start_idx = np.where(seq_timestamps == track_timestamps[0])[0][0]
    end_idx = np.where(seq_timestamps == track_timestamps[-1])[0][0]

    # Edge padding in front and rear, i.e., repeat the first and last coordinates

    padded_track_array = np.pad(track_vals,
                                ((start_idx, obs_len - end_idx - 1),
                                 (0, 0)), "edge")
    if padded_track_array.shape[0] < obs_len:
        padded_track_array = fill_track_lost_in_middle(
            padded_track_array, seq_timestamps, raw_data_format)

    # Overwrite the timestamps in padded part
    for i in range(padded_track_array.shape[0]):
        padded_track_array[i, 0] = seq_timestamps[i]
    return padded_track_array


def filter_tracks(seq_df, obs_len: int,
                  raw_data_format) -> np.ndarray:
    """Pad tracks which don't last throughout the sequence. Also, filter out non-relevant tracks.

    Args:
            seq_df (pandas Dataframe): Dataframe containing all the tracks in the sequence
            obs_len (int): Length of observed trajectory
            raw_data_format (Dict): Format of the sequence
        Returns:
            social_tracks (numpy array): Array of relevant tracks

    """
    social_tracks = np.empty((0, obs_len, len(raw_data_format)))

    # Timestamps in the sequence
    seq_timestamps = np.unique(seq_df["TIMESTAMP"].values)

    # Track groups
    df_groups = seq_df.groupby("TRACK_ID")
    for group_name, group_data in df_groups:

        # Check if the track is long enough
        if len(group_data) < EXIST_THRESHOLD:
            continue

        # Skip if agent track
        if group_data["OBJECT_TYPE"].iloc[0] == "AGENT":
            continue

        # Check if the track is stationary
        if get_is_track_stationary(group_data):
            continue

        padded_track_array = pad_track(group_data, seq_timestamps,
                                       obs_len,
                                       raw_data_format).reshape(
            (1, obs_len, -1))
        social_tracks = np.vstack((social_tracks, padded_track_array))

    return social_tracks


def get_is_front_or_back(
        track: np.ndarray,
        neigh_x: float,
        neigh_y: float,
        raw_data_format):
    """Check if the neighbor is in front or back of the track.

        Args:
            track (numpy array): Track data
            neigh_x (float): Neighbor x coordinate
            neigh_y (float): Neighbor y coordinate
        Returns:
            _ (str): 'front' if in front, 'back' if in back

    """
    # We don't have heading information. So we need at least 2 coordinates to determine that.
    # Here, front and back is determined wrt to last 2 coordinates of the track
    x2 = track[-1, raw_data_format["X"]]
    y2 = track[-1, raw_data_format["Y"]]
    # Keep taking previous coordinate until first distinct coordinate is found.
    idx1 = track.shape[0] - 2
    while idx1 > -1:
        x1 = track[idx1, raw_data_format["X"]]
        y1 = track[idx1, raw_data_format["Y"]]
        if x1 != x2 or y1 != y2:
            break
        idx1 -= 1

    # If all the coordinates in the track are the same, there's no way to find front/back
    if idx1 < 0:
        return None

    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    p3 = np.array([neigh_x, neigh_y])
    proj_dist = np.abs(np.cross(p2 - p1,
                                p1 - p3)) / np.linalg.norm(p2 - p1)

    # Interested in only those neighbors who are not far away from the direction of travel
    if proj_dist < FRONT_OR_BACK_OFFSET_THRESHOLD:

        dist_from_end_of_track = np.sqrt(
            (track[-1, raw_data_format["X"]] - neigh_x) ** 2 +
            (track[-1, raw_data_format["Y"]] - neigh_y) ** 2)
        dist_from_start_of_track = np.sqrt(
            (track[0, raw_data_format["X"]] - neigh_x) ** 2 +
            (track[0, raw_data_format["Y"]] - neigh_y) ** 2)
        dist_start_end = np.sqrt((track[-1, raw_data_format["X"]] -
                                  track[0, raw_data_format["X"]]) ** 2 +
                                 (track[-1, raw_data_format["Y"]] -
                                  track[0, raw_data_format["Y"]]) ** 2)

        return ("front"
                if dist_from_end_of_track < dist_from_start_of_track
                   and dist_from_start_of_track > dist_start_end else "back")

    else:
        return None


def get_min_distance_front_and_back(
        agent_track: np.ndarray,
        social_tracks: np.ndarray,
        obs_len: int,
        raw_data_format,
        viz=True,
) -> np.ndarray:
    """Get minimum distance of the tracks in front and in back.

        Args:
            agent_track (numpy array): Data for the agent track
            social_tracks (numpy array): Array of relevant tracks
            obs_len (int): Length of the observed trajectory
            raw_data_format (Dict): Format of the sequence
            viz (bool): Visualize tracks
        Returns:
            min_distance_front_and_back (numpy array): obs_len x 2, minimum front and back distances

    """
    min_distance_front_and_back = np.full(
        (obs_len, 2), DEFAULT_MIN_DIST_FRONT_AND_BACK)

    # Compute distances for each timestep in the sequence
    for i in range(obs_len):

        # Agent coordinates
        agent_x, agent_y = (
            agent_track[i, raw_data_format["X"]],
            agent_track[i, raw_data_format["Y"]],
        )

        # Compute distances for all the social tracks
        for social_track in social_tracks[:, i, :]:

            neigh_x = social_track[raw_data_format["X"]]
            neigh_y = social_track[raw_data_format["Y"]]
            if viz:
                plt.scatter(neigh_x, neigh_y, color="green")

            # Distance between agent and social
            instant_distance = np.sqrt((agent_x - neigh_x) ** 2 +
                                       (agent_y - neigh_y) ** 2)

            # If not a neighbor, continue
            if instant_distance > NEARBY_DISTANCE_THRESHOLD:
                continue

            # Check if the social track is in front or back
            is_front_or_back = get_is_front_or_back(
                agent_track[:2, :] if i == 0 else agent_track[:i + 1, :],
                neigh_x,
                neigh_y,
                raw_data_format,
            )
            if is_front_or_back == "front":
                min_distance_front_and_back[i, 0] = min(
                    min_distance_front_and_back[i, 0], instant_distance)

            elif is_front_or_back == "back":
                min_distance_front_and_back[i, 1] = min(
                    min_distance_front_and_back[i, 1], instant_distance)

        if viz:
            plt.scatter(agent_x, agent_y, color="red")
            plt.text(
                agent_track[i, raw_data_format["X"]],
                agent_track[i, raw_data_format["Y"]],
                "{0:.1f}".format(min_distance_front_and_back[i, 0]),
                fontsize=5,
            )

    if viz:
        plt.text(
            agent_track[0, raw_data_format["X"]],
            agent_track[0, raw_data_format["Y"]],
            "s",
            fontsize=12,
        )
        plt.text(
            agent_track[-1, raw_data_format["X"]],
            agent_track[-1, raw_data_format["Y"]],
            "e",
            fontsize=12,
        )
        plt.axis("equal")
        plt.show()
    return min_distance_front_and_back


def get_num_neighbors(agent_track: np.ndarray,
                      social_tracks: np.ndarray,
                      obs_len: int,
                      raw_data_format,
                      ) -> np.ndarray:
    """Get minimum distance of the tracks in front and back.

        Args:
            agent_track (numpy array): Data for the agent track
            social_tracks (numpy array): Array of relevant tracks
            obs_len (int): Length of observed trajectory
            raw_data_format (Dict): Format of the sequence
        Returns:
            num_neighbors (numpy array): Number of neighbors at each timestep

    """
    num_neighbors = np.full((obs_len, 1), 0)

    for i in range(obs_len):

        agent_x, agent_y = (
            agent_track[i, raw_data_format["X"]],
            agent_track[i, raw_data_format["Y"]],
        )

        for social_track in social_tracks[:, i, :]:

            neigh_x = social_track[raw_data_format["X"]]
            neigh_y = social_track[raw_data_format["Y"]]

            instant_distance = np.sqrt((agent_x - neigh_x) ** 2 +
                                       (agent_y - neigh_y) ** 2)

            if instant_distance < NEARBY_DISTANCE_THRESHOLD:
                num_neighbors[i, 0] += 1

    return num_neighbors


def compute_social_features(df,
                            agent_track: np.ndarray,
                            obs_len: int,
                            seq_len: int,
                            raw_data_format,
                            ) -> np.ndarray:
    """Compute social features for the given sequence.

        Social features are meant to capture social context.
        Here we use minimum distance to the vehicle in front, to the vehicle in back,
        and number of neighbors as social features.

        Args:
            df (pandas Dataframe): Dataframe containing all the tracks in the sequence
            agent_track (numpy array): Data for the agent track
            obs_len (int): Length of observed trajectory
            seq_len (int): Length of the sequence
            raw_data_format (Dict): Format of the sequence
        Returns:
            social_features (numpy array): Social features for the agent track

    """
    agent_ts = np.sort(np.unique(df["TIMESTAMP"].values))

    if agent_ts.shape[0] == obs_len:
        df_obs = df
        agent_track_obs = agent_track

    else:
        # Get obs dataframe and agent track
        df_obs = df[df["TIMESTAMP"] < agent_ts[obs_len]]
        assert (np.unique(df_obs["TIMESTAMP"].values).shape[0] == obs_len
                ), "Obs len mismatch"
        agent_track_obs = agent_track[:obs_len]

    # Filter out non-relevant tracks
    social_tracks_obs = filter_tracks(df_obs, obs_len,
                                      raw_data_format)

    # Get minimum following distance in front and back
    min_distance_front_and_back_obs = get_min_distance_front_and_back(
        agent_track_obs,
        social_tracks_obs,
        obs_len,
        raw_data_format,
        viz=True)

    # Get number of neighbors
    num_neighbors_obs = get_num_neighbors(agent_track_obs,
                                          social_tracks_obs, obs_len,
                                          raw_data_format)

    # Agent track with social features
    social_features_obs = np.concatenate(
        (min_distance_front_and_back_obs, num_neighbors_obs), axis=1)

    social_features = np.full((seq_len, social_features_obs.shape[1]), None)
    social_features[:obs_len] = social_features_obs

    return social_features


pass


def cut_lanes(lane, target_track):
    """

    :param lane:
    :param target_track: Target track: the first point is in the past, the
        last point is in the present, and is the reference point.
    :return:
    """
    lane_before = 30
    lane_after = 50
    final_precision = 1  # meter
    multiply_by = 100
    divide_by = 0.01
    # First step: segments lengths, then total length
    segments_lengths = []
    for index in range(len(lane) - 1):
        segments_lengths.append(np.linalg.norm(lane[index + 1] - lane[index]))
    total_length = sum(segments_lengths)

    # print(segments_lengths)

    # Second step: equally space the points
    final_lane = list()  # np.zeros((int(multiply_by * total_length), 2))

    _reference_point = lane[0]

    for index in range(1, len(lane)):
        coordinate = lane[index]
        length = np.linalg.norm(coordinate - _reference_point)
        nb_coords = int(length * multiply_by)
        for i in range(nb_coords):
            alpha = i / nb_coords
            final_lane.append(alpha * coordinate + (1 - alpha) * _reference_point)
        _reference_point = coordinate

    position = len(final_lane)
    final_lane = np.array(final_lane[::int(final_precision * multiply_by)])
    ref_ref = min(range(len(final_lane)), key=(lambda x: np.linalg.norm(final_lane[x] - target_track[-1])))
    ref_before = min(range(len(final_lane)), key=(lambda x: np.linalg.norm(final_lane[x] - target_track[0])))
    # TODO: Make sure the lane is in the right direction
    # show_trajectory(final_lane, target_track)
    if ref_ref < ref_before:
        raise NotEnoughPoints('Wrong direction for the car', ref_ref, ref_before)
    if ref_ref < lane_before:
        raise NotEnoughPoints('Not enough points backward', ref_ref, lane_before)
    if ref_ref + lane_after > len(final_lane) - 1:
        raise NotEnoughPoints('Not enough points forward', ref_ref, lane_after, len(final_lane))

    return final_lane[ref_ref - lane_before:ref_ref + lane_after]


class NotEnoughPoints(Exception):
    pass


from sqlitedict import SqliteDict

cache = SqliteDict('/var/tmp/cache.db')


def get_items():
    # TODO: Check dimensions
    def generate():
        for item in cache.values():
            """dict(
                history=local_history,
                future=local_future,
                neighbors=local_neighbors,
                lanes=local_lanes,
                neighbors_indices=neighbors_indices,
                distances=distances,
                stationary=stationary,
                presences=presences,
                reference_lane_index=m,
                reference_lane_coordinates=n,
                local_lane)"""
            history = item['history']
            future = item['future']
            neighbors = item['neighbors']
            lanes = item['lanes']
            reference_lane = item['reference_lane_index']
            # TODO: Find them
            translation = np.array(history[-1])
            transform = get_rot(*history[-2:])
            yield history, future, neighbors, lanes, np.array([reference_lane]), translation, transform

            '''nu = (lambda j: j + 1)
            d = (lambda l: sum(min(np.linalg.norm(c - m) for m in l) * nu(i) for i, c in enumerate(item['future'])))
            m, n = min(enumerate(item['lanes']), key=(lambda k: d(k[1])))
            print([(i, d(j)) for i, j in sorted(enumerate(item['lanes']), key=(lambda k: d(k[1])))])  # m, n
            plt.plot(*zip(*item['history']))
            plt.plot(*zip(*item['future']))
            for lane in item['lanes']:
                print(lane[0].shape)
                plt.plot(*zip(*lane[0]))
            for neighbor in item['neighbors']:
                neighbor = [i[0] for i in neighbor]
                plt.plot(*zip(*neighbor))
            plt.plot(*zip(*item['lanes'][m][0]), '.-r')
            # exit()
            plt.show()'''

    return list(generate())


def main():
    vels_neighbors = []
    if ArgoverseForecastingLoader is not None:
        afl = ArgoverseForecastingLoader(root_dir)
        fu = MapFeaturesUtils()
    else:
        afl = range(205942)
    #
    for afl_index in tqdm(range(len(afl))):
        # afl_index = randrange(len(afl))
        if afl_index in cache:
            if afl_index in cache:
                """dict(
                    history=local_history,
                    future=local_future,
                    neighbors=local_neighbors,
                    lanes=local_lanes,
                    neighbors_indices=neighbors_indices,
                    distances=distances,
                    stationary=stationary,
                    presences=presences,
                    reference_lane_index=m,
                    reference_lane_coordinates=n,
                    local_lane)"""
                item = cache[afl_index]
                plt.plot(*zip(*item['history']))
                plt.plot(*zip(*item['future']))
                plt.show()

            continue
        try:
            for item in [afl[afl_index]]:
                track_id, = list(set(item.seq_df[item.seq_df["OBJECT_TYPE"] == "AGENT"]['TRACK_ID']))
                others = sorted(list(set(item.seq_df[item.seq_df["OBJECT_TYPE"] != "AGENT"]['TRACK_ID'])))
                neighbors_df = item.seq_df[item.seq_df["OBJECT_TYPE"] != "AGENT"]
                all_track_ids = [track_id] + others

                # First step, read the file
                data = read_csv_into_dict(item.current_seq)
                full_coordinates = get_full_coordinates(data, all_track_ids)
                presences, distances = get_presence_and_distance(full_coordinates)

                def is_stationary(list_of_velocities):
                    sorted_vel = sorted(list_of_velocities)
                    threshold_vel = sorted_vel[STATIONARY_THRESHOLD]
                    return threshold_vel < VELOCITY_THRESHOLD

                full_coordinates = pad_coordinates(full_coordinates)
                stationary = list(map(is_stationary, np.transpose(get_velocities(full_coordinates, sorted(data)))))
                neighbors_indices, neighbors = filter_nearest_neighbors(
                    full_coordinates, distances, stationary,
                    tau=HISTORY_SIZE + PREDICTION_SIZE)

                _ = neighbors_indices.pop(0)
                try:
                    assert not _, (_, neighbors_indices)
                except AssertionError:
                    raise NotEnoughPoints('Unknown error')
                target_history = [i.pop(0) for i in neighbors][:HISTORY_SIZE]
                target_future = [i[0] for i in full_coordinates][20:]
                assert len(target_future) == PREDICTION_SIZE, len(target_future)

                rot = get_rot(*target_history[-2:])

                # df = pd.read_csv(item.current_seq, dtype={"TIMESTAMP": str})
                # agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values
                # Social features are computed using only the observed trajectory
                # social_features = compute_social_features(
                #     df, agent_track, HISTORY_SIZE, HISTORY_SIZE + PREDICTION_SIZE, RAW_DATA_FORMAT)
                # Track groups

                # show_trajectory(target_history, target_future, )  # list(zip(x, y)), )
                local_history = convert_global_coords_to_local(target_history, target_history[-1], rot)
                local_future = convert_global_coords_to_local(target_future, target_history[-1], rot)
                global_n_plus = np.full((N, HISTORY_SIZE + PREDICTION_SIZE, 2), np.nan)
                global_n = np.zeros((N, HISTORY_SIZE, 2))  # [[] for i in range(len(neighbors[0]))]  # 6 x 20
                for ts_index, time_step in list(enumerate(neighbors)):  # x 20
                    if ts_index < HISTORY_SIZE:
                        for index, neighbor in enumerate(time_step):  # x 6
                            global_n[index, ts_index, 0] = neighbor[0]
                            global_n[index, ts_index, 1] = neighbor[1]
                    for index, neighbor in enumerate(time_step):  # x 6
                        global_n_plus[index, ts_index, 0] = neighbor[0]
                        global_n_plus[index, ts_index, 1] = neighbor[1]

                def f(x, full):
                    exception = NotEnoughPoints()
                    possibilities = []
                    for y in fu.get_candidate_centerlines_for_trajectory(
                            full, item.city, am,
                            viz=False, seq_len=90,
                            max_search_radius=5):
                        try:
                            possibilities.append(cut_lanes(y, x))
                        except NotEnoughPoints as g:
                            exception = g
                    if not possibilities:
                        raise exception
                    return possibilities

                def convert(thing):
                    return convert_global_coords_to_local(thing, target_history[-1], rot)

                global_lanes = np.array([f(*i) for i in zip(global_n, global_n_plus)])
                local_neighbors = np.array([list(map(convert, i)) for i in global_n])
                local_lanes = np.array([list(map(convert, i)) for i in global_lanes])
                agent_x = item.seq_df[item.seq_df["OBJECT_TYPE"] == "AGENT"]["X"][:HISTORY_SIZE + 1]
                agent_y = item.seq_df[item.seq_df["OBJECT_TYPE"] == "AGENT"]["Y"][:HISTORY_SIZE + 1]
                agent_history, agent_history_bis = np.column_stack((agent_x, agent_y)), [agent_x, agent_y]
                agent_x = item.seq_df[item.seq_df["OBJECT_TYPE"] == "AGENT"]["X"]
                agent_y = item.seq_df[item.seq_df["OBJECT_TYPE"] == "AGENT"]["Y"]
                global_lane = f(np.array(list(zip(agent_x, agent_y))), agent_history)
                local_lane = list(map(convert, global_lane))

                nu = (lambda j: j + 1)
                d = (lambda l: sum(min(np.linalg.norm(c - m) for m in l) * nu(i) for i, c in enumerate(local_future)))
                m, n = sorted(enumerate(local_lanes), key=(lambda k: d(k[1])))[0]

                # show_trajectory(local_history, local_future, *local_neighbors[0], *local_lanes[0])
                # show_trajectory(local_history, local_future, local_lane)

                cache[afl_index] = dict(
                    history=local_history,
                    future=local_future,
                    neighbors=local_neighbors,
                    lanes=local_lanes,
                    neighbors_indices=neighbors_indices,
                    distances=distances,
                    stationary=stationary,
                    presences=presences,
                    reference_lane_index=m,
                    reference_lane_coordinates=n,
                    local_lane=local_lane)
                cache.commit()

                """
                def f(x):
                    return cut_lanes(x[0], None)
        
                global_lanes = np.array([f(am.get_candidate_centerlines_for_traj(i, item.city, False)) for i in global_n])
                global_lane = f(am.get_candidate_centerlines_for_traj(np.array(list(zip(agent_x, agent_y))), item.city, False))
                print(np.array(global_lanes).shape, np.array(global_n).shape)
                local_neighbors = np.array([convert_global_coords_to_local(i, target_history[-1], rot) for i in global_n])
                local_lanes = np.array([convert_global_coords_to_local(i, target_history[-1], rot) for i in global_lanes])
                local_lane = convert_global_coords_to_local(global_lane, target_history[-1], rot)"""

                # show_trajectory(local_history, local_future, *local_neighbors, *local_lanes)
                # show_trajectory(local_history, local_future, local_lane)
                # print(local_lane.shape)
                # print(local_lanes.shape)
                # raise ValueError()

                # print(neighbors_indices, )

                continue

                """sequence_iterator = iter(item.seq_df)
                headers = next(sequence_iterator)
                for row in sequence_iterator:
                    print(dict(zip(headers, row)))"""

                print(item.current_seq)

                for timestamp in set(item.seq_df['TIMESTAMP']):
                    timestamp_coordinate = []
                    for other in others:
                        print((item.seq_df['TIMESTAMP'] == timestamp))
                        raise RuntimeError(
                            item.seq_df[(item.seq_df['TIMESTAMP'] == timestamp)]['TRACK_ID'] == other
                        )
                raise RuntimeError

                others_histories = []
                for other in others:
                    others_histories.append((
                        item.seq_df[item.seq_df["TRACK_ID"] == other]["X"][:HISTORY_SIZE + 1],
                        item.seq_df[item.seq_df["TRACK_ID"] == other]["Y"][:HISTORY_SIZE + 1]))
                    print(others_histories[-1][0].shape, others_histories[-1][0].shape)
                others_histories = np.array(others_histories)  # 44 x 2

                agent_x = item.seq_df[item.seq_df["OBJECT_TYPE"] == "AGENT"]["X"]
                agent_y = item.seq_df[item.seq_df["OBJECT_TYPE"] == "AGENT"]["Y"]
                agent_full = np.column_stack((agent_x, agent_y))

                current_coordinates = np.array([list(agent_x)[HISTORY_SIZE], list(agent_y)[HISTORY_SIZE]])

                agent_x = item.seq_df[item.seq_df["OBJECT_TYPE"] == "AGENT"]["X"][HISTORY_SIZE + 1:][:PREDICTION_SIZE]
                agent_y = item.seq_df[item.seq_df["OBJECT_TYPE"] == "AGENT"]["Y"][HISTORY_SIZE + 1:][:PREDICTION_SIZE]
                agent_future, agent_future_bis = np.column_stack((agent_x, agent_y)), [agent_x, agent_y]

                for all_x, all_y in others_histories:
                    for (x, y) in zip(all_x, all_y):
                        print(x, y)
                        exit()
                    am.get_lane_segments_containing_xy()

                args = []
                for lane in am.get_candidate_centerlines_for_traj(agent_full, item.city, False):
                    args.extend(np.array(list(zip(*lane))))
                plt.plot(*agent_future_bis, *agent_history_bis, *args)
                plt.show()

                # agent_traj = np.column_stack((agent_x, agent_y))

                # print(track_id)

            # am.get_nearest_centerline(item.agent_traj[0], item.city, False)

            print('Total number of sequences:', len(afl))
        except NotEnoughPoints as e:
            print('Error in', afl_index, *e.args)
        except Exception as e:
            print(e)
        else:
            print('Success', afl_index)

        if afl_index in cache:
            """dict(
                history=local_history,
                future=local_future,
                neighbors=local_neighbors,
                lanes=local_lanes,
                neighbors_indices=neighbors_indices,
                distances=distances,
                stationary=stationary,
                presences=presences,
                reference_lane_index=m,
                reference_lane_coordinates=n,
                local_lane)"""
            item = cache[afl_index]
            plt.plot(*zip(*item['history']))
            plt.plot(*zip(*item['future']))
            for lane in item['lanes']:
                plt.plot(*zip(*lane))
            for neighbor in item['neighbors']:
                plt.plot(*zip(*neighbor))
            plt.show()


if __name__ == '__main__':
    # main()
    for i in get_items():
        print(i)
