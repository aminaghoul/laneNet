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


