import matplotlib.pyplot as plt
import tqdm
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.arcline_path_utils import length_of_lane
from nuscenes.map_expansion.map_api import NuScenesMap, locations
from nuscenes.prediction import PredictHelper, convert_global_coords_to_local
def main():
    nusc_map = NuScenesMap(dataroot='nuscenes', map_name='singapore-hollandvillage')

    print(nusc_map.non_geometric_layers)

    """for i in range(6):
        sample_drivable_area = nusc_map.drivable_area[i]
        fig, ax = nusc_map.render_record('drivable_area', sample_drivable_area['token'], other_layers=[])
        plt.show()"""
    sample_road_segment = nusc_map.road_segment[0]

    sample_intersection_road_segment = nusc_map.road_segment[0]
    """fig, ax = nusc_map.render_record('road_segment', sample_intersection_road_segment['token'], other_layers=[])
    plt.show()"""
    sample_road_block = nusc_map.road_block[60]
    print(sample_road_block['from_edge_line_token'])
    print(sample_road_block['to_edge_line_token'])
    """fig, ax = nusc_map.render_record('road_block', sample_road_block['token'], other_layers=[])
    plt.show()"""
    sample_lane_record = nusc_map.lane[60]
    """fig, ax = nusc_map.render_record('lane', sample_lane_record['token'], other_layers=[])
    plt.show()"""
    sample_ped_crossing_record = nusc_map.ped_crossing[0]
    print(sample_ped_crossing_record)
    """fig, ax = nusc_map.render_record('ped_crossing', sample_ped_crossing_record['token'])
    plt.show()"""
    sample_walkway_record = nusc_map.walkway[0]
    """fig, ax = nusc_map.render_record('walkway', sample_walkway_record['token'])
    plt.show()"""
    sample_stop_line_record = nusc_map.stop_line[1]
    fig, ax = nusc_map.render_record('stop_line', sample_stop_line_record['token'])
    plt.show()
    sample_carpark_area_record = nusc_map.carpark_area[1]
    fig, ax = nusc_map.render_record('carpark_area', sample_carpark_area_record['token'])
    plt.show()
    sample_road_divider_record = nusc_map.road_divider[0]








if __name__ == '__main__':
    main()