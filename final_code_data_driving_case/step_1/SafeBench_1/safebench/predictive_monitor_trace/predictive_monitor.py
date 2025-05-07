import numpy
import pickle
import sys

import pandas as pd

sys.path.append('/home/cpsgroup/CARLA_0.9.13_safebench/PythonAPI/carla')
import carla
import matplotlib.pyplot as plt
import math


def preprocess_driving_data(driving_trace_df):
    # get acc, and the change of acc
    # df = pd.read_csv(driving_trace_df, index_col=0)
    # change time format, save hour and minute
    acc_list = []
    delta_acc_list = []
    for item, row in driving_trace_df.iterrows():
        acc_x = row['acc_x']
        acc_y = row['acc_y']
        acc_z = row['acc_z']
        acc = math.sqrt(math.pow(acc_x, 2) + math.pow(acc_y, 2) + math.pow(acc_z, 2))
        acc_list.append(acc)
        if item == 0:
            delta_acc_list.append(0)
        else:
            delta_acc_list.append(acc - acc_list[item - 1])

    driving_trace_df['acc'] = acc_list
    driving_trace_df['delta_acc'] = delta_acc_list
    return driving_trace_df

behavior_type_list = [1]
route_id_list = [7]
seed_dict = {7:[21, 22, 23], 8:[14, 15, 16], 10:[17, 18, 19]}
episode = 1
folder = '/home/cpsgroup/SafeBench/safebench/predictive_monitor_trace/trace_file'
for behavior_id in behavior_type_list:
    for route_id in route_id_list:
        seed_list = seed_dict[route_id]
        for seed in seed_list:
            file_path = '{folder}/eval_agent_behavior_type_{behavior_id}_scenario_3_route_{route_id}_episode_10_seed_{seed}.csv'.format(
                behavior_id=behavior_id, route_id=route_id, seed=seed,folder=folder
            )
            df = pd.read_csv(file_path)
            df = preprocess_driving_data(df)
            for eps in range(episode):
                #part_df = df[df['episode']==eps][['speed', 'acc', 'radar_obj_depth_aveg', 'radar_obj_velocity_aveg']]
                part_df = df[df['episode'] == eps][['speed', 'acc']]
                print(part_df)
                part_df.plot()
            plt.show()


