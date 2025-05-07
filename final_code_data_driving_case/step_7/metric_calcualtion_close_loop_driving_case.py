import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def draw_trace(trace_file_path_list, col_name_list, folder):
    for trace_file_path in trace_file_path_list:
        df = pd.read_csv(trace_file_path, index_col=0)
        for col_name in col_name_list:
            df_draw = df[df['episode']==0][col_name].to_list()
            df_draw = list(map(abs,df_draw))
            x = np.arange(1,len(df_draw)+1)
            plt.plot(x, df_draw)

            plt.legend(labels=['proposed', 'baseline'], loc='best')
            plt.title(col_name)
            plt.savefig('{folder}/acc_trace_baseline.png'.format(folder=folder))
    plt.show()
        #print(df_draw.max(), df_draw.min(), df_draw.mean())
    return

def draw_trace_each_eps(trace_file_path, col_name_list, folder):
    df = pd.read_csv(trace_file_path, index_col=0)
    total_eps = 1
    for eps in range(0, total_eps):
        df_part = df[df['episode']==eps]
        print(eps, df_part)
        for col_name in col_name_list:
            df_draw = df_part[col_name].to_list()
            #df_draw = list(map(abs, df_draw))
            x = np.arange(1, len(df_draw)+1)
            plt.plot(x, df_draw)
            plt.title(col_name)
            plt.xlabel('Time step')
            plt.ylabel('Acceleration(m/s2)')
    plt.savefig('{folder}/acc_trace_baseline.png'.format(folder=folder))

    plt.show()
    return

figure_folder = '/home/cpsgroup/predictive_monitor_new_stl/closed_loop_simulation'
trace_file_path_1 = '/home/cpsgroup/SafeBench/safebench/predictive_monitor_trace_new_stl/trace_file_close_loop_simulation/behavior_type_2/lstm_with_monitor/eval_agent_behavior_type_2_scenario_3_route_7_episode_30_seed_0.csv'

trace_file_path_2 = '/home/cpsgroup/SafeBench/safebench/predictive_monitor_trace_new_stl/trace_file_close_loop_simulation/behavior_type_2/no_lstm/eval_agent_behavior_type_2_scenario_3_route_7_episode_30_seed_0.csv'


###################################
# -*- coding: utf-8 -*-
import pandas as pd
import os
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from matplotlib.colors import to_rgba
import seaborn as sns
from tqdm.notebook import tqdm
from tqdm import trange
import time
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import torch._VF as _VF

import json
import random
import sys

sys.path.append('/home/cpsgroup/predictive_monitor/stlu_monitor')
import ustlmonitor as ustl
import confidencelevel
import argparse

torch.set_printoptions(precision=4, sci_mode=False)
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '%.6f' % x)
pd.set_option('display.max_columns', None)


# STL for checking acc
def requirement_func_always_acc_in_range(signal, trace_len, conf=0.95, lower_acc=-6.0, upper_acc=6.0,
                                         func='monitor'):
    # signal: Speed trace, Keep speed in range
    # G[0,t](speed > lower_speed & speed < upper_speed)
    # STL: G[0,10](signal>lower_speed) and (signal<upper_speed)
    # convert to:
    # G[0,10] (signal>lower_speed) and neg(signal>upper_speed)
    threshold_1 = lower_acc
    threshold_2 = upper_acc
    varphi_1 = (("mu", signal), [threshold_1, conf])
    varphi_2 = ((("neg", 0), (("mu", signal), [threshold_2, conf])))
    varphi_3 = (("always", (0, trace_len - 1)), (("and", 0), (varphi_1, varphi_2)))

    varphi_1_1 = (("mu", signal), threshold_1)
    varphi_2_1 = ((("neg", 0), (("mu", signal), threshold_2)))
    varphi_3_1 = (("always", (0, trace_len - 1)), (("and", 0), (varphi_1_1, varphi_2_1)))
    if func == 'monitor':
        return ustl.umonitor(varphi_3, 0)
    else:
        return (varphi_3_1, 0)


def requirement_func_always_acc_not_low(signal, trace_len, conf, lower_acc=-6.0, func='monitor'):
    # signal: acc trace, check acc not lower than -3.4
    # G[0,t](acc > lower_acc)
    # STL: G[0,20](signal>lower_acc)
    # convert to:
    # G[0,20] (signal>lower_acc)
    threshold_1 = lower_acc
    varphi_1 = (("mu", signal), [threshold_1, conf])
    varphi_3 = (("always", (0, trace_len - 1)), varphi_1)

    varphi_1_1 = (("mu", signal), threshold_1)
    varphi_3_1 = (("always", (0, trace_len - 1)), varphi_1_1)
    if func == 'monitor':
        return ustl.umonitor(varphi_3, 0)
    else:
        return (varphi_3_1, 0)


def requirement_func_always_acc_not_high(signal, trace_len, conf, upper_acc=6.0, func='monitor'):
    # signal: acc trace
    # G[0,t](acc < upper_acc)
    # STL: G[0,20](signal<upper_acc)
    # convert to:
    # G[0,20] neg(signal>upper_acc)
    threshold_2 = upper_acc
    varphi_2 = ((("neg", 0), (("mu", signal), [threshold_2, conf])))
    varphi_3 = (("always", (0, trace_len - 1)), varphi_2)

    varphi_2_1 = ((("neg", 0), (("mu", signal), threshold_2)))
    varphi_3_1 = (("always", (0, trace_len - 1)), varphi_2_1)
    if func == 'monitor':
        return ustl.umonitor(varphi_3, 0)
    else:
        return (varphi_3_1, 0)


def requirement_func_check_rho_low(signal, trace_len, conf=0.95, func='monitor'):
    # signal: acc trace
    # G[0,t](rho_low < 0)
    # STL: G[0,t]neg(signal>0)
    threshold_1 = 0
    varphi_1 = ((("neg", 0), (("mu", signal), [threshold_1, conf])))
    varphi_3 = (("always", (0, trace_len - 1)), varphi_1)

    varphi_1_1 = ((("neg", 0), (("mu", signal), threshold_1)))
    varphi_3_1 = (("always", (0, trace_len - 1)), varphi_1_1)
    if func == 'monitor':
        return ustl.umonitor(varphi_3, 0)
    else:
        return (varphi_3_1, 0)


# calculate evaluation metrics-with dropout & STLU rho_set
def get_satisfaction_type_one_segment_STLU(rho):
    # define satisfaction type for one rho_set or rho_orig provided by STLU
    satisfaction_type = 0
    lower = round(rho[0], 3)
    upper = round(rho[1], 3)
    if (lower * upper < 0):
        satisfaction_type = 1  # weak satisfaction, one boundary<0, another>0
    elif (lower > 0 and upper > 0):
        satisfaction_type = 2  # strong satisfaction, both boundary>0
    else:
        satisfaction_type = 3  # violation, both boundary<=0
    return satisfaction_type


def calculate_consecutive_prediction_metrics_total_STLU(segment_df, rho_real_col_name, rho_pred_col_name, hazard_type,
                                                        method):
    rho_low_pred_list = []
    rho_low_real_list = []
    rho_check_rho_low_pred_list = []
    rho_check_rho_low_real_list = []
    satisfaction_type_real = []
    satisfaction_type_predict = []
    if_warning_real_list = []
    if_warning_predict_list = []
    for index, row in segment_df.iterrows():
        rho_set_acc = [float(x) for x in row[rho_pred_col_name]]
        rho_orig_acc = [float(x) for x in row[rho_real_col_name]]
        rho_low_pred_list.append(rho_set_acc[0])
        rho_low_real_list.append(rho_orig_acc[0])
    trace_len = 3  # check on 3 consecutive segments' rho intervals?
    for i in range(len(rho_low_pred_list) - (trace_len - 1)):
        rho_low_pred_trace = torch.Tensor(rho_low_pred_list[i:i + trace_len])
        rho_low_real_trace = torch.Tensor(rho_low_real_list[i:i + trace_len])
        trace_pred = torch.stack((rho_low_pred_trace, torch.zeros(rho_low_pred_trace.size())), dim=-1)
        trace_real = torch.stack((rho_low_real_trace, torch.zeros(rho_low_real_trace.size())), dim=-1)
        rho_check_rho_low_pred = requirement_func_check_rho_low(trace_pred, trace_len=trace_len, conf=0.95,
                                                                func='monitor')
        rho_check_rho_low_real = requirement_func_check_rho_low(trace_real, trace_len=trace_len, conf=0.95,
                                                                func='monitor')
        rho_check_rho_low_pred_list.append(rho_check_rho_low_pred.tolist())
        rho_check_rho_low_real_list.append(rho_check_rho_low_real.tolist())
        st_central_real = get_satisfaction_type_one_segment_STLU(rho_check_rho_low_real)
        st_central_predict = get_satisfaction_type_one_segment_STLU(rho_check_rho_low_pred)
        satisfaction_type_predict.append(st_central_predict)
        satisfaction_type_real.append(st_central_real)

        if (st_central_real == 2):
            if_warning_orig = 1
        else:
            if_warning_orig = 0
        if (st_central_predict == 2):
            if_warning_predict = 1
        else:
            if_warning_predict = 0
        if_warning_real_list.append(if_warning_orig)
        if_warning_predict_list.append(if_warning_predict)
    total_num = len(if_warning_real_list)
    if_warning_real_sum = sum(if_warning_real_list)
    real_warning_percentage = if_warning_real_sum / total_num
    TP, TN, FP, FN = 0, 0, 0, 0
    TPR, TNR, FPR, FNR = 0, 0, 0, 0
    accuracy = 0
    for k in range(len(if_warning_real_list)):
        if_warning_real = if_warning_real_list[k]
        if_warning_predict = if_warning_predict_list[k]
        predicted_warning = if_warning_predict
        if if_warning_real == 1 and predicted_warning == 1:
            TP += 1
        elif if_warning_real == 1 and predicted_warning == 0:
            FN += 1
        elif if_warning_real == 0 and predicted_warning == 1:
            FP += 1
        elif if_warning_real == 0 and predicted_warning == 0:
            TN += 1

    TPR = TP / (TP + FN) if (TP + FN) != 0 else None
    TNR = TN / (TN + FP) if (TN + FP) != 0 else None
    FPR = FP / (FP + TN) if (FP + TN) != 0 else None
    FNR = FN / (TP + FN) if (TP + FN) != 0 else None
    accuracy = (TP + TN) / total_num if total_num != 0 else None

    percision = TP / (TP + FP) if (TP + FP) != 0 else None
    recall = TP / (TP + FN) if (TP + FN) != 0 else None
    if percision != None and recall != None and (percision + recall) != 0:
        F1_score = 2 * percision * recall / (percision + recall)
    else:
        F1_score = None

    results_dict = {'hazard_type': hazard_type, 'method': method, 'metric_type': 'consecutive', 'TP': TP, 'TN': TN,
                    'FP': FP, 'FN': FN, 'TPR': TPR, 'TNR': TNR, 'FPR': FPR, 'FNR': FNR,
                    'accuracy': accuracy, 'F1_score': F1_score, 'real_warning_percentage': real_warning_percentage}
    print('Metrics STLU if_warning_predict_consecutive: ', results_dict)

    return results_dict


# calculate metrics for total, high(>6.0) and low(-6.0) separately
def calculate_metrics_for_each_hazard_type(segment_output_file_path, hazard_type='high', method='flowpipe_mean'):
    predict_len = 20
    segment_df = pd.read_csv(segment_output_file_path, index_col=0)
    if hazard_type == 'high':
        requirement_func = requirement_func_always_acc_not_high
    if hazard_type == 'low':
        requirement_func = requirement_func_always_acc_not_low
    if hazard_type == 'total':
        requirement_func = requirement_func_always_acc_in_range
    rho_real_list = []
    rho_pred_list = []
    satisfaction_type_real = []
    satisfaction_type_predict = []
    if_warning_real_list = []
    if_warning_predict_list = []
    for index, row in segment_df.iterrows():
        predict_trace_mean = torch.Tensor([float(x) for x in row['acc_predicted_mean'][1:-1].split(',')])
        predict_trace_std = torch.Tensor([float(x) for x in row['acc_predicted_std'][1:-1].split(',')])
        real_trace = torch.Tensor([float(x) for x in row['each_real_acc_in_batch'][1:-1].split(',')][-predict_len:])
        if method == 'flowpipe_mean':
            trace_pred = torch.stack((predict_trace_mean, torch.zeros(predict_trace_mean.size())), dim=-1)
            trace_real = torch.stack((real_trace, torch.zeros(real_trace.size())), dim=-1)
        elif method == 'flowpipe_rho':
            trace_pred = torch.stack((predict_trace_mean, predict_trace_std), dim=-1)
            trace_real = torch.stack((real_trace, torch.zeros(real_trace.size())), dim=-1)
        rho_pred = requirement_func(trace_pred, trace_len=predict_len, conf=0.95, func='monitor')
        rho_real = requirement_func(trace_real, trace_len=predict_len, conf=0.95, func='monitor')
        rho_real_list.append(rho_real.tolist())
        rho_pred_list.append(rho_pred.tolist())

        st_central_real = get_satisfaction_type_one_segment_STLU(rho_real)
        st_central_predict = get_satisfaction_type_one_segment_STLU(rho_pred)
        satisfaction_type_predict.append(st_central_predict)
        satisfaction_type_real.append(st_central_real)

        if (st_central_real == 1 or st_central_real == 3):
            if_warning_orig = 1  # warning for st=violation & weak satisfaction
        else:
            if_warning_orig = 0  # no warning for st=strong satisfy
        if (st_central_predict == 1 or st_central_predict == 3):
            if_warning_predict = 1  # warning for st=violation & weak satisfaction
        else:
            if_warning_predict = 0  # no warning for st=strong satisfy

        if_warning_real_list.append(if_warning_orig)
        if_warning_predict_list.append(if_warning_predict)

    segment_df['rho_real_hazard'] = rho_real_list
    segment_df['rho_pred_hazard'] = rho_pred_list
    segment_df['satisfaction_type_real_rho_hazard'] = satisfaction_type_real
    segment_df['satisfaction_type_predict_rho_hazard'] = satisfaction_type_predict
    segment_df['if_warning_real_hazard'] = if_warning_real_list
    segment_df['if_warning_predict_hazard'] = if_warning_predict_list

    warning_label_df = pd.DataFrame(segment_df, columns=['if_warning_real_hazard', 'if_warning_predict_hazard'])
    total_num = len(warning_label_df)
    if_warning_real_sum = warning_label_df['if_warning_real_hazard'].sum()
    real_warning_percentage = if_warning_real_sum / total_num
    TP, TN, FP, FN = 0, 0, 0, 0
    TPR, TNR, FPR, FNR = 0, 0, 0, 0
    accuracy = 0
    for item, row in warning_label_df.iterrows():
        if_warning_real = int(row['if_warning_real_hazard'])
        if_warning_predict = int(row['if_warning_predict_hazard'])
        predicted_warning = if_warning_predict
        if if_warning_real == 1 and predicted_warning == 1:
            TP += 1
        elif if_warning_real == 1 and predicted_warning == 0:
            FN += 1
        elif if_warning_real == 0 and predicted_warning == 1:
            FP += 1
        elif if_warning_real == 0 and predicted_warning == 0:
            TN += 1

    TPR = TP / (TP + FN) if (TP + FN) != 0 else None
    TNR = TN / (TN + FP) if (TN + FP) != 0 else None
    FPR = FP / (FP + TN) if (FP + TN) != 0 else None
    FNR = FN / (TP + FN) if (TP + FN) != 0 else None
    accuracy = (TP + TN) / total_num if total_num != 0 else None

    percision = TP / (TP + FP) if (TP + FP) != 0 else None
    recall = TP / (TP + FN) if (TP + FN) != 0 else None
    if percision != None and recall != None and (percision + recall) != 0:
        F1_score = 2 * percision * recall / (percision + recall)
    else:
        F1_score = None

    normal_results_dict = {'hazard_type': hazard_type, 'method': method, 'metric_type': 'not_consecutive', 'TP': TP,
                           'TN': TN, 'FP': FP, 'FN': FN, 'TPR': TPR, 'TNR': TNR, 'FPR': FPR, 'FNR': FNR,
                           'accuracy': accuracy, 'F1_score': F1_score,
                           'real_warning_percentage': real_warning_percentage}
    print('Metrics STLU if_warning_predict_hazard: ', normal_results_dict)
    consecutive_results_dict = {}
    if method == 'flowpipe_rho':
        consecutive_results_dict = calculate_consecutive_prediction_metrics_total_STLU(segment_df, 'rho_real_hazard',
                                                                                       'rho_pred_hazard', hazard_type,
                                                                                       method)
    else:
        consecutive_results_dict = {}
    if hazard_type == 'total' and method == 'flowpipe_rho':
        segment_df['rho_orig_acc'] = rho_real_list
        segment_df['rho_set_acc'] = rho_pred_list
        segment_df['satisfaction_type_real_acc'] = satisfaction_type_real
        segment_df['satisfaction_type_predict_acc'] = satisfaction_type_predict
        segment_df['if_warning_real'] = if_warning_real_list
        segment_df['if_warning_predict'] = if_warning_predict_list

    segment_df.to_csv(segment_output_file_path)

    return normal_results_dict, consecutive_results_dict


############
def get_ground_truth_whole_trace(segment_file_path, trace_start_index, trace_end_index):
    segment_df = pd.read_csv(segment_file_path, index_col=0)[trace_start_index:trace_end_index + 1]
    acc_list = []
    acc_df = pd.DataFrame()
    for index, row in segment_df.iterrows():
        segment_real_acc = [float(x) for x in row['each_real_acc_in_batch'][1:-1].split(',')]
        if index == 0:
            acc_list = segment_real_acc
        else:
            acc_list.append(segment_real_acc[-1])

    acc_df['real_acc'] = acc_list
    return acc_df


def mark_hazard_id_for_real_acc(driving_trace_df, low=-6.0, high=6.0):
    hazard_id_list = []
    hazard_type_list = []
    hazard_time_list = []
    acc_list = driving_trace_df['acc'].tolist()
    hazard_id_count = 0
    hazard_type = 0
    hazard_time = 0
    last_hazard_end_time = 0
    for i in range(len(acc_list)):
        current_acc = acc_list[i]
        if current_acc <= low:
            hazard_type = 4  # severe low
        elif low < current_acc < low * 0.5:
            hazard_type = 2  # mild low
        elif high < current_acc:
            hazard_type = 3  # mild high
        # elif 250<=current_acc:
        #    hazard_type = 5 # severe high
        else:
            hazard_type = 0  # no hazard
        hazard_type_list.append(hazard_type)
        if i == 0:
            if hazard_type != 0:
                hazard_id_count += 1
                hazard_id_list.append(hazard_id_count)
                last_hazard_end_time = i
            else:
                hazard_id_list.append(0)
        else:
            last_hazard_type = hazard_type_list[i - 1]
            if hazard_type != last_hazard_type and hazard_type != 0:
                if i - last_hazard_end_time <= 10:  # within 1.0 s
                    hazard_id_list.append(0)
                    last_hazard_end_time = i
                else:
                    hazard_id_count += 1
                    hazard_id_list.append(hazard_id_count)
                    last_hazard_end_time = i
            elif hazard_type == 0:
                hazard_id_list.append(0)
            else:
                hazard_id_list.append(0)
                last_hazard_end_time = i

    driving_trace_df['hazard_type'] = hazard_type_list
    driving_trace_df['hazard_id'] = hazard_id_list

    hazard_id_index_list = driving_trace_df[driving_trace_df['hazard_id'] != 0].index.to_list()
    final_hazard_type_list = []
    # print('hazard_id_index_list: ', hazard_id_index_list)
    for j in range(len(acc_list)):
        if j in hazard_id_index_list:
            next_index = hazard_id_index_list[hazard_id_index_list.index(j) + 1] if hazard_id_index_list.index(j) != (
                        len(hazard_id_index_list) - 1) else len(acc_list) - 1
            if j != next_index:
                hazard_time_step_num = len(driving_trace_df[j:next_index])
                # print('j: ', j, ' next_index: ', next_index, ' hazard_time_step_num: ', hazard_time_step_num)
                # print('list: ', driving_trace_df[j:next_index][driving_trace_df['hazard_type']!=0]['real_acc'].to_list())
                max_acc_this_hazard = max(
                    driving_trace_df[j:next_index][driving_trace_df['hazard_type'] != 0]['acc'].to_list())
                min_acc_this_hazard = min(
                    driving_trace_df[j:next_index][driving_trace_df['hazard_type'] != 0]['acc'].to_list())
                high_time_this_hazard = len(driving_trace_df[j:next_index][(driving_trace_df['hazard_type'] != 0) & (
                            driving_trace_df['acc'] > high)])
                low_time_this_hazard = len(driving_trace_df[j:next_index][(driving_trace_df['hazard_type'] != 0) & (
                            driving_trace_df['acc'] < low * 0.5)])
                if high_time_this_hazard >= low_time_this_hazard:
                    if max_acc_this_hazard >= high:
                        final_hazard_type = 3
                    # elif 180<=max_acc_this_hazard<250:
                    #  final_hazard_type = 3
                else:
                    if low * 0.5 >= min_acc_this_hazard > low:
                        final_hazard_type = 2
                    elif min_acc_this_hazard <= low:
                        final_hazard_type = 4
                final_hazard_type_list.append(final_hazard_type)
                hazard_time_list.append(hazard_time_step_num)
            else:
                hazard_time_list.append(0)
                final_hazard_type_list.append(0)
        else:
            hazard_time_list.append(0)
            final_hazard_type_list.append(0)
    driving_trace_df['hazard_time'] = hazard_time_list
    driving_trace_df['final_hazard_type'] = final_hazard_type_list
    return driving_trace_df


def calculate_pre_alert_time(segment_file_path, real_acc_trace_path, trace_start_index, trace_end_index,
                             control_type='lstm_with_monitor', hazard_type='total', low=-6.0, high=6.0):
    segment_df = pd.read_csv(segment_file_path, index_col=0)[trace_start_index:trace_end_index + 1]
    real_acc_df = get_ground_truth_whole_trace(segment_file_path, trace_start_index, trace_end_index)
    # print(len(real_acc_df))
    real_acc_df = mark_hazard_id_for_real_acc(real_acc_df)
    # print('segment_df: ', segment_df)
    # print('real_acc_df: ', real_acc_df)
    pre_alert_time_list = []
    pre_alert_time = 0
    hazard_index_list = real_acc_df[(real_acc_df.hazard_id != 0)].index.to_list()
    # print('hazard_index_list: ', hazard_index_list)
    earlist_prediction_index_list = []
    # calculate pre-alert time
    for i in range(len(real_acc_df)):
        if i in hazard_index_list:
            # print('i: ', i)
            if i >= 49:  # total_segment_len=50, past=30, pred=20
                check_prediction_segment_index = range(i - 49, i - 29)  # pred index that includes this hazard i, i-19
            elif 29 <= i < 49:
                check_prediction_segment_index = range(0, i - 29)
            else:
                check_prediction_segment_index = []
            # print('check_prediction_segment_index: ', check_prediction_segment_index)
            if len(check_prediction_segment_index) > 0:
                for k in check_prediction_segment_index:
                    if k < len(segment_df):
                        # print(k, check_prediction_segment_index)
                        # print('segment_df[rho_set_acc]: ', segment_df['rho_set_acc'])
                        rho_interval = [float(x) for x in (segment_df['rho_set_acc'].iloc[k])[1:-1].split(',')]
                        mean_trace = [float(x) for x in (segment_df['acc_predicted_mean'].iloc[k])[1:-1].split(',')]
                        if control_type == 'lstm_with_monitor':
                            if rho_interval[0] < 0:
                                pre_alert_time = i - 29 - k  # i-19-k
                                earlist_prediction_index_list.append(k)
                                break
                            else:
                                if k == max(check_prediction_segment_index):
                                    earlist_prediction_index_list.append(0)
                                continue
                        elif control_type == 'lstm_no_monitor':
                            if max(mean_trace) > high or min(mean_trace) < low:
                                pre_alert_time = i - 29 - k  # i-19-k
                                earlist_prediction_index_list.append(k)
                                break
                            else:
                                if k == max(check_prediction_segment_index):
                                    earlist_prediction_index_list.append(0)
                                continue
                # print('pre_alert_time: ', pre_alert_time)
                pre_alert_time_list.append(pre_alert_time)
                pre_alert_time = 0
            else:
                pre_alert_time_list.append(0)
        else:
            pre_alert_time_list.append(0)
    real_acc_df['pre_alert_time'] = pre_alert_time_list
    if hazard_type == 'total':
        total_hazard_num = len(real_acc_df[real_acc_df['hazard_id'] != 0])
        total_pre_alert_time = sum(real_acc_df[real_acc_df['hazard_id'] != 0]['pre_alert_time'].to_list())
    elif hazard_type == 'high':
        hazard_df = real_acc_df[(real_acc_df['hazard_id'] != 0) & (
                    (real_acc_df['final_hazard_type'] == 3) | (real_acc_df['final_hazard_type'] == 5))]
        total_hazard_num = len(hazard_df)
        total_pre_alert_time = sum(hazard_df['pre_alert_time'].to_list())
    elif hazard_type == 'low':
        hazard_df = real_acc_df[(real_acc_df['hazard_id'] != 0) & (
                    (real_acc_df['final_hazard_type'] == 2) | (real_acc_df['final_hazard_type'] == 4))]
        total_hazard_num = len(hazard_df)
        total_pre_alert_time = sum(hazard_df['pre_alert_time'].to_list())
    if total_hazard_num != 0:
        average_pre_alert_time = total_pre_alert_time / total_hazard_num
    else:
        average_pre_alert_time = -99

    print('control type: ', control_type, ' total_hazard_num: ', total_hazard_num, ' average_pre_alert_time step: ',
          average_pre_alert_time)

    # count number of flowpipes/mean traces that successfully predicts a hazard for one hazard
    total_successful_prediction_index_list_all_hazard = []
    total_successful_prediction_index_list_one_hazard = []
    total_successful_prediction_num = 0
    total_successful_prediction_num_list = []
    for i in range(len(real_acc_df)):
        # print('i: ', i, 'len(real_acc_df): ', len(real_acc_df))
        if i in hazard_index_list:
            if i >= 49:  # total_segment_len=50, past=30, pred=20
                check_prediction_segment_index = range(i - 49, i - 29)  # pred index that includes this hazard i, i-19
            elif 29 <= i < 49:
                check_prediction_segment_index = range(0, i - 29)
            else:
                check_prediction_segment_index = []
            # print('check_prediction_segment_index: ', check_prediction_segment_index)
            for k in check_prediction_segment_index:
                if k < len(segment_df):
                    # print('k: ', k, 'segment_df[rho_set_acc].iloc[k]: ', segment_df['rho_set_acc'].iloc[k])
                    rho_interval = [float(x) for x in (segment_df['rho_set_acc'].iloc[k])[1:-1].split(',')]
                    mean_trace = [float(x) for x in (segment_df['acc_predicted_mean'].iloc[k])[1:-1].split(',')]
                    if control_type == 'lstm_with_monitor':
                        if rho_interval[0] < 0:
                            total_successful_prediction_num += 1
                            total_successful_prediction_index_list_one_hazard.append(k)
                    elif control_type == 'lstm_no_monitor':
                        if max(mean_trace) > high or min(mean_trace) < low:
                            total_successful_prediction_num += 1
                            total_successful_prediction_index_list_one_hazard.append(k)
            total_successful_prediction_index_list_all_hazard.append(total_successful_prediction_index_list_one_hazard)
            total_successful_prediction_num_list.append(total_successful_prediction_num)
            total_successful_prediction_num = 0
            total_successful_prediction_index_list_one_hazard = []
        else:
            total_successful_prediction_num_list.append(0)
            total_successful_prediction_index_list_all_hazard.append([])
    real_acc_df['successful_prediction_num'] = total_successful_prediction_num_list
    real_acc_df['successful_prediction_index'] = total_successful_prediction_index_list_all_hazard
    real_acc_df.to_csv(real_acc_trace_path)
    return total_hazard_num, average_pre_alert_time

# count the number of each hazard type for each trace of each behavior type
def preprocess_driving_data(driving_trace_file):
    # choose one of the acc_x, acc_y to represent total acc, and rename columns
    # decide the sign of the final acc, according to acc/dec rather than vector direction
    acc_list = []
    df = pd.read_csv(driving_trace_file, index_col=0)
    col_list = list(df.columns.values)
    if 'acc_cal' in col_list:
        print('already convert acc.')
    else:
        print('Convert acc.')
        df.rename(columns={'acc': 'acc_cal'}, inplace=True)
        #print('df after rename: ', df)
        acc_x_abs_mean = df['acc_x'].abs().mean()
        acc_y_abs_mean = df['acc_y'].abs().mean()
        if acc_x_abs_mean>=acc_y_abs_mean:
            df['acc'] = df['acc_x']
            df['v'] = df['v_x']
        else:
            df['acc'] = df['acc_y']
            df['v'] = df['v_y']

        for item, row in df.iterrows():
            acc = row['acc']
            v = row['v']
            if acc*v<0:
                acc_list.append(-abs(acc))
            else:
                acc_list.append(abs(acc))
        df['acc'] = acc_list
    df.to_csv(driving_trace_file)
    return

# ###
trace_folder = '/home/cpsgroup/SafeBench/safebench/predictive_monitor_trace_new_stl/trace_file_close_loop_simulation'
result_folder = '/home/cpsgroup/predictive_monitor_new_stl/closed_loop_simulation'
behavior_type_list = [0, 2]
control_type_list = ['lstm_with_monitor', 'no_lstm']
#hazard_type_list = ['total', 'high', 'low']
agent_mode = 'behavior'
scenario_id = 3
route_id = 7
total_episode = 30
seed_list = [0, 1, 2, 3]
hazard_type_count_result_df = pd.DataFrame(columns=['behavior_type', 'control_type', 'episode', 'severe_low', 'mild_low', 'high', 'total',
                                                    'average_speed', 'max_acc', 'min_acc', 'average_acc', 'average_distance'])
hazard_type_count_result_file_path = '{folder}/hazard_result_each_trace.csv'.format(folder=result_folder)

#
for behavior_type in behavior_type_list:
    for control_type in control_type_list:
        for seed in seed_list:
            trace_file_path = '{folder}/behavior_type_{behavior_type}/{control_type}/eval_agent_{agent_mode}_type_{behavior_type}_scenario_{scenario_id}_route_{route_id}_episode_{e_i}_seed_{seed}.csv'.format(
                folder=trace_folder, agent_mode=agent_mode, behavior_type=behavior_type,control_type=control_type,
                scenario_id=scenario_id, route_id=route_id, e_i=total_episode, seed=seed)
            preprocess_driving_data(trace_file_path)
            whole_trace_df = pd.read_csv(trace_file_path, index_col=0)
            trace_per_eps_df_with_hazard_id_list = []

            for eps in range(0, total_episode):
                hazard_type_count_dict = {'behavior_type':behavior_type, 'control_type':control_type, 'episode':eps}
                trace_per_eps_df = whole_trace_df[whole_trace_df['episode']==eps]
                trace_per_eps_df = trace_per_eps_df.reset_index()
                #print(trace_per_eps_df)
                trace_per_eps_df_with_hazard_id = mark_hazard_id_for_real_acc(trace_per_eps_df)
                trace_per_eps_df_with_hazard_id_list.append(trace_per_eps_df_with_hazard_id)
                for final_hazard_type in [2, 3, 4]:
                    hazard_number = len(trace_per_eps_df_with_hazard_id[trace_per_eps_df_with_hazard_id['final_hazard_type']==final_hazard_type])
                    hazard_type_count_dict[final_hazard_type] = hazard_number
                hazard_list = trace_per_eps_df_with_hazard_id[trace_per_eps_df_with_hazard_id['hazard_id']!=0]['hazard_id'].to_list()
                total_hazard_number = max(hazard_list) if len(hazard_list)!=0 else 0

                hazard_type_count_dict["severe_low"] = hazard_type_count_dict.pop(4)
                hazard_type_count_dict["mild_low"] = hazard_type_count_dict.pop(2)
                hazard_type_count_dict["high"] = hazard_type_count_dict.pop(3)
                hazard_type_count_dict['total'] = hazard_type_count_dict["severe_low"]+hazard_type_count_dict["high"]

                ###
                # calculate average speed, min/max acc, average relative distance
                average_speed = sum(trace_per_eps_df_with_hazard_id['speed'].to_list())/len(trace_per_eps_df_with_hazard_id['speed'].to_list())
                max_acc = max(trace_per_eps_df_with_hazard_id['acc'].to_list())
                min_acc = min(trace_per_eps_df_with_hazard_id['acc'].to_list())
                average_acc = sum(trace_per_eps_df_with_hazard_id['acc_cal'].to_list())/len(trace_per_eps_df_with_hazard_id['acc_cal'].to_list())
                average_distance = sum(trace_per_eps_df_with_hazard_id['radar_obj_depth_aveg'].to_list()) / len(
                    trace_per_eps_df_with_hazard_id['radar_obj_depth_aveg'].to_list())
                hazard_type_count_dict["average_speed"] = average_speed
                hazard_type_count_dict["max_acc"] = max_acc
                hazard_type_count_dict["min_acc"] = min_acc
                hazard_type_count_dict["average_acc"] = average_acc
                hazard_type_count_dict["average_distance"] = average_distance
                hazard_type_count_result_df = hazard_type_count_result_df.append(hazard_type_count_dict, ignore_index=True)
            whole_trace_df_with_hazard_id = pd.concat(trace_per_eps_df_with_hazard_id_list)
            whole_trace_df_with_hazard_id_path = '{folder}/behavior_type_{behavior_type}/{control_type}/eval_agent_{agent_mode}_type_{behavior_type}_scenario_{scenario_id}_route_{route_id}_episode_{e_i}_seed_{seed}_with_hazard_id.csv'.format(
                folder=trace_folder, agent_mode=agent_mode, behavior_type=behavior_type,control_type=control_type,
                scenario_id=scenario_id, route_id=route_id, e_i=total_episode, seed=seed)
            whole_trace_df_with_hazard_id.to_csv(whole_trace_df_with_hazard_id_path)
hazard_type_count_result_df.to_csv(hazard_type_count_result_file_path)
####

###
# calculate the average number of each hazard type for all traces of each behavior type
average_hazard_result_file_path = '{folder}/hazard_result_average.csv'.format(folder=result_folder)
average_hazard_result_df = pd.DataFrame(columns=['behavior_type', 'control_type', 'total_episode', 'average_sever_low', 'average_mild_low',
                                                 'average_high', 'average_total',
                                                'average_speed', 'average_acc', 'average_distance'])
hazard_type_count_result_df = pd.read_csv(hazard_type_count_result_file_path, index_col=0)
for behavior_type in behavior_type_list:
    for control_type in control_type_list:
        df = hazard_type_count_result_df[(hazard_type_count_result_df['behavior_type']==behavior_type) & (hazard_type_count_result_df['control_type']==control_type)]
        total_eps = max(df['episode'].to_list()) + 1
        average_sever_low = sum(df['severe_low'].to_list())/len(df['severe_low'].to_list()) if len(df['severe_low'].to_list())!=0 else 0
        average_mild_low = sum(df['mild_low'].to_list()) / len(df['mild_low'].to_list()) if len(
            df['mild_low'].to_list()) != 0 else 0
        average_high = sum(df['high'].to_list()) / len(df['high'].to_list()) if len(
            df['high'].to_list()) != 0 else 0
        average_total = sum(df['total'].to_list()) / len(df['total'].to_list()) if len(
            df['total'].to_list()) != 0 else 0
        average_speed = sum(df['average_speed'].to_list()) / len(df['average_speed'].to_list()) if len(
            df['average_speed'].to_list()) != 0 else 0
        average_distance = sum(df['average_distance'].to_list()) / len(df['average_distance'].to_list()) if len(
            df['average_distance'].to_list()) != 0 else 0
        average_acc = sum(df['average_acc'].to_list()) / len(df['average_acc'].to_list()) if len(
            df['average_acc'].to_list()) != 0 else 0

        average_hazard_result_dict = {'behavior_type': behavior_type, 'control_type': control_type, 'total_episode': total_eps,
                                      'average_sever_low':average_sever_low, 'average_mild_low':average_mild_low,
                                      'average_high': average_high, 'average_total':average_total,
                                      'average_speed':average_speed, 'average_acc':average_acc, 'average_distance':average_distance}

        average_hazard_result_df = average_hazard_result_df.append(average_hazard_result_dict, ignore_index=True)
average_hazard_result_df.to_csv(average_hazard_result_file_path)
###


