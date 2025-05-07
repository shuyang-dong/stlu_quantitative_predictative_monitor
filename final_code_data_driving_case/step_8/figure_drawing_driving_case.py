# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from matplotlib.colors import to_rgba
from tqdm.notebook import tqdm
from tqdm import trange
import time
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
pd.set_option('display.max_rows', None)


# draw Fig.5: acc/speed trace compare
figure_folder = '/home/cpsgroup/predictive_monitor_new_stl/closed_loop_simulation'
#trace_len = 300
seed = 2
feature_to_draw_1 = 'speed'#'acc'
ylabel_1 = 'Speed(m/s)'#'Acceleration'
feature_to_draw_2 = 'acc'
ylabel_2 = 'Acceleration(m/s2)'
figure_name = 'trace_compare'

feature_to_draw = feature_to_draw_2
ylabel = ylabel_2
for id in range(1):
  b0_propose_trace_path = '/home/cpsgroup/SafeBench/safebench/predictive_monitor_trace_new_stl/trace_file_close_loop_simulation/behavior_type_0/lstm_with_monitor/eval_agent_behavior_type_0_scenario_3_route_7_episode_30_seed_{seed}.csv'.format(seed=seed)
  b0_baseline_trace_path = '/home/cpsgroup/SafeBench/safebench/predictive_monitor_trace_new_stl/trace_file_close_loop_simulation/behavior_type_0/no_lstm/eval_agent_behavior_type_0_scenario_3_route_7_episode_30_seed_{seed}.csv'.format(seed=seed)

  b0_df_propose = pd.read_csv(b0_propose_trace_path, index_col=0)
  b0_df_baseline = pd.read_csv(b0_baseline_trace_path, index_col=0)
  b0_propose_trace = b0_df_propose[b0_df_propose['episode']==10][feature_to_draw].to_list()
  b0_baseline_trace = b0_df_baseline[b0_df_baseline['episode']==10][feature_to_draw].to_list()

  b2_propose_trace_path = '/home/cpsgroup/SafeBench/safebench/predictive_monitor_trace_new_stl/trace_file_close_loop_simulation/behavior_type_2/lstm_with_monitor/eval_agent_behavior_type_2_scenario_3_route_7_episode_30_seed_{seed}.csv'.format(seed=seed)
  b2_baseline_trace_path = '/home/cpsgroup/SafeBench/safebench/predictive_monitor_trace_new_stl/trace_file_close_loop_simulation/behavior_type_2/no_lstm/eval_agent_behavior_type_2_scenario_3_route_7_episode_30_seed_{seed}.csv'.format(seed=seed)
  b2_df_propose = pd.read_csv(b2_propose_trace_path, index_col=0)
  b2_df_baseline = pd.read_csv(b2_baseline_trace_path, index_col=0)
  b2_propose_trace = b2_df_propose[b2_df_propose['episode'] == 10][feature_to_draw].to_list()
  b2_baseline_trace = b2_df_baseline[b2_df_baseline['episode'] == 10][feature_to_draw].to_list()

  time_list_b0_proposed = range(len(b0_propose_trace))
  time_list_b0_baseline = range(len(b0_baseline_trace))
  #time_list_b0_v1 = range(len(b0_v1_trace))
  total_time_b0 = range(max([len(b0_propose_trace),len(b0_baseline_trace)]))
  time_list_b2_proposed = range(len(b2_propose_trace))
  time_list_b2_baseline = range(len(b2_baseline_trace))
  #time_list_b2_v1 = range(len(b2_v1_trace))
  total_time_b2 = range(max([len(b2_propose_trace), len(b2_baseline_trace)]))

  fig = plt.figure(figsize=(13, 13))
  fontsize=10
  y_tick_list = [-6.0, 6.0]
  #y_tick_list = range(10,2)
  # b0
  ax1 = fig.add_subplot(211)
  ax1.plot(time_list_b0_baseline, b0_baseline_trace, label='Baseline', linestyle='--', color='steelblue')
  ax1.plot(time_list_b0_proposed, b0_propose_trace, label='Proposed', linestyle='-', color='salmon')
  #ax1.plot(time_list_b0_v1, b0_v1_trace, label='para_v1', linestyle='-.', color='green', alpha=0.8)
  ax1.plot(total_time_b0, [-6.0]*len(total_time_b0), color='black', linestyle='--',dashes=[25, 5],linewidth=1)
  ax1.plot(total_time_b0, [6.0]*len(total_time_b0), color='black', linestyle='--',dashes=[25, 5],linewidth=1)
  ax1.set_title('Time',y=-0.15,fontsize=fontsize, loc='right')
  ax1.set_xlim(xmin=0)
  plt.ylabel(ylabel,fontsize=fontsize)
  plt.xlabel('(a) Behavior type: Cautious',fontsize=fontsize)
  plt.legend(fontsize=fontsize+6,loc = 'upper right', ncol=2)
  plt.xticks(fontsize=fontsize)
  plt.yticks(y_tick_list,fontsize=fontsize)
  # b2
  ax2 = fig.add_subplot(212)
  ax2.plot(time_list_b2_baseline, b2_baseline_trace, label='Baseline', linestyle='--', color='steelblue')
  ax2.plot(time_list_b2_proposed, b2_propose_trace, label='Proposed', linestyle='-', color='salmon')
  #ax2.plot(time_list_b2_v1, b2_v1_trace, label='para_v1', linestyle='-.', color='green', alpha=0.8)
  ax2.plot(total_time_b2, [6.0]*len(total_time_b2), color='black', linestyle='--',dashes=[25, 5],linewidth=1)
  ax2.plot(total_time_b2, [-6.0] * len(total_time_b2), color='black', linestyle='--', dashes=[25, 5], linewidth=1)
  ax2.set_title('Time',y=-0.15,fontsize=fontsize, loc='right')
  ax2.set_xlim(xmin=0)
  plt.ylabel(ylabel,fontsize=fontsize)
  plt.xlabel('(b) Behavior type: Aggressive',fontsize=fontsize)
  plt.legend(fontsize=fontsize+6,loc = 'upper right', ncol=2)
  plt.xticks(fontsize=fontsize)
  plt.yticks(y_tick_list,fontsize=fontsize)

  plt.savefig('{fig_folder}/{figure_name}_{feature_to_draw_2}_seed_{seed}.png'.format(fig_folder=figure_folder,
                                                                                      feature_to_draw_2=feature_to_draw,
                                                                                      figure_name=figure_name,seed=seed))
  plt.show()
#




# draw Fig.6: compare number of hazards, average speed/acc/distance
def draw_total_hazard_num_bar(hazard_df_b0, hazard_df_b2, save_folder):
  metric_name = 'average_distance' #'average_total', 'average_speed', 'average_acc'
  title = 'Average distance' #'Average number of hazard', 'Average speed', 'Average acceleration'
  figure_name = 'average_distance' #'average_hazards_number', 'average_speed', 'average_acc'
  pre_alert_time_with_monitor_b0_df = hazard_df_b0[
    (hazard_df_b0['control_type'] == 'lstm_with_monitor')][metric_name]
  pre_alert_time_no_monitor_b0_df = hazard_df_b0[(hazard_df_b0['control_type'] == 'no_lstm')][metric_name]
  pre_alert_time_with_monitor_b2_df = hazard_df_b2[
    (hazard_df_b2['control_type'] == 'lstm_with_monitor')][metric_name]
  pre_alert_time_no_monitor_b2_df = hazard_df_b2[(hazard_df_b2['control_type'] == 'no_lstm')][metric_name]

  lstm_with_monitor_mean_list = []
  lstm_no_monitor_mean_list = []
  lstm_with_monitor_std_list = []
  lstm_no_monitor_std_list = []

  lstm_with_monitor_mean_list.append(pre_alert_time_with_monitor_b0_df.mean())
  lstm_with_monitor_mean_list.append(pre_alert_time_with_monitor_b2_df.mean())

  lstm_no_monitor_mean_list.append(pre_alert_time_no_monitor_b0_df.mean())
  lstm_no_monitor_mean_list.append(pre_alert_time_no_monitor_b2_df.mean())

  lstm_with_monitor_std_list.append(pre_alert_time_with_monitor_b0_df.std())
  lstm_with_monitor_std_list.append(pre_alert_time_with_monitor_b2_df.std())

  lstm_no_monitor_std_list.append(pre_alert_time_no_monitor_b0_df.std())
  lstm_no_monitor_std_list.append(pre_alert_time_no_monitor_b2_df.std())

  size = 2
  group_num = 2
  tick_step = 1
  group_gap = 0.2
  group_width = tick_step - group_gap
  bar_span = group_width / group_num
  x = np.arange(size) * tick_step
  ticks = x

  a = lstm_with_monitor_mean_list
  a_SD = lstm_with_monitor_std_list
  b = lstm_no_monitor_mean_list
  b_SD = lstm_no_monitor_std_list

  fontsize = 50
  total_width, n = 0.8, 2
  width = total_width / n
  x = x - (total_width - width) / 2
  labels = ['Cautious', 'Aggressive']
  fig = plt.figure(figsize=(40, 18))
  plt.barh(x + width, b, width, xerr=b_SD, label='Baseline', color='steelblue', alpha=0.7)
  plt.barh(x, a, width, xerr=a_SD, label='Proposed', color='#E57200', alpha=0.8)
  plt.xlabel('{title}'.format(title=title), fontsize=fontsize + 20)
  plt.yticks(ticks, labels, fontsize=fontsize)
  plt.xticks(fontsize=fontsize)
  plt.legend(fontsize=fontsize + 20, loc='upper right')
  plt.savefig('{fig_folder}/{figure_name}'.format(fig_folder=save_folder, figure_name=figure_name))
  plt.show()
  return

# draw Fig.4~8: loss values with different dropout types and rates for each behavior agent
def draw_loss_value(file_path, behavior_type, save_path, loss_func):
  df = pd.read_csv(file_path, index_col=0)
  dropout_type_list = range(1,5)
  marker_list = ['s', 'o', '+', '*']
  color_list = ['tomato', 'darkorange', 'seagreen', 'teal']
  dropout_name_list = ['Bernoulli Dropout', 'Bernoulli dropConnect', 'Gaussian Dropout', 'Gaussian dropConnect']
  line_style_list = ['-', '--', '-.', ':']
  ms=4
  size = 18
  plt.figure(figsize=(10, 6))
  # if behavior_type=='adult':
  #   plt.figure(figsize=(10, 5))
  for dt in dropout_type_list:
    print(dt)
    loss_value_list = df[(df['dropout_type']==dt)]['loss_value'].to_list()
    x_value_list = df[(df['dropout_type']==dt)]['dropout_rate'].to_list()
    print(loss_value_list)
    print(x_value_list)
    plt.plot(x_value_list, loss_value_list, marker=marker_list[dt-1], color=color_list[dt-1], ms=ms, linestyle=line_style_list[dt-1],label='{dt}'.format(dt=dropout_name_list[dt-1]))
  plt.xlabel('Dropout rate',fontdict={'size':size})
  plt.ylabel('Loss',fontdict={'size':size})
  plt.legend(prop={'size':size-2})
  plt.xticks(fontsize=size)
  plt.yticks(fontsize=size)
  plt.savefig('{save_path}/{behavior}_loss_{loss_func}.png'.format(save_path=save_path, behavior=behavior_type, loss_func=loss_func))
  plt.show()
  return

# loss_func = 'LACC'
# b0_folder = '/home/cpsgroup/predictive_monitor_new_stl/with_dropout_new_stl/stl_6/stl_6_beta_0.5/select_dropout_with_lossfunc_{loss_func}/results'.format(loss_func=loss_func)
# b0_path = '{folder}/behavior_type_0_f_17/dropout_choice_result_train_type_4_train_rate_0.9_lr_0.001_e_1.csv'.format(folder=b0_folder)
# b2_folder = '/home/cpsgroup/predictive_monitor_new_stl/with_dropout_new_stl/stl_6/stl_6_beta_0.5/select_dropout_with_lossfunc_{loss_func}/results'.format(loss_func=loss_func)
# b2_path = '{folder}/behavior_type_2_f_17/dropout_choice_result_train_type_4_train_rate_0.9_lr_0.001_e_1.csv'.format(folder=b2_folder)
# save_path = '/home/cpsgroup/predictive_monitor_new_stl/Loss_choice_figure/stl_6'
# draw_loss_value(b0_path, 'cautious', save_path, loss_func)
# draw_loss_value(b2_path, 'aggressive', save_path, loss_func)