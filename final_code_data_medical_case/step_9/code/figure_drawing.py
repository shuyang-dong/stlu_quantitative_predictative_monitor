# -*- coding: utf-8 -*-

from google.colab import files
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
from torch.utils.tensorboard import SummaryWriter
import json
import random
import sys


# draw Fig.5: cgm trace compare
folder = '/content/drive/MyDrive/ColabNotebooks/Medical_case/medical_case_pipeline/results/all_patient/10_patient_7_day'
figure_folder = '/content/drive/MyDrive/ColabNotebooks/Medical_case/medical_case_pipeline/results/all_patient/10_patient_7_day/patient_trace_figure/0812/patient_trace'
patient_id_num_list = ["%03d" % x for x in range(1,11)]
lstm_type_list = ['lstm_with_monitor', 'no_lstm']
patient_type_list =['adolescent','adult','child']
control_param = '15_0'
for id in range(1):
  adult_id = patient_id_num_list[4] # adult #5
  child_id = patient_id_num_list[4] # child #5
  adolescent_id = patient_id_num_list[3] # adolescent #4
  print(adult_id)
  patient_id_dict = {'adult':id, 'child':id, 'adolescent':id}
  adult_propose_trace_path = '{folder}/adult_patient/10_patient_7_day_{control_param}/lstm_with_monitor/patient_trace_adult#{id}_lstm_with_monitor.csv'.format(folder=folder, id=adult_id, control_param=control_param)
  adult_baseline_trace_path = '{folder}/no_lstm/patient_trace_adult#{id}_no_lstm.csv'.format(folder=folder, id=adult_id)
  all_length = len(pd.read_csv(adult_propose_trace_path, index_col=0))
  adult_propose_trace = pd.read_csv(adult_propose_trace_path, index_col=0)['CGM'].to_list()[:int(all_length*3/7)]
  adult_baseline_trace = pd.read_csv(adult_baseline_trace_path, index_col=0)['CGM'].to_list()[:int(all_length*3/7)]

  adolescent_propose_trace_path = '{folder}/adolescent_patient/10_patient_7_day_{control_param}/lstm_with_monitor/patient_trace_adolescent#{id}_lstm_with_monitor.csv'.format(folder=folder, id=adolescent_id, control_param=control_param)
  adolescent_baseline_trace_path = '{folder}/no_lstm/patient_trace_adolescent#{id}_no_lstm.csv'.format(folder=folder, id=adolescent_id)
  adolescent_propose_trace = pd.read_csv(adolescent_propose_trace_path, index_col=0)['CGM'].to_list()[:int(all_length*3/7)]
  adolescent_baseline_trace = pd.read_csv(adolescent_baseline_trace_path, index_col=0)['CGM'].to_list()[:int(all_length*3/7)]

  child_propose_trace_path = '{folder}/child_patient/10_patient_7_day_{control_param}/lstm_with_monitor/patient_trace_child#{id}_lstm_with_monitor.csv'.format(folder=folder, id=child_id, control_param=control_param)
  child_baseline_trace_path = '{folder}/no_lstm/patient_trace_child#{id}_no_lstm.csv'.format(folder=folder, id=child_id)
  child_propose_trace = pd.read_csv(child_propose_trace_path, index_col=0)['CGM'].to_list()[:int(all_length*3/7)]
  child_baseline_trace = pd.read_csv(child_baseline_trace_path, index_col=0)['CGM'].to_list()[:int(all_length*3/7)]
  time_list = range(len(adult_propose_trace))

  fig = plt.figure(figsize=(13, 15))  
  fontsize=15
  y_tick_list = [30, 70, 180, 220]
  # adult
  ax1 = fig.add_subplot(311)
  ax1.plot(time_list, adult_baseline_trace, label='Baseline', linestyle='--', color='steelblue')
  ax1.plot(time_list, adult_propose_trace, label='Proposed', linestyle='-', color='salmon') 
  ax1.plot(time_list, [70]*len(time_list), color='black', linestyle='--',dashes=[25, 5],linewidth=1)
  ax1.plot(time_list, [180]*len(time_list), color='black', linestyle='--',dashes=[25, 5],linewidth=1)
  ax1.set_title('Time',y=-0.15,fontsize=fontsize, loc='right')  
  ax1.set_xlim(xmin=0)
  plt.ylabel('Blood Glucose (mg/dL)',fontsize=fontsize)
  plt.xlabel('(a) adult_{id}'.format(id=adult_id),fontsize=fontsize)
  plt.legend(fontsize=fontsize+6,loc = 'upper right', ncol=2)
  plt.xticks(fontsize=fontsize)
  plt.yticks(y_tick_list,fontsize=fontsize)
  # adolescent
  ax2 = fig.add_subplot(312)
  ax2.plot(time_list, adolescent_baseline_trace, label='Baseline'.format(id=adolescent_id), linestyle='--', color='steelblue')
  ax2.plot(time_list, adolescent_propose_trace, label='Proposed'.format(id=adolescent_id), linestyle='-', color='salmon')
  ax2.plot(time_list, [70]*len(time_list), color='black', linestyle='--',dashes=[25, 5],linewidth=1)
  ax2.plot(time_list, [180]*len(time_list), color='black', linestyle='--',dashes=[25, 5],linewidth=1)
  ax2.set_title('Time',y=-0.15,fontsize=fontsize, loc='right')
  ax2.set_xlim(xmin=0)
  plt.ylabel('Blood Glucose (mg/dL)',fontsize=fontsize)
  plt.xlabel('(b) adolescent_{id}'.format(id=adolescent_id),fontsize=fontsize)
  plt.legend(fontsize=fontsize+6,loc = 'upper right', ncol=2)
  plt.xticks(fontsize=fontsize)
  plt.yticks(y_tick_list,fontsize=fontsize)
  # child
  ax3 = fig.add_subplot(313)
  ax3.plot(time_list, child_baseline_trace, label='Baseline'.format(id=child_id), linestyle='--', color='steelblue')
  ax3.plot(time_list, child_propose_trace, label='Proposed'.format(id=child_id), linestyle='-', color='salmon')
  ax3.plot(time_list, [70]*len(time_list), color='black', linestyle='--',dashes=[25, 5],linewidth=1)
  ax3.plot(time_list, [180]*len(time_list), color='black', linestyle='--',dashes=[25, 5],linewidth=1)
  ax3.set_title('Time',y=-0.15,fontsize=fontsize, loc='right')
  ax3.set_xlim(xmin=0)
  plt.ylabel('Blood Glucose (mg/dL)',fontsize=fontsize)
  plt.xlabel('(c) child_{id}'.format(id=child_id),fontsize=fontsize)
  plt.legend(fontsize=fontsize+6,loc = 'upper right', ncol=2)
  plt.xticks(fontsize=fontsize)
  plt.yticks(y_tick_list,fontsize=fontsize)
  plt.savefig('{fig_folder}/cgm_trace_compare.png'.format(fig_folder=figure_folder))
  plt.show()

# draw Fig.6: compare number of hazards
def draw_total_hazard_num_bar(hazard_df_adult, hazard_df_child, hazard_df_adolescent, save_folder):
  pre_alert_time_with_monitor_adult_df = hazard_df_adult[
    (hazard_df_adult['control_type'] == 'lstm_with_monitor') & (hazard_df_adult['control_parameter'] == '15_0')][
    'total_hazard_num']
  pre_alert_time_no_monitor_adult_df = hazard_df_adult[(hazard_df_adult['control_type'] == 'no_lstm')][
    'total_hazard_num']
  pre_alert_time_with_monitor_child_df = hazard_df_child[
    (hazard_df_child['control_type'] == 'lstm_with_monitor') & (hazard_df_child['control_parameter'] == '15_0')][
    'total_hazard_num']
  pre_alert_time_no_monitor_child_df = hazard_df_child[(hazard_df_child['control_type'] == 'no_lstm')][
    'total_hazard_num']
  pre_alert_time_with_monitor_adolescent_df = hazard_df_adolescent[
    (hazard_df_adolescent['control_type'] == 'lstm_with_monitor') & (
              hazard_df_adolescent['control_parameter'] == '15_0')]['total_hazard_num']
  pre_alert_time_no_monitor_adolescent_df = hazard_df_adolescent[(hazard_df_adolescent['control_type'] == 'no_lstm')][
    'total_hazard_num']

  lstm_with_monitor_mean_list = []
  lstm_no_monitor_mean_list = []
  lstm_with_monitor_std_list = []
  lstm_no_monitor_std_list = []

  lstm_with_monitor_mean_list.append(pre_alert_time_with_monitor_child_df.mean())
  lstm_with_monitor_mean_list.append(pre_alert_time_with_monitor_adolescent_df.mean())
  lstm_with_monitor_mean_list.append(pre_alert_time_with_monitor_adult_df.mean())

  lstm_no_monitor_mean_list.append(pre_alert_time_no_monitor_child_df.mean())
  lstm_no_monitor_mean_list.append(pre_alert_time_no_monitor_adolescent_df.mean())
  lstm_no_monitor_mean_list.append(pre_alert_time_no_monitor_adult_df.mean())

  lstm_with_monitor_std_list.append(pre_alert_time_with_monitor_child_df.std())
  lstm_with_monitor_std_list.append(pre_alert_time_with_monitor_adolescent_df.std())
  lstm_with_monitor_std_list.append(pre_alert_time_with_monitor_adult_df.std())

  lstm_no_monitor_std_list.append(pre_alert_time_no_monitor_child_df.std())
  lstm_no_monitor_std_list.append(pre_alert_time_no_monitor_adolescent_df.std())
  lstm_no_monitor_std_list.append(pre_alert_time_no_monitor_adult_df.std())

  size = 3
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
  labels = ['Child', 'Adolescent', 'Adult']
  fig = plt.figure(figsize=(40, 13))
  plt.barh(x + width, b, width, xerr=b_SD, label='Baseline', color='steelblue', alpha=0.7)
  plt.barh(x, a, width, xerr=a_SD, label='Proposed', color='#E57200', alpha=0.8)
  plt.xlabel('Number of hazard', fontsize=fontsize + 20)
  plt.yticks(ticks, labels, fontsize=fontsize)
  plt.xticks(fontsize=fontsize)
  plt.legend(fontsize=fontsize + 20, loc='upper right')
  plt.savefig('{fig_folder}/hazards_number.png'.format(fig_folder=save_folder))
  plt.show()
  return


folder = '/content/drive/MyDrive/ColabNotebooks/Medical_case/medical_case_pipeline/results/all_patient/10_patient_7_day'
save_folder = '/content/drive/MyDrive/ColabNotebooks/Medical_case/statistical_test'
adult_path = '{folder}/controller_metrics_result_adult.csv'.format(folder=folder)
child_path = '{folder}/controller_metrics_result_child.csv'.format(folder=folder)
adolescent_path = '{folder}/controller_metrics_result_adolescent.csv'.format(folder=folder)
hazard_df_adult = pd.read_csv(adult_path, index_col=0)
hazard_df_child = pd.read_csv(child_path, index_col=0)
hazard_df_adolescent = pd.read_csv(adolescent_path, index_col=0)
draw_total_hazard_num_bar(hazard_df_adult, hazard_df_child, hazard_df_adolescent, save_folder)


# draw Fig.4~8: loss values with different dropout types and rates for each type of patient
def draw_loss_value(file_path, patient_type, save_path):
  df = pd.read_csv(file_path, index_col=0)
  dropout_type_list = range(1,5)
  marker_list = ['s', 'o', '+', '*']
  color_list = ['tomato', 'darkorange', 'seagreen', 'teal']
  dropout_name_list = ['Bernoulli Dropout', 'Bernoulli dropConnect', 'Gaussian Dropout', 'Gaussian dropConnect']
  line_style_list = ['-', '--', '-.', ':']
  ms=4
  size = 18
  plt.figure(figsize=(10, 6))
  if patient_type=='adult':
    plt.figure(figsize=(10, 5))
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
  plt.savefig('{save_path}/{patient}_loss'.format(save_path=save_path, patient=patient_type))
  plt.show()
  return

folder = '/content/drive/MyDrive/ColabNotebooks/Medical_case/with_dropout/select_dropout_with_lossfunc_new/figure_data/data/choose_dropout_LQT'
ado_path = '{folder}/dropout_choice_result_train_type_4_train_rate_0.9_lr_0.001_e_50_adolescent.csv'.format(folder=folder)
child_path = '{folder}/dropout_choice_result_train_type_4_train_rate_0.9_lr_0.01_e_50_child.csv'.format(folder=folder)
adult_path = '{folder}/dropout_choice_result_train_type_4_train_rate_0.9_lr_0.01_e_50_adult.csv'.format(folder=folder)
save_path = '/content/drive/MyDrive/ColabNotebooks/Medical_case/with_dropout/select_dropout_with_lossfunc_new/figure_data'
draw_loss_value(ado_path, 'adolescent', save_path)
draw_loss_value(child_path, 'child', save_path)
draw_loss_value(adult_path, 'adult', save_path)