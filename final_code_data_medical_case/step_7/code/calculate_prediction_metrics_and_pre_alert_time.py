# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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
from torch.utils.tensorboard import SummaryWriter

import json
import random
import sys
sys.path.append('/content/drive/MyDrive/ColabNotebooks')
import ustlmonitor as ustl
import confidencelevel
import argparse
torch.set_printoptions(precision=4,sci_mode=False)


# STL formula for checking CGM
def requirement_func_always_BG_in_range(signal, trace_len, conf, lower_BG=70, upper_BG=180, func='monitor'):
    # signal: CGM trace
    # Keep BG in range
    # G[0,t](CGM > lower_BG & CGM < upper_BG)
    # STL: G[0,10](signal>lower_BG) and (signal<upper_BG)
    # convert to:
    # G[0,10] (signal>lower_BG) and neg(signal>upper_BG)
    threshold_1 = lower_BG
    threshold_2 = upper_BG
    varphi_1 = (("mu", signal), [threshold_1, conf])
    varphi_2 = ((("neg", 0), (("mu", signal), [threshold_2, conf])))
    varphi_3 = (("always", (0, trace_len-1)), (("and", 0), (varphi_1, varphi_2)))

    varphi_1_1 = (("mu", signal), threshold_1)
    varphi_2_1 = ((("neg", 0), (("mu", signal), threshold_2)))
    varphi_3_1 = (("always", (0, trace_len-1)), (("and", 0), (varphi_1_1, varphi_2_1)))
    if func=='monitor':
        return ustl.umonitor(varphi_3, 0)
    else:
        return (varphi_3_1, 0)

def requirement_func_always_BG_not_hypo(signal, trace_len, conf, lower_BG=70, func='monitor'):
    # signal: CGM trace
    # Keep BG in range
    # G[0,t](CGM > lower_BG)
    # STL: G[0,10](signal>lower_BG)
    # convert to:
    # G[0,10] (signal>lower_BG)
    threshold_1 = lower_BG
    varphi_1 = (("mu", signal), [threshold_1, conf])
    varphi_3 = (("always", (0, trace_len-1)), varphi_1)

    varphi_1_1 = (("mu", signal), threshold_1)
    varphi_3_1 = (("always", (0, trace_len-1)), varphi_1_1)
    if func=='monitor':
        return ustl.umonitor(varphi_3, 0)
    else:
        return (varphi_3_1, 0)

def requirement_func_always_BG_not_hyper(signal, trace_len, conf, upper_BG=180, func='monitor'):
    # signal: CGM trace
    # Keep BG in range
    # G[0,t](CGM < upper_BG)
    # STL: G[0,10](signal<upper_BG)
    # convert to:
    # G[0,10] neg(signal>upper_BG)
    threshold_2 = upper_BG
    varphi_2 = ((("neg", 0), (("mu", signal), [threshold_2, conf])))
    varphi_3 = (("always", (0, trace_len-1)), varphi_2)

    varphi_2_1 = ((("neg", 0), (("mu", signal), threshold_2)))
    varphi_3_1 = (("always", (0, trace_len-1)), varphi_2_1)
    if func=='monitor':
        return ustl.umonitor(varphi_3, 0)
    else:
        return (varphi_3_1, 0)

def requirement_func_check_rho_low(signal, trace_len, conf=0.95, func='monitor'):
    # signal: CGM trace
    # Keep BG in range
    # G[0,t](rho_low < 0)
    # STL: G[0,1]neg(signal>0)
    threshold_1 = 0
    varphi_1 = ((("neg", 0), (("mu", signal), [threshold_1, conf])))
    varphi_3 = (("always", (0, trace_len-1)), varphi_1)

    varphi_1_1 = ((("neg", 0), (("mu", signal), threshold_1)))
    varphi_3_1 = (("always", (0, trace_len-1)), varphi_1_1)
    if func=='monitor':
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
        satisfaction_type = 1 # weak satisfaction, one boundary<0, another>0
    elif (lower>0 and upper>0) :
        satisfaction_type = 2  # strong satisfaction, both boundary>0
    else:
        satisfaction_type = 3  # violation, both boundary<=0
    return satisfaction_type

def calculate_consecutive_prediction_metrics_total_STLU(segment_df, rho_real_col_name, rho_pred_col_name, hazard_type, method):
  rho_low_pred_list = []
  rho_low_real_list = []
  rho_check_rho_low_pred_list = []
  rho_check_rho_low_real_list = []
  satisfaction_type_real = []
  satisfaction_type_predict = []
  if_warning_real_list = []
  if_warning_predict_list = []
  for index, row in segment_df.iterrows():
    rho_set_CGM = [float(x) for x in row[rho_pred_col_name]]
    rho_orig_CGM = [float(x) for x in row[rho_real_col_name]]
    rho_low_pred_list.append(rho_set_CGM[0])
    rho_low_real_list.append(rho_orig_CGM[0])
  trace_len = 3
  for i in range(len(rho_low_pred_list)-(trace_len-1)):
    rho_low_pred_trace = torch.Tensor(rho_low_pred_list[i:i+trace_len])
    rho_low_real_trace = torch.Tensor(rho_low_real_list[i:i+trace_len])
    trace_pred = torch.stack((rho_low_pred_trace, torch.zeros(rho_low_pred_trace.size())), dim=-1)
    trace_real = torch.stack((rho_low_real_trace, torch.zeros(rho_low_real_trace.size())), dim=-1)
    rho_check_rho_low_pred = requirement_func_check_rho_low(trace_pred, trace_len=trace_len, conf=0.95, func='monitor')
    rho_check_rho_low_real = requirement_func_check_rho_low(trace_real, trace_len=trace_len, conf=0.95, func='monitor')
    rho_check_rho_low_pred_list.append(rho_check_rho_low_pred.tolist())
    rho_check_rho_low_real_list.append(rho_check_rho_low_real.tolist())
    st_central_real = get_satisfaction_type_one_segment_STLU(rho_check_rho_low_real)
    st_central_predict = get_satisfaction_type_one_segment_STLU(rho_check_rho_low_pred)
    satisfaction_type_predict.append(st_central_predict)
    satisfaction_type_real.append(st_central_real)

    if (st_central_real==2):
            if_warning_orig = 1 
    else:
            if_warning_orig = 0  
    if (st_central_predict==2):
            if_warning_predict = 1 
    else:
            if_warning_predict = 0  
    if_warning_real_list.append(if_warning_orig)
    if_warning_predict_list.append(if_warning_predict)
  total_num = len(if_warning_real_list)
  if_warning_real_sum = sum(if_warning_real_list)
  real_warning_percentage = if_warning_real_sum/total_num
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
  accuracy = (TP + TN) / total_num if total_num!=0 else None

  percision = TP / (TP + FP) if (TP + FP) != 0 else None
  recall = TP / (TP + FN) if (TP + FN) != 0 else None
  if percision!=None and recall!=None and (percision + recall)!=0:
    F1_score = 2 * percision * recall / (percision + recall)
  else:
    F1_score = None

  results_dict = {'hazard_type': hazard_type, 'method':method,'metric_type':'consecutive','TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'TPR': TPR, 'TNR': TNR, 'FPR': FPR, 'FNR': FNR,
                    'accuracy': accuracy, 'F1_score': F1_score, 'real_warning_percentage':real_warning_percentage}
  print('Metrics STLU if_warning_predict_consecutive: ', results_dict)

  return results_dict

# calculate metrics for total, hyper and hypo separately
def calculate_metrics_for_each_hazard_type(segment_output_file_path, hazard_type='hyper', method='flowpipe_mean'):
  predict_len = 10
  segment_df = pd.read_csv(segment_output_file_path, index_col=0)
  if hazard_type=='hyper':
    requirement_func = requirement_func_always_BG_not_hyper
  if hazard_type=='hypo':
    requirement_func = requirement_func_always_BG_not_hypo
  if hazard_type=='total':
    requirement_func = requirement_func_always_BG_in_range
  rho_real_list = []
  rho_pred_list = []
  satisfaction_type_real = []
  satisfaction_type_predict = []
  if_warning_real_list = []
  if_warning_predict_list = []
  for index, row in segment_df.iterrows():
    predict_trace_mean = torch.Tensor([float(x) for x in row['CGM_predicted_mean'][1:-1].split(',')])
    predict_trace_std = torch.Tensor([float(x) for x in row['CGM_predicted_std'][1:-1].split(',')])
    real_trace = torch.Tensor([float(x) for x in row['each_real_CGM_in_batch'][1:-1].split(',')][-predict_len:])
    if method=='flowpipe_mean':
      trace_pred = torch.stack((predict_trace_mean, torch.zeros(predict_trace_mean.size())), dim=-1)
      trace_real = torch.stack((real_trace, torch.zeros(real_trace.size())), dim=-1)
    elif method=='flowpipe_rho':
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
    
    if (st_central_real==1 or st_central_real==3):
            if_warning_orig = 1 # warning for st=violation & weak satisfaction
    else:
            if_warning_orig = 0  # no warning for st=strong satisfy
    if (st_central_predict==1 or st_central_predict==3):
            if_warning_predict = 1 # warning for st=violation & weak satisfaction
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

  warning_label_df = pd.DataFrame(segment_df,columns=['if_warning_real_hazard', 'if_warning_predict_hazard'])
  total_num = len(warning_label_df)
  if_warning_real_sum = warning_label_df['if_warning_real_hazard'].sum()
  real_warning_percentage = if_warning_real_sum/total_num
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
  accuracy = (TP + TN) / total_num if total_num!=0 else None

  percision = TP / (TP + FP) if (TP + FP) != 0 else None
  recall = TP / (TP + FN) if (TP + FN) != 0 else None
  if percision!=None and recall!=None and (percision + recall)!=0:
    F1_score = 2 * percision * recall / (percision + recall)
  else:
    F1_score = None

  normal_results_dict = {'hazard_type': hazard_type, 'method':method,'metric_type':'not_consecutive','TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'TPR': TPR, 'TNR': TNR, 'FPR': FPR, 'FNR': FNR,
                    'accuracy': accuracy, 'F1_score': F1_score, 'real_warning_percentage':real_warning_percentage}
  print('Metrics STLU if_warning_predict_hazard: ', normal_results_dict)
  consecutive_results_dict = {}
  if method=='flowpipe_rho':
    consecutive_results_dict = calculate_consecutive_prediction_metrics_total_STLU(segment_df, 'rho_real_hazard', 'rho_pred_hazard',hazard_type, method)
  else:
    consecutive_results_dict = {}
  if hazard_type=='total' and method=='flowpipe_rho':
    segment_df['rho_orig_CGM'] = rho_real_list
    segment_df['rho_set_CGM'] = rho_pred_list
    segment_df['satisfaction_type_real_CGM'] = satisfaction_type_real
    segment_df['satisfaction_type_predict_CGM'] = satisfaction_type_predict
    segment_df['if_warning_real'] = if_warning_real_list
    segment_df['if_warning_predict'] = if_warning_predict_list

  segment_df.to_csv(segment_output_file_path)

  return normal_results_dict, consecutive_results_dict

hazard_type_list = ['total', 'hyper', 'hypo']
method_list = ['flowpipe_rho','flowpipe_mean']
patient_type_list = ['adult', 'adolescent', 'child']
segment_folder_LQT = '/content/drive/MyDrive/ColabNotebooks/Medical_case/with_dropout/select_dropout_with_lossfunc_new/figure_data/data/segment_file_test_set_LQT'
adult_path_LQT = '{segment_folder}/adult_patient/segment_results_b_1024_seqlen_20_stepback_10_e_1_lr_0.01_f_8_conf_0.95_dt_2_dr_0.8.csv'.format(segment_folder=segment_folder_LQT)
child_path_LQT = '{segment_folder}/child_patient/segment_results_b_1024_seqlen_20_stepback_10_e_1_lr_0.01_f_8_conf_0.95_dt_3_dr_0.9.csv'.format(segment_folder=segment_folder_LQT)
adolescent_path_LQT = '{segment_folder}/adolescent_patient/segment_results_b_1024_seqlen_20_stepback_10_e_1_lr_0.001_f_8_conf_0.95_dt_2_dr_0.9.csv'.format(segment_folder=segment_folder_LQT)
path_dict_LQT={'adult':adult_path_LQT, 'child':child_path_LQT, 'adolescent':adolescent_path_LQT}
col = ['patient_type','hazard_type', 'method','metric_type','TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR',
                    'accuracy', 'F1_score', 'real_warning_percentage']
prediction_metric_result_df = pd.DataFrame(columns=col)
prediction_metric_result_summary_path = '{folder}/prediction_metric_summary_LQT.csv'.format(folder=segment_folder_LQT)
for patient_type in patient_type_list:
  print('patient_type: ', patient_type)
  segment_file_path = path_dict_LQT[patient_type]
  for hazard_type in hazard_type_list:
    print('hazard_type: ', hazard_type)
    for method in method_list:
      normal_results_dict, consecutive_results_dict = calculate_metrics_for_each_hazard_type(segment_file_path, hazard_type, method)
      normal_results_dict['patient_type']=patient_type
      consecutive_results_dict['patient_type']=patient_type
      prediction_metric_result_df=prediction_metric_result_df.append(normal_results_dict, ignore_index=True)
      if method=='flowpipe_rho':
        prediction_metric_result_df=prediction_metric_result_df.append(consecutive_results_dict,ignore_index=True)
prediction_metric_result_df.to_csv(prediction_metric_result_summary_path)



def get_ground_truth_whole_trace(segment_file_path):
  segment_df = pd.read_csv(segment_file_path, index_col=0)
  CGM_list = []
  CGM_df = pd.DataFrame()
  for index, row in segment_df.iterrows():
    segment_real_CGM = [float(x) for x in row['each_real_CGM_in_batch'][1:-1].split(',')]
    if index==0:
      CGM_list = segment_real_CGM
    else:
      CGM_list.append(segment_real_CGM[-1])

  CGM_df['real_CGM'] = CGM_list
  return CGM_df

def mark_hazard_id_for_real_CGM(patient_trace_df):
  hazard_id_list = []
  hazard_type_list = []
  hazard_time_list = []
  bg_list = patient_trace_df['real_CGM'].tolist()
  hazard_id_count = 0
  hazard_type = 0
  hazard_time = 0
  last_hazard_end_time = 0
  for i in range(len(bg_list)):
    current_bg = bg_list[i]
    if current_bg<=50:
        hazard_type = 4 # severe low
    elif 50<current_bg<70:
        hazard_type = 2 # mild low
    elif 180<current_bg<250:
        hazard_type = 3 # mild high
    elif 250<=current_bg:
        hazard_type = 5 # severe high
    else:
        hazard_type = 0 # no hazard
    hazard_type_list.append(hazard_type)
    if i==0:
      if hazard_type != 0:
        hazard_id_count += 1
        hazard_id_list.append(hazard_id_count)
        last_hazard_end_time = i
      else:
        hazard_id_list.append(0)
    else:
      last_hazard_type = hazard_type_list[i-1]
      if hazard_type!=last_hazard_type and hazard_type!=0:
        if i-last_hazard_end_time<=10: # within 30min
          hazard_id_list.append(0)
          last_hazard_end_time = i
        else:
          hazard_id_count += 1
          hazard_id_list.append(hazard_id_count)
          last_hazard_end_time = i
      elif hazard_type==0:
        hazard_id_list.append(0)
      else:
        hazard_id_list.append(0)
        last_hazard_end_time = i
        
  patient_trace_df['hazard_type'] = hazard_type_list
  patient_trace_df['hazard_id'] = hazard_id_list

  hazard_id_index_list = patient_trace_df[patient_trace_df['hazard_id']!=0].index.to_list()
  final_hazard_type_list = []
  for j in range(len(bg_list)):
    if j in hazard_id_index_list:
      next_index = hazard_id_index_list[hazard_id_index_list.index(j)+1] if hazard_id_index_list.index(j)!=(len(hazard_id_index_list)-1) else len(bg_list)-1
      hazard_time_step_num = len(patient_trace_df[j:next_index])
      max_CGM_this_hazard = max(patient_trace_df[j:next_index][patient_trace_df['hazard_type']!=0]['real_CGM'].to_list())
      min_CGM_this_hazard = min(patient_trace_df[j:next_index][patient_trace_df['hazard_type']!=0]['real_CGM'].to_list())
      hyper_time_this_hazard = len(patient_trace_df[j:next_index][(patient_trace_df['hazard_type']!=0) & (patient_trace_df['real_CGM']>180)])
      hypo_time_this_hazard = len(patient_trace_df[j:next_index][(patient_trace_df['hazard_type']!=0) & (patient_trace_df['real_CGM']<70)])
      if hyper_time_this_hazard>=hypo_time_this_hazard:
        if max_CGM_this_hazard>=250:
          final_hazard_type = 5
        elif 180<=max_CGM_this_hazard<250:
          final_hazard_type = 3
      else:
        if 70>=min_CGM_this_hazard>50:
          final_hazard_type = 2
        elif min_CGM_this_hazard<=50:
          final_hazard_type = 4
      final_hazard_type_list.append(final_hazard_type)
      hazard_time_list.append(hazard_time_step_num)
    else:
      hazard_time_list.append(0)
      final_hazard_type_list.append(0)
  patient_trace_df['hazard_time'] = hazard_time_list
  patient_trace_df['final_hazard_type'] = final_hazard_type_list
  return patient_trace_df

def calculate_pre_alert_time(segment_file_path, real_cgm_trace_path, control_type='lstm_with_monitor', hazard_type='total'):
  segment_df = pd.read_csv(segment_file_path, index_col=0)
  real_CGM_df = get_ground_truth_whole_trace(segment_file_path)
  real_CGM_df = mark_hazard_id_for_real_CGM(real_CGM_df)
  pre_alert_time_list = []
  pre_alert_time = 0
  hazard_index_list = real_CGM_df[(real_CGM_df.hazard_id!=0)].index.to_list()
  earlist_prediction_index_list = []
  # calculate pre-alert time
  for i in range(len(real_CGM_df)):
    if i in hazard_index_list:
      if i>=19:
        check_prediction_segment_index = range(i-19, i-9)
      else: 
        check_prediction_segment_index = range(0, i-9)
      for k in check_prediction_segment_index:
        rho_interval = [float(x) for x in (segment_df['rho_set_CGM'].iloc[k])[1:-1].split(',')]
        mean_trace = [float(x) for x in (segment_df['CGM_predicted_mean'].iloc[k])[1:-1].split(',')]
        if control_type=='lstm_with_monitor':
          if rho_interval[0]<0:
            pre_alert_time = i-9-k
            earlist_prediction_index_list.append(k)
            break
          else: 
            if k==max(check_prediction_segment_index):
              earlist_prediction_index_list.append(0)
            continue
        elif control_type=='lstm_no_monitor':
          if max(mean_trace)>180 or min(mean_trace)<70:
            pre_alert_time = i-9-k
            earlist_prediction_index_list.append(k)
            break
          else: 
            if k==max(check_prediction_segment_index):
              earlist_prediction_index_list.append(0)
            continue
      pre_alert_time_list.append(pre_alert_time)
      pre_alert_time = 0
    else:
      pre_alert_time_list.append(0)
  real_CGM_df['pre_alert_time'] = pre_alert_time_list
  if hazard_type=='total':
    total_hazard_num = len(real_CGM_df[real_CGM_df['hazard_id']!=0])
    average_pre_alert_time = sum(real_CGM_df[real_CGM_df['hazard_id']!=0]['pre_alert_time'].to_list())/total_hazard_num
  elif hazard_type=='hyper':
    hazard_df = real_CGM_df[(real_CGM_df['hazard_id']!=0) & ((real_CGM_df['final_hazard_type']==3) | (real_CGM_df['final_hazard_type']==5))]
    total_hazard_num = len(hazard_df)
    average_pre_alert_time = sum(hazard_df['pre_alert_time'].to_list())/total_hazard_num
  elif hazard_type=='hypo':
    hazard_df = real_CGM_df[(real_CGM_df['hazard_id']!=0) & ((real_CGM_df['final_hazard_type']==2) | (real_CGM_df['final_hazard_type']==4))]
    total_hazard_num = len(hazard_df)
    average_pre_alert_time = sum(hazard_df['pre_alert_time'].to_list())/total_hazard_num
  print('control type: ', control_type, ' total_hazard_num: ', total_hazard_num, ' average_pre_alert_time: ', average_pre_alert_time)
  
  # count number of flowpipes/mean traces that successfully predicts a hazard for one hazard
  total_successful_prediction_index_list_all_hazard = []
  total_successful_prediction_index_list_one_hazard = []
  total_successful_prediction_num = 0
  total_successful_prediction_num_list = []
  for i in range(len(real_CGM_df)):
    if i in hazard_index_list:
      if i>=9:
        check_prediction_segment_index = range(i-19, i-9)
      else: 
        check_prediction_segment_index = range(0, i-9)
      for k in check_prediction_segment_index:
        rho_interval = [float(x) for x in (segment_df['rho_set_CGM'].iloc[k])[1:-1].split(',')]
        mean_trace = [float(x) for x in (segment_df['CGM_predicted_mean'].iloc[k])[1:-1].split(',')]
        if control_type=='lstm_with_monitor':
          if rho_interval[0]<0:
            total_successful_prediction_num += 1
            total_successful_prediction_index_list_one_hazard.append(k)
        elif control_type=='lstm_no_monitor':
          if max(mean_trace)>180 or min(mean_trace)<70:
            total_successful_prediction_num += 1
            total_successful_prediction_index_list_one_hazard.append(k)
      total_successful_prediction_index_list_all_hazard.append(total_successful_prediction_index_list_one_hazard)
      total_successful_prediction_num_list.append(total_successful_prediction_num)
      total_successful_prediction_num = 0
      total_successful_prediction_index_list_one_hazard = []
    else:
      total_successful_prediction_num_list.append(0)
      total_successful_prediction_index_list_all_hazard.append([])
  real_CGM_df['successful_prediction_num'] = total_successful_prediction_num_list
  real_CGM_df['successful_prediction_index'] = total_successful_prediction_index_list_all_hazard
  real_CGM_df.to_csv(real_cgm_trace_path)
  return total_hazard_num, average_pre_alert_time

patient_type_list = ['adolescent','adult', 'child']
control_type_list = ['lstm_with_monitor', 'lstm_no_monitor']
hazard_type_list = ['total', 'hyper', 'hypo']
all_pre_alert_time_path = '{folder}/pre_alert_time_summary_LQT.csv'.format(folder=folder_LQT)
all_pre_alert_time_df = pd.DataFrame(columns=['patient_type','control_type', 'hazard_type' ,'total_hazard_num', 'average_pre_alert_time_step', 'average_pre_alert_time_minute'])
for patient_type in patient_type_list:
  for control_type in control_type_list:
    for hazard_type in hazard_type_list:
      real_cgm_trace_path = '{folder}/real_cgm_trace_with_hazard_lable_{patient_type}_{control_type}.csv'.format(patient_type=patient_type, folder=segment_folder_LQT, control_type=control_type)
      segment_file_path = path_dict_LQT[patient_type]
      total_hazard_num, average_pre_alert_time_step = calculate_pre_alert_time(segment_file_path, real_cgm_trace_path, control_type=control_type, hazard_type=hazard_type)
      average_pre_alert_time_minute = average_pre_alert_time_step*3
      result_dict = {'patient_type':patient_type,'control_type':control_type, 'hazard_type':hazard_type,'total_hazard_num':total_hazard_num, 
                    'average_pre_alert_time_step':average_pre_alert_time_step, 'average_pre_alert_time_minute':average_pre_alert_time_minute}
      all_pre_alert_time_df = all_pre_alert_time_df.append(result_dict, ignore_index=True)
all_pre_alert_time_df.to_csv(all_pre_alert_time_path)








