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
from torch.utils.tensorboard import SummaryWriter

import json
import random
import sys
import argparse


#Calculate set of population stats on the data
#takes in dataframe of BG values
# Example DF:
# pt | time0 | time1 | time2 | ... | time_n
# 0  | 50    | 52    | 55    | ... | 100
# 1  | 130   | 133   | 150   | ... | 330
def calculatePopulationStats(df):
    # Variance of BG for each patient
    varn = df.var(axis=1).mean()
    #print('varn: ', varn)

    # Time Hypo
    total_time = df.shape[1]
    hypoCount = df[df < 70.0].count(axis=1) / total_time
    hypoPercent = hypoCount.mean() * 100

    #Time Hyper
    hyperCount = df[df > 180.0].count(axis=1) / total_time
    hyperPercet = hyperCount.mean() * 100

    #Time in Range
    TIRReal = (total_time - (df[df < 70.0].count(axis=1) + df[df > 180.0].count(axis=1))) / total_time
    TIR = TIRReal.mean() * 100

    #Glycemic Variability Index
    gviReal = getGVI(df)

    #Patient Glycemic Status
    aveBG = df.mean(axis=1)
    TORReal = 1 - TIRReal #time out of range --> BG > 180 or BG < 70
    pgsReal = gviReal * aveBG * TORReal
    avePGS = pgsReal.mean()
    results_dict = {'varn':varn, 'hypoPercent':hypoPercent, 'hyperPercent':hyperPercet, 'TIR':TIR, 'gviReal':gviReal.values[0], 'pgsReal':pgsReal.values[0]}
    return results_dict

def getGVI(df):
    def lineDiff(timeStart, timeEnd, cgmStart, cgmEnd):
        return np.sqrt((timeEnd - timeStart) ** 2 + (abs(cgmEnd - cgmStart)) ** 2)

    diffs = df.diff(axis=1).abs().sum(axis=1)
    expectedDiffs = df.apply(lambda row: lineDiff(0, df.shape[1], row[0], row[df.shape[1]-1]), axis=1)
    return diffs/expectedDiffs

#Plot single cgm trace
def plotEGVTrace(trace, title=None):
    time = range(len(trace))
    plt.figure(figsize=(12, 7))
    plt.plot(time, trace)
    plt.xlabel("Time")
    plt.ylabel("Glucose mg/dL")

    if title != None:
        plt.title(title)

def mark_warning_id(patient_trace_path):
  '''
  give each warning an id at the start of a preiod of warnings
  '''
  patient_trace_df = pd.read_csv(patient_trace_path, index_col=0)
  warning_id_list = []
  warning_df_list = patient_trace_df['if_warning_this_step'].tolist()
  warning_id_count = 0
  last_warning_type = 0
  last_warning_end_time = 0
  for i in range(len(warning_df_list)):
    if i==0:
      if warning_df_list[i]!=0:
        warning_id_count += 1
        last_warning_end_time = i
      else:
        warning_id_count = 0
      warning_id_list.append(warning_id_count)
    else:
      if warning_df_list[i]==0:
        warning_id_list.append(0)
      elif warning_df_list[i]!=0 and warning_df_list[i]!=warning_df_list[i-1]:
        if (warning_df_list[i]==2 or warning_df_list[i]==4) and (warning_df_list[i-1]==2 or warning_df_list[i-1]==4):
          warning_id_count += 1
          warning_id_list.append(warning_id_count)
        elif (warning_df_list[i]==3 or warning_df_list[i]==5) and (warning_df_list[i-1]==3 or warning_df_list[i-1]==5):
          warning_id_count += 1
          warning_id_list.append(warning_id_count)
        else:
          if i-last_warning_end_time<=10:
            warning_id_list.append(0)
          else:
            warning_id_count += 1
            warning_id_list.append(warning_id_count)
        last_warning_end_time = i
      elif warning_df_list[i]!=0 and warning_df_list[i]==warning_df_list[i-1]:
        warning_id_list.append(0)
        last_warning_end_time = i
  
  patient_trace_df['warning_id'] = warning_id_list
  patient_trace_df.to_csv(patient_trace_path)

  return patient_trace_df

def mark_hazard_id_and_time(patient_trace_path):
  hazard_id_list = []
  hazard_type_list = []
  hazard_time_list = []
  patient_trace_df = pd.read_csv(patient_trace_path, index_col=0)
  bg_list = patient_trace_df['CGM'].tolist()
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
    if (j in hazard_id_index_list) and (j!=len(bg_list)-1):
      next_index = hazard_id_index_list[hazard_id_index_list.index(j)+1] if hazard_id_index_list.index(j)!=(len(hazard_id_index_list)-1) else len(bg_list)-1
      hazard_time_step_num = len(patient_trace_df[j:next_index][patient_trace_df['hazard_type']!=0])
      this_hazard_df = patient_trace_df[j:next_index][patient_trace_df['hazard_type']!=0]['CGM']
      max_CGM_this_hazard = max(this_hazard_df.to_list())
      min_CGM_this_hazard = min(this_hazard_df.to_list())
      hyper_time_this_hazard = len(patient_trace_df[j:next_index][(patient_trace_df['hazard_type']!=0) & (patient_trace_df['CGM']>180)])
      hypo_time_this_hazard = len(patient_trace_df[j:next_index][(patient_trace_df['hazard_type']!=0) & (patient_trace_df['CGM']<70)])
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
  patient_trace_df.to_csv(patient_trace_path)
  return patient_trace_df

def calculate_aveg_risk_indicators(patient_trace_path):
  patient_trace_df = pd.read_csv(patient_trace_path, index_col=0)
  aveg_LBGI = patient_trace_df['LBGI'].describe()['mean']
  aveg_HBGI = patient_trace_df['HBGI'].describe()['mean']
  aveg_Risk = patient_trace_df['Risk'].describe()['mean']
  max_LBGI = patient_trace_df['LBGI'].describe()['max']
  max_HBGI = patient_trace_df['HBGI'].describe()['max']
  max_Risk = patient_trace_df['Risk'].describe()['max']

  result_dict = {'aveg_LBGI': aveg_LBGI, 'aveg_HBGI':aveg_HBGI, 'aveg_Risk':aveg_Risk,
           'max_LBGI':max_LBGI, 'max_HBGI':max_HBGI, 'max_Risk':max_Risk}

  return result_dict

def mark_basal_bolus_adjust_id(patient_trace_path, adjust_type='bolus_correction_adjust'):
  '''
  give each correction bolus an id at the start of a preiod of warnings
  '''
  patient_trace_df = pd.read_csv(patient_trace_path, index_col=0)
  correction_bolus_id_list = []
  correction_bolus_list = patient_trace_df[adjust_type].tolist()
  correction_bolus_id_count = 0
  last_correction_bolus_type = 0
  last_correction_bolus_end_time = 0
  for i in range(len(correction_bolus_list)):
    if i==0:
      if correction_bolus_list[i]!=0:
        correction_bolus_id_count += 1
        last_correction_bolus_end_time = i
      else:
        correction_bolus_id_count = 0
      correction_bolus_id_list.append(correction_bolus_id_count)
    else:
      if correction_bolus_list[i]==0:
        correction_bolus_id_list.append(0)
      elif correction_bolus_list[i]!=0 and correction_bolus_list[i]!=correction_bolus_list[i-1]:
        if i-last_correction_bolus_end_time<=10:
            correction_bolus_id_list.append(0)
        else:
            correction_bolus_id_count += 1
            correction_bolus_id_list.append(correction_bolus_id_count)
        last_correction_bolus_end_time = i
        #'''
      elif correction_bolus_list[i]!=0 and correction_bolus_list[i]==correction_bolus_list[i-1]:
        correction_bolus_id_list.append(0)
        last_correction_bolus_end_time = i
  
  patient_trace_df['{adjust_type}_id'.format(adjust_type=adjust_type)] = correction_bolus_id_list
  patient_trace_df.to_csv(patient_trace_path)

  return patient_trace_df

def mark_warning_eat_id(patient_trace_path):
  '''
  give each eat warning an id at the start of a preiod of warnings
  '''
  patient_trace_df = pd.read_csv(patient_trace_path, index_col=0)
  if_warning_eat_id_list = []
  if_warning_eat_list = patient_trace_df['if_warning_eat_this_step'].tolist()
  if_warning_eat_id_count = 0
  last_if_warning_eat_type = 0
  last_if_warning_eat_end_time = 0
  for i in range(len(if_warning_eat_list)):
    if i==0:
      if if_warning_eat_list[i]!=0:
        if_warning_eat_id_count += 1
        last_if_warning_eat_end_time = i
      else:
        if_warning_eat_id_count = 0
      if_warning_eat_id_list.append(if_warning_eat_id_count)
    else:
      if if_warning_eat_list[i]==0:
        if_warning_eat_id_list.append(0)
      elif if_warning_eat_list[i]!=0 and if_warning_eat_list[i]!=if_warning_eat_list[i-1]:
        if i-last_if_warning_eat_end_time<=10:
            if_warning_eat_id_list.append(0)
        else:
            if_warning_eat_id_count += 1
            if_warning_eat_id_list.append(if_warning_eat_id_count)
        last_if_warning_eat_end_time = i
        #'''
      elif if_warning_eat_list[i]!=0 and if_warning_eat_list[i]==if_warning_eat_list[i-1]:
        if_warning_eat_id_list.append(0)
        last_if_warning_eat_end_time = i
  
  patient_trace_df['if_warning_eat_id'] = if_warning_eat_id_list
  patient_trace_df.to_csv(patient_trace_path)

  return patient_trace_df

def count_basal_bolus_adjust_num(patient_trace_path, adjust_type_id='bolus_correction_adjust_id'):
  patient_trace_df = pd.read_csv(patient_trace_path, index_col=0)
  id_df = patient_trace_df[patient_trace_df[adjust_type_id]!=0][adjust_type_id]
  adjust_num = len(id_df)
  return adjust_num

def count_warning_eat_num(patient_trace_path):
  patient_trace_df = pd.read_csv(patient_trace_path, index_col=0)
  id_df = patient_trace_df[patient_trace_df['if_warning_eat_id']!=0]['if_warning_eat_id']
  warning_eat_num = len(id_df)
  return warning_eat_num

def count_hazard_type_num(patient_trace_path, hazard_type=4):
  patient_trace_df = pd.read_csv(patient_trace_path, index_col=0)
  hazard_index_list = patient_trace_df[(patient_trace_df.hazard_type==hazard_type)].index.to_list()
  hazard_count = 0
  for i in hazard_index_list:
    this_hazard_id_index = i
    last_hazard_id_index = hazard_index_list[hazard_index_list.index(this_hazard_id_index)-1] if hazard_index_list.index(this_hazard_id_index)!=0 else 0
    if hazard_index_list.index(i)==0:
      hazard_count += 1
    else:
      if this_hazard_id_index-last_hazard_id_index<=10: # within 30 min
        continue
      else:
        hazard_count += 1

  return hazard_count

def metrics_summary(patient_trace_path, patient_type, patient_id, control_type, adjust_type_list):
  patient_trace_df = pd.read_csv(patient_trace_path, index_col=0)
  length = len(patient_trace_df)
  total_warning_num = len(patient_trace_df[patient_trace_df['warning_id']!=0])
  warning_statistic = patient_trace_df[patient_trace_df['warning_id']!=0]['if_warning_this_step'].value_counts()#.sum()
  hazard_type_statistic = patient_trace_df[patient_trace_df['hazard_id']!=0]['final_hazard_type'].value_counts()
  hazard_time_statistic = patient_trace_df[patient_trace_df['hazard_time']!=0]['hazard_time'].describe()

  result_dict = {'patient_type':patient_type, 'id':patient_id, 'control_type':control_type}
  for i in range(2, 6):
    # result_dict['warning_type_{i}'.format(i=i)] = warning_statistic[i] if (i in warning_statistic.index) else 0
    result_dict['hazard_type_{i}'.format(i=i)] = hazard_type_statistic[i] if (i in hazard_type_statistic.index) else 0
  result_dict['total_warning_num'] = warning_statistic.sum()
  result_dict['total_hazard_num'] = hazard_type_statistic.sum()
  # result_dict['aveg_hazard_time'] = hazard_time_statistic['mean'] if hazard_time_statistic['count']!=0 else 0

  # count number of each basal/bolus adjust
  for adjust_type in adjust_type_list:
    adjust_type_number = count_basal_bolus_adjust_num(patient_trace_path, adjust_type_id='{adjust_type}_id'.format(adjust_type=adjust_type))
    result_dict['{adjust_type}_num'.format(adjust_type=adjust_type)] = adjust_type_number
  # count eat warning num
  warning_eat_num = count_warning_eat_num(patient_trace_path)
  result_dict['warning_eat_num'] = warning_eat_num
  return result_dict


# get controller metric value for each patient
adjust_type_list = ['basal_severe_low_adjust','basal_severe_high_not_last_adjust','basal_severe_high_last_adjust', 
                    'basal_mild_low_not_last_adjust','basal_mild_low_last_adjust',
                    'basal_mild_high_not_last_adjust','basal_mild_high_last_adjust','bolus_correction_adjust',
                    'meal_bolus_adjust']
lstm_type_list = ['no_lstm', 'lstm_with_monitor']
patient_type_list = ['child', 'adult', 'adolescent']
patient_id_num_list = ["%03d" % x for x in range(1,11)]
start_step_list = [15]
end_step_list = [0]
folder = '/content/drive/MyDrive/ColabNotebooks/Medical_case/medical_case_pipeline/results/all_patient/10_patient_7_day'
metric_col_num = ['patient_type', 'id', 'control_type','control_parameter',
            'hazard_type_2', 'hazard_type_3', 'hazard_type_4','hazard_type_5','total_hazard_num',
                    'meal_bolus_adjust_num','warning_eat_num']

for patient_type in  patient_type_list:
  result_df = pd.DataFrame(columns=metric_col_num)
  metric_result_path = '{patient_folder}/controller_metrics_result_{patient_type}.csv'.format(patient_folder=folder,patient_type=patient_type)
  aveg_controller_metric_result_path = '{patient_folder}/aveg_controller_metrics_result_{patient_type}.csv'.format(patient_folder=folder,patient_type=patient_type)
  for id in patient_id_num_list:
    for lstm_type in lstm_type_list:
      if lstm_type=='lstm_with_monitor':
        for start_step in start_step_list:
          for end_step in end_step_list:
                control_param = '{start_step}_{end_step}'.format(start_step=start_step,end_step=end_step)
                print('lstm_type: ', lstm_type, ' patient id: ', id, ' control_param: ', control_param)
                patient_file_path = '{patient_folder}/{patient_type}_patient/10_patient_7_day_{control_param}/{lstm_type}/patient_trace_{patient_type}#{id}_{lstm_type}.csv'.format(lstm_type=lstm_type, 
                                              patient_folder=folder,control_param=control_param,patient_type=patient_type, id=id)
                
                mark_warning_id(patient_file_path)
                mark_hazard_id_and_time(patient_file_path)
                for adjust_type in adjust_type_list:
                  mark_basal_bolus_adjust_id(patient_file_path, adjust_type)
                mark_warning_eat_id(patient_file_path)
                results_dict = metrics_summary(patient_file_path, patient_type, id, lstm_type,adjust_type_list)
                results_dict['control_parameter'] = control_param
                result_df=result_df.append(results_dict, ignore_index=True)
      else:
        patient_file_path = '{patient_folder}/{lstm_type}/patient_trace_{patient_type}#{id}_{lstm_type}.csv'.format(patient_folder=folder, 
                                    lstm_type=lstm_type, patient_type=patient_type, id=id)
        mark_warning_id(patient_file_path)
        mark_hazard_id_and_time(patient_file_path)
        for adjust_type in adjust_type_list:
          mark_basal_bolus_adjust_id(patient_file_path, adjust_type)
        mark_warning_eat_id(patient_file_path)
        results_dict = metrics_summary(patient_file_path, patient_type, id, lstm_type,adjust_type_list)
        results_dict['control_parameter'] = 0
        result_df=result_df.append(results_dict, ignore_index=True)

  result_df.to_csv(metric_result_path)
  aveg_controller_metrics_results = pd.DataFrame(columns=['patient_type', 'control_type', 'control_parameter'])
  df_list = []
  for lstm_type in lstm_type_list:
    if lstm_type=='lstm_with_monitor':
      for start_step in start_step_list:
          for end_step in end_step_list:
                control_param = '{start_step}_{end_step}'.format(start_step=start_step,end_step=end_step)
                df = result_df[(result_df['control_type']==lstm_type)  & (result_df['control_parameter']==control_param) ]
                mean_values = df.mean()
                mean_values['id'] = lstm_type
                mean_values = mean_values.rename(index={'id':'control_type'})
                mean_values['control_parameter'] = control_param
                df_list.append(mean_values)
    else:
      df = result_df[result_df['control_type']==lstm_type]
      mean_values = df.mean()
      mean_values['id'] = lstm_type
      mean_values = mean_values.rename(index={'id':'control_type'})
      mean_values['control_parameter'] = 0
      df_list.append(mean_values)
  aveg_controller_metrics_results = pd.concat(df_list, axis=1)
  print(aveg_controller_metrics_results)
  aveg_controller_metrics_results.to_csv(aveg_controller_metric_result_path)

# get medical metric values for each patient
result_col = ['id', 'control_type', 'control_parameter','hypoPercent', 'hyperPercent', 'TIR']
results_path = '{folder}/medical_metrics_results.csv'.format(folder=folder)
for patient_type in patient_type_list:
  print('patient_type: ', patient_type)
  result_df = pd.DataFrame(columns=result_col)
  results_path = '{folder}/medical_metrics_results_{patient_type}.csv'.format(folder=folder,patient_type=patient_type)
  for lstm_type in lstm_type_list:
    print('lstm_type: ', lstm_type)
    if lstm_type=='lstm_with_monitor':
      for start_step in start_step_list:
          for end_step in end_step_list:
              control_param = '{start_step}_{end_step}'.format(start_step=start_step,end_step=end_step)
              print('control_param: ', control_param)
              file_folder = '{folder}/{patient_type}_patient/10_patient_7_day_{control_param}/{lstm_type}'.format(lstm_type=lstm_type, folder=folder,control_param=control_param,
                                                                patient_type=patient_type)
              for id in patient_id_num_list:
                  BG_df = pd.DataFrame()
                  file_name = '{file_folder}/patient_trace_{patient_type}#{id}_{lstm_type}.csv'.format(file_folder=file_folder, patient_type=patient_type, 
                                                      id=id, lstm_type=lstm_type)
                  df = pd.read_csv(file_name)
                  part_df = df[['CGM']]
                  part_df = part_df.rename(columns={"CGM": id})
                  BG_df[id] = part_df[id]
                  bg_df = BG_df.T
                  results_dict_1 = calculatePopulationStats(bg_df)
                  results_dict_1['id'] = id
                  results_dict_1['control_type'] = lstm_type
                  results_dict_1['control_parameter'] = control_param
                  results_dict_2 = calculate_aveg_risk_indicators(file_name)
                  final_result = results_dict_1.copy()
                  final_result.update(results_dict_2)
                  result_df = result_df.append(final_result, ignore_index=True)
    else:
      file_folder = '{folder}/{lstm_type}'.format(lstm_type=lstm_type, folder=folder)
      for id in patient_id_num_list:
          BG_df = pd.DataFrame()
          file_name = '{file_folder}/patient_trace_{patient_type}#{id}_{lstm_type}.csv'.format(file_folder=file_folder, patient_type=patient_type, 
                                              id=id, lstm_type=lstm_type)
          df = pd.read_csv(file_name)
          part_df = df[['CGM']]
          part_df = part_df.rename(columns={"CGM": id})
          BG_df[id] = part_df[id]
          bg_df = BG_df.T
          results_dict_1 = calculatePopulationStats(bg_df)
          results_dict_1['id'] = id
          results_dict_1['control_type'] = lstm_type
          results_dict_1['control_parameter'] = 0
          results_dict_2 = calculate_aveg_risk_indicators(file_name)
          final_result = results_dict_1.copy()
          final_result.update(results_dict_2)
          result_df = result_df.append(final_result, ignore_index=True)
  result_df.to_csv(results_path)

  aveg_medical_result_df_path = '{folder}/aveg_medical_metrics_results_{patient_type}.csv'.format(folder=folder,patient_type=patient_type)
  df = pd.read_csv(results_path, index_col=0)
  control_type_list = lstm_type_list
  describe_df_list = []
  for control_type in control_type_list:
    if control_type=='lstm_with_monitor':
      for start_step in start_step_list:
          for end_step in end_step_list:
                control_param = '{start_step}_{end_step}'.format(start_step=start_step,end_step=end_step)
                df_1 = df[(df['control_type']==control_type) & (df['control_parameter']==control_param)]
                describe_df = df_1.mean()
                describe_df['id'] = control_type
                describe_df['control_parameter'] = control_param
                describe_df = describe_df.rename(index={'id':'control_type'})
                describe_df_list.append(describe_df)
    else:
      df_1 = df[df['control_type']==control_type]
      describe_df = df_1.mean()
      describe_df['id'] = control_type
      describe_df['control_parameter'] = 0
      describe_df = describe_df.rename(index={'id':'control_type'})
      describe_df_list.append(describe_df)
  aveg_medical_result_df = pd.concat(describe_df_list, axis=1)
  print(aveg_medical_result_df)
  aveg_medical_result_df.to_csv(aveg_medical_result_df_path)

# get all aveg metrics value
patient_type_list = ['child', 'adult', 'adolescent']
for patient_type in patient_type_list:
  aveg_medical_result_df_path = '{folder}/aveg_medical_metrics_results_{patient_type}.csv'.format(folder=folder,patient_type=patient_type)
  aveg_controller_metric_result_path = '{folder}/aveg_controller_metrics_result_{patient_type}.csv'.format(folder=folder,patient_type=patient_type)
  aveg_medical_results_df = pd.read_csv(aveg_medical_result_df_path, index_col=0)
  aveg_controller_results_df = pd.read_csv(aveg_controller_metric_result_path, index_col=0)
  all_aveg_results_df = pd.concat([aveg_controller_results_df,aveg_medical_results_df])
  all_aveg_results_df = all_aveg_results_df[~all_aveg_results_df.index.duplicated(keep='first')]
  print(all_aveg_results_df)
  all_aveg_results_df_path = '{folder}/all_aveg_metrics_results_{patient_type}.csv'.format(folder=folder,patient_type=patient_type)
  all_aveg_results_df.to_csv(all_aveg_results_df_path)
