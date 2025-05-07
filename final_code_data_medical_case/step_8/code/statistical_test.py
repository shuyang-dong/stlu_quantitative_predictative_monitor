# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from numpy.random import randn
from numpy.random import seed
from numpy import mean
from numpy import var
from math import sqrt
from scipy.stats import shapiro,ttest_rel

# statistical test for pre-alert time
def get_aveg_prealert_time_each_patient(patient_type, control_type, df):
  hazard_pre_alert_time_list = df[df['hazard_id']!=0]['pre_alert_time'].to_list()
  return hazard_pre_alert_time_list
control_type_list = ['lstm_no_monitor', 'lstm_with_monitor']
patient_type_list = ['child', 'adult', 'adolescent']
folder = '/content/drive/MyDrive/ColabNotebooks/Medical_case/with_dropout/select_dropout_with_lossfunc_new/figure_data/data/segment_file_test_set_LQT'
result_folder = '/content/drive/MyDrive/ColabNotebooks/Medical_case/statistical_test'
aveg_pre_alert_time_path = '{result_folder}/all_mean_pre_alert_time.csv'.format(result_folder=result_folder)
aveg_pre_alert_time_df = pd.DataFrame()
aveg_pre_alert_time_list = []
for control_type in control_type_list:
  for patient_type in patient_type_list:
    patient_type_trace_file = '{folder}/real_cgm_trace_with_hazard_lable_{patient_type}_{control_type}.csv'.format(folder=folder,patient_type=patient_type,control_type=control_type)
    df = pd.read_csv(patient_type_trace_file, index_col=0)
    all_patient_aveg_pre_alert_time_list = get_aveg_prealert_time_each_patient(patient_type, control_type, df)
    col_name = '{patient_type}_{control_type}'.format(patient_type=patient_type, control_type=control_type)
    aveg_pre_alert_time_list.append(pd.DataFrame({col_name: all_patient_aveg_pre_alert_time_list}))
    
aveg_pre_alert_time_df = pd.concat(aveg_pre_alert_time_list, axis=1)
aveg_pre_alert_time_df.to_csv(aveg_pre_alert_time_path)

def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = mean(d1), mean(d2)
	# calculate the effect size
	return (u1 - u2) / s

def get_t_test_results_pre_alert_time(aveg_pre_alert_time_df, patient_type):
  pre_alert_time_with_monitor_df = aveg_pre_alert_time_df['{patient_type}_lstm_with_monitor'.format(patient_type=patient_type)].dropna(how='all')
  pre_alert_time_no_monitor_df = aveg_pre_alert_time_df['{patient_type}_lstm_no_monitor'.format(patient_type=patient_type)].dropna(how='all')
  d = cohend(pre_alert_time_with_monitor_df, pre_alert_time_no_monitor_df)
  t_test_value = ttest_rel(pre_alert_time_with_monitor_df, pre_alert_time_no_monitor_df)
  DF = len(pre_alert_time_with_monitor_df)-1

  return t_test_value.statistic, t_test_value.pvalue, d, DF

p_type_list = []
DF_list = []
t_list = []
p_list = []
d_list = []
t_test_result_df_prealert_time_df = pd.DataFrame(columns=['patient_type', 'DF', 't_value', 'p_value', 'd'])
t_test_result_df_prealert_time_path = '{result_folder}/t_test_pre_alert_time.csv'.format(result_folder=result_folder)
for patient_type in patient_type_list: 
  t, p, d, DF= get_t_test_results_pre_alert_time(aveg_pre_alert_time_df, patient_type=patient_type)
  t_list.append(t)
  p_list.append(p)
  d_list.append(d)
  p_type_list.append(patient_type)
  DF_list.append(DF)

t_test_result_df_prealert_time_df['patient_type'] = p_type_list
t_test_result_df_prealert_time_df['DF'] = DF_list 
t_test_result_df_prealert_time_df['t_value'] = t_list
t_test_result_df_prealert_time_df['p_value'] = p_list
t_test_result_df_prealert_time_df['d'] = d_list
t_test_result_df_prealert_time_df.to_csv(t_test_result_df_prealert_time_path)

# statistical test for TIR and total hazard number
def get_t_test_results_TIR_hazard_num(patient_controller_metric_file_path, patient_medical_metric_file_path, patient_type):
  controller_df = pd.read_csv(patient_controller_metric_file_path, index_col=0)
  medical_df = pd.read_csv(patient_medical_metric_file_path, index_col=0)
  controller_total_hazard_no_lstm_df = controller_df[controller_df['control_type']=='no_lstm']['total_hazard_num']
  controller_total_hazard_lstm_with_monitor_df = controller_df[(controller_df['control_type']=='lstm_with_monitor') & (controller_df['control_parameter']=='15_0')]['total_hazard_num']
  medical_TIR_no_lstm_df = medical_df[medical_df['control_type']=='no_lstm']['TIR']/100
  medical_TIR_lstm_with_monitor_df = medical_df[(medical_df['control_type']=='lstm_with_monitor') & (medical_df['control_parameter']=='15_0')]['TIR']/100
  d_hazard = cohend(controller_total_hazard_lstm_with_monitor_df, controller_total_hazard_no_lstm_df)
  t_hazard = ttest_rel(controller_total_hazard_lstm_with_monitor_df, controller_total_hazard_no_lstm_df)
  d_TIR = cohend(medical_TIR_lstm_with_monitor_df, medical_TIR_no_lstm_df)
  t_TIR = ttest_rel(medical_TIR_lstm_with_monitor_df, medical_TIR_no_lstm_df)
  DF = 9
  return d_hazard, t_hazard.statistic, t_hazard.pvalue, d_TIR, t_TIR.statistic, t_TIR.pvalue, DF

folder = '/content/drive/MyDrive/ColabNotebooks/Medical_case/medical_case_pipeline/results/all_patient/10_patient_7_day'
result_folder = '/content/drive/MyDrive/ColabNotebooks/Medical_case/statistical_test'
t_test_result_df = pd.DataFrame(columns=['patient_type', 'DF', 't_value_hazard_num', 'p_value_hazard_num', 'd_hazard_num', 
                            't_value_TIR', 'p_value_TIR', 'd_TIR'])
t_test_result_df_path = '{result_folder}/t_test_hazard_TIR_15_0.csv'.format(result_folder=result_folder)
patient_type_list = ['child', 'adult', 'adolescent']
for patient_type in patient_type_list:
  p_type_list = []
  DF_list = []
  t_hazard_list = []
  p_hazard_list = []
  d_hazard_list = []
  t_TIR_list = []
  p_TIR_list = []
  d_TIR_list = []
  patient_controller_metric_file_path = '{folder}/controller_metrics_result_{patient_type}.csv'.format(folder=folder,patient_type=patient_type)
  patient_medical_metric_file_path = '{folder}/medical_metrics_results_{patient_type}.csv'.format(folder=folder,patient_type=patient_type)
  d_h, t_h, p_h, d_t, t_t, p_t, DF = get_t_test_results_TIR_hazard_num(patient_controller_metric_file_path, patient_medical_metric_file_path, patient_type)
  result_dict = {'patient_type':patient_type, 'DF':DF, 't_value_hazard_num':t_h, 'p_value_hazard_num':p_h, 'd_hazard_num':d_h, 
                            't_value_TIR':t_t, 'p_value_TIR':p_t, 'd_TIR':d_t}
  t_test_result_df = t_test_result_df.append(result_dict, ignore_index=True)
t_test_result_df.to_csv(t_test_result_df_path)

