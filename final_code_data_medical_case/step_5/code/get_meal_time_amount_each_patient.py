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

def get_one_patient_meal_time(patient_trace_file_path):
  df = pd.read_csv(patient_trace_file_path)
  meal_time_list = df[(df['CHO']!=0) & (df['CHO'].notnull())]['Time'].to_list()
  meal_amount_list = df[(df['CHO']!=0) & (df['CHO'].notnull())]['CHO'].to_list()
  return meal_time_list,meal_amount_list

trace_folder = '/content/drive/MyDrive/ColabNotebooks/Medical_case/medical_case_pipeline/results/all_patient/10_patient_7_day/no_lstm'
patient_type_list = ['child', 'adult', 'adolescent']
patient_id_num_list = ["%03d" % x for x in range(1,11)]
all_patient_meal_time_df = pd.DataFrame()
all_patient_meal_time_path = '/content/drive/MyDrive/ColabNotebooks/Medical_case/medical_case_pipeline/all_meal_time.csv'
all_patient_meal_amount_df = pd.DataFrame()
all_patient_meal_amount_path = '/content/drive/MyDrive/ColabNotebooks/Medical_case/medical_case_pipeline/all_meal_amount.csv'
all_meal_time_list = []
all_meal_amount_list = []
for patient_type in patient_type_list:
  for id in patient_id_num_list:
    patient_file_path = '{patient_folder}/{patient_type}#{id}.csv'.format(patient_folder=trace_folder, patient_type=patient_type, id=id)
    meal_time_list,meal_amount_list = get_one_patient_meal_time(patient_file_path)
    all_meal_time_list.append(pd.DataFrame({'{patient_type}#{id}'.format(patient_type=patient_type, id=id): meal_time_list}))
    all_meal_amount_list.append(pd.DataFrame({'{patient_type}#{id}'.format(patient_type=patient_type, id=id): meal_amount_list}))
all_patient_meal_time_df = pd.concat(all_meal_time_list, axis=1)
all_patient_meal_amount_df = pd.concat(all_meal_amount_list, axis=1)
all_patient_meal_time_df.to_csv(all_patient_meal_time_path)
all_patient_meal_amount_df.to_csv(all_patient_meal_amount_path)