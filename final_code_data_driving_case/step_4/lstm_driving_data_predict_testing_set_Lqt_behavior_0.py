# -*- coding: utf-8 -*-
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.6f' % x)
pd.set_option('display.max_columns', None)
#from sklearn.preprocessing import MinMaxScaler
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
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import torch._VF as _VF
#from torch.utils.tensorboard import SummaryWriter

import json
import random
import scipy
import sys
sys.path.append('/home/cpsgroup/predictive_monitor_new_stl/stlu_monitor')
import ustlmonitor as ustl
import confidencelevel
import argparse

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('New folder ok.')
    else:
        print('There is this folder')

    return

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
    # #
    # acc_list = []
    # delta_acc_list = []
    # for item, row in df.iterrows():
    #     acc_x = row['acc_x']
    #     acc_y = row['acc_y']
    #     acc_z = row['acc_z']
    #     acc = math.sqrt(math.pow(acc_x, 2) + math.pow(acc_y, 2) + math.pow(acc_z, 2))
    #     acc_list.append(acc)
    #     if item == 0:
    #         delta_acc_list.append(0)
    #     else:
    #         delta_acc_list.append(acc-acc_list[item-1])
    #
    # df['acc'] = acc_list
    # df['delta_acc'] = delta_acc_list
    # #
    #print('df after decide acc: ', df)
    df.to_csv(driving_trace_file)
    return

# Path to the folder where the datasets are/should be downloaded
behavior_type = 0
scenario_id = 3
route_id_list = [7]
episode_num = 50
trace_seed_list = np.arange(15).tolist()
data_folder = '/home/cpsgroup/SafeBench/safebench/predictive_monitor_trace/trace_file'
#data_folder = '/home/cpsgroup/SafeBench/safebench/predictive_monitor_trace/trace_file_orig_new'
DATASET_PATH = data_folder
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "/home/cpsgroup/predictive_monitor_new_stl/lstm_checkpoint/behavior_type_{behavior_type}".format(behavior_type=behavior_type)
mkdir(CHECKPOINT_PATH)
# Function for setting the seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(32)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

# selected_feature_col = ['v_x', 'v_y', 'lateral_dis', '-delta_yaw', 'vehicle_front', 'acc_x', 'acc_y', 'acc_z',
# 'ego_yaw', 'radar_obj_depth_aveg', 'radar_obj_depth_std', 'radar_obj_velocity_aveg', 'radar_obj_velocity_std',
#                         'throttle', 'steer', 'brake', 'speed', 'acc', 'delta_acc']
selected_feature_col = ['v_x', 'v_y', 'lateral_dis', '-delta_yaw', 'vehicle_front', 'acc_x', 'acc_y', 'acc_z',
'ego_yaw', 'radar_obj_depth_aveg', 'radar_obj_velocity_aveg',
                        'throttle', 'steer', 'brake', 'speed', 'acc', 'delta_acc']
selected_feature_num = len(selected_feature_col)

def data_standardlize(dataset, whole_dataset):
  dataset = (dataset-whole_dataset.mean())/whole_dataset.std()
  return dataset

def generate_dataset_by_type(trace_folder, behavior_type, route_id_list, episode_num, scenario_id,
                             seq_length, selected_feature_col):
  # standardlize data and get samples by sliding window
  df_list = []
  new_df_list = []
  for trace_seed in trace_seed_list:
      # trace_seed_list = trace_seed_list_dict[id]
      # trace_seed_list = np.arange(10).tolist()
      for id in route_id_list:
          trace_file_path = '{trace_folder}/eval_agent_behavior_type_{behavior_type}_scenario_{scenario_id}_route_{id}_episode_{episode_num}_seed_{trace_seed}.csv'.format(
              behavior_type=behavior_type, scenario_id=scenario_id, id=id, episode_num=episode_num,
              trace_seed=trace_seed,
              trace_folder=trace_folder)
          df = pd.read_csv(trace_file_path)[selected_feature_col]
          df.dropna(axis=0, how='any', inplace=True)
          # print('df: ', df)
          # df.drop([len(df) - 1], inplace=True)
          df_list.append(df)

  whole_dataset = pd.concat(df_list)
  #print('whole dataset before standardlize: ', whole_dataset)

  mean_value_df = whole_dataset.mean()
  std_value_df = whole_dataset.std()
  #print('mean_value_df: ', mean_value_df)
  #print('std_value_df: ', std_value_df)
  for df in df_list:
      new_df = data_standardlize(df, whole_dataset)
      new_df_list.append(new_df)
  #print('dataset after standardlize: ', new_df_list[0])
  #random.shuffle(new_df_list)
  # slicing by sliding window
  x_0=[]
  y_0=[]
  for dataset_single_trace in new_df_list:
    for i in range(len(dataset_single_trace)-seq_length): # range(0,len(data)-seq_length-1,seq_length)
          _x = dataset_single_trace[i:(i+seq_length)]
          _y = dataset_single_trace[i+1:i+seq_length+1]
          x_0.append(_x)
          y_0.append(_y)

  x, y = np.array(x_0),np.array(y_0)
  # print('x before shuffle: ', x)
  # np.random.shuffle(x)
  # np.random.shuffle(y)
  # print('x after shuffle: ', x)
  # convert to tensor
  dataX = Variable(torch.from_numpy(x))
  dataY = Variable(torch.from_numpy(y))
  return dataX, dataY, mean_value_df, std_value_df

# def generate_dataset_by_type(patient_folder, patient_type, patient_id_num_list, seq_length, selected_feature_col):
#   # standardlize data and get samples by sliding window
#   df_list = []
#   new_df_list = []
#   for id in patient_id_num_list:
#     patient_file_path = '{patient_folder}/{patient_type}#{id}.csv'.format(patient_folder=patient_folder, patient_type=patient_type, id=id)
#     df = pd.read_csv(patient_file_path)[selected_feature_col]
#     df.drop([len(df)-1], inplace=True)
#     df_list.append(df)
#   whole_dataset = pd.concat(df_list)
#   mean_value_df = whole_dataset.mean()
#   std_value_df = whole_dataset.std()
#   for df in df_list:
#     new_df = data_standardlize(df, whole_dataset)
#     new_df_list.append(new_df)
#   # slicing by sliding window
#   x_0=[]
#   y_0=[]
#   for dataset_single_patient in new_df_list:
#     for i in range(len(dataset_single_patient)-seq_length): # range(0,len(data)-seq_length-1,seq_length)
#           _x = dataset_single_patient[i:(i+seq_length)]
#           _y = dataset_single_patient[i+1:i+seq_length+1]
#           x_0.append(_x)
#           y_0.append(_y)
#
#   x, y = np.array(x_0),np.array(y_0)
#   # convert to tensor
#   dataX = Variable(torch.from_numpy(x))
#   dataY = Variable(torch.from_numpy(y))
#   return dataX, dataY, mean_value_df, std_value_df

seq_length = 50 # past 30 steps(3s) for predicting future 30 steps(3s)
step_look_back= 30
predict_length=seq_length-step_look_back

batch_size = 1024
# patient_type_list = ['child', 'adult', 'adolescent']
# patient_id_num_list = ["%03d" % x for x in range(1, 11)]
# patient_folder = data_folder

# preprocess trace, get the value of acc and delta acc and add to trace file
for route_id in  route_id_list:
    # trace_seed_list = trace_seed_list_dict[route_id]
    for trace_seed in trace_seed_list:
        print(route_id, trace_seed)
        driving_file_path = '{data_folder}/eval_agent_behavior_type_{behavior_type}_scenario_{scenario_id}_route_{route_id}_episode_{episode_num}_seed_{trace_seed}.csv'.format(data_folder=data_folder,
                                    behavior_type=behavior_type, scenario_id=scenario_id, route_id=route_id,
                                    episode_num=episode_num, trace_seed=trace_seed)
        preprocess_driving_data(driving_file_path)
#"""
all_dataX_1, all_dataY_1, mean_value_1, std_value_1 = generate_dataset_by_type(
    data_folder, behavior_type, route_id_list, episode_num, scenario_id,
    seq_length, selected_feature_col)

all_dataX = all_dataX_1
all_dataY = all_dataY_1
train_sample_x = all_dataX[:int(len(all_dataX) * 0.80)]  # 80% for training
train_sample_y = all_dataY[:int(len(all_dataY) * 0.80)]
test_sample_x = all_dataX[int(len(all_dataX) * 0.80):int(len(all_dataX) * 0.90)]  # 10% for test
test_sample_y = all_dataY[int(len(all_dataY) * 0.80):int(len(all_dataY) * 0.90)]
valid_sample_x = all_dataX[int(len(all_dataX) * 0.90):]  # 10% for validation
valid_sample_y = all_dataY[int(len(all_dataY) * 0.90):]
mean_value, std_value = mean_value_1, std_value_1

"""
if behavior_type==0:
    all_dataX = all_dataX_1
    all_dataY = all_dataY_1
    train_sample_x = all_dataX[:int(len(all_dataX) * 0.70)]  # 70 days for training
    train_sample_y = all_dataY[:int(len(all_dataY) * 0.70)]
    test_sample_x = all_dataX[int(len(all_dataX) * 0.70):int(len(all_dataX) * 0.85)]  # 10 days for test
    test_sample_y = all_dataY[int(len(all_dataY) * 0.70):int(len(all_dataY) * 0.85)]
    valid_sample_x = all_dataX[int(len(all_dataX) * 0.85):]  # 5 days for validation
    valid_sample_y = all_dataY[int(len(all_dataY) * 0.85):]
    mean_value, std_value = mean_value_1, std_value_1
"""

# get training set, testing set and validation set for chosen patient type
trainX = Variable(torch.Tensor(np.array(train_sample_x)))
trainY = Variable(torch.Tensor(np.array(train_sample_y)))
testX = Variable(torch.Tensor(np.array(test_sample_x)))
testY = Variable(torch.Tensor(np.array(test_sample_y)))
validX = Variable(torch.Tensor(np.array(valid_sample_x)))
validY = Variable(torch.Tensor(np.array(valid_sample_y)))

train_set = torch.utils.data.TensorDataset(trainX,trainY)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)
valid_set = torch.utils.data.TensorDataset(validX,validY)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, num_workers=0, shuffle=False)
test_set = torch.utils.data.TensorDataset(testX,testY)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=0, shuffle=False)

print('Preprocess finished.')

# LSTM module
class LSTMCellWithMask(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCellWithMask, self).__init__(input_size, hidden_size, bias=True)
        
    def forward_with_mask(self, input, mask, hx=None):
        (mask_ih, mask_hh) = mask
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]])->Tuple[Tensor, Tensor]
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        # LSTMCell.weight_ih： [4*hidden_size, input_size]
        # LSTMCell.weight_hh [4*hidden_size, hidden_size]
        # LSTMCell.bias_ih [4*hidden_size]
        # LSTMCell.bias_hh [4*hidden_size]
        return _VF.lstm_cell(
            input, hx,
            self.weight_ih * mask_ih, self.weight_hh * mask_hh,
            self.bias_ih, self.bias_hh
        )

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, step_look_back, train_dropout_type):
        super(LSTM, self). __init__()
        self.lstm = LSTMCellWithMask(input_size=input_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.train_dropout_type = train_dropout_type
        print('train dropout type: ', train_dropout_type)
        # store all hyperparameters in a dictionary for saving and loading of the model
        self.config = {"input_size": input_size, "output_size": output_size, "hidden_size": hidden_size}
      
    def forward(self, input, step_look_back, hidden_size, dropout_rate):
        # funcation for train, dropout type using train_dropout_type
        outputs = []
        device = input.device
        h_t = torch.zeros(input.size(0), input_size, dtype=torch.float, device=device)
        c_t = torch.zeros(input.size(0), input_size, dtype=torch.float, device=device)

        if self.train_dropout_type == 1:
            mask1 = torch.bernoulli(torch.ones(4*hidden_size, input_size, dtype=torch.float)*dropout_rate)/dropout_rate  
            mask2 = torch.bernoulli(torch.ones(4*hidden_size, hidden_size, dtype=torch.float)*dropout_rate)/dropout_rate
        elif self.train_dropout_type == 2:
            para = torch.bernoulli(torch.ones(4*hidden_size, input_size, dtype=torch.float)*dropout_rate)/dropout_rate 
            mask1 = para
            mask2 = para.expand(-1, hidden_size)
        elif self.train_dropout_type == 3:
            p = math.sqrt((1-dropout_rate)/dropout_rate)
            mask1 = torch.normal(1, torch.ones(4*hidden_size, input_size, dtype=torch.float)*p)
            mask2 = torch.normal(1, torch.ones(4*hidden_size, hidden_size, dtype=torch.float)*p)
        elif self.train_dropout_type == 4:
            p = math.sqrt((1-dropout_rate)/dropout_rate)
            para = torch.normal(1, torch.ones(4*hidden_size, input_size, dtype=torch.float)*p)
            mask1 = para
            mask2 = para.expand(-1, hidden_size)
        else:
            print("Please select the correct DROPOUT_TYPE: 1-4")
        mask  = (mask1.to(device),mask2.to(device))

        sequence_length = input.size(1)
        predict_length = sequence_length-step_look_back
        for i in range(input.size(1)):
          if i<=step_look_back-1:
            h_t, c_t = self.lstm.forward_with_mask(input[:, i, :], mask, (h_t, c_t))
            output = self.linear(h_t)
            outputs.append(output)
          else:
            h_t, c_t = self.lstm.forward_with_mask(outputs[i-1], mask, (h_t, c_t))
            output = self.linear(h_t)
            outputs.append(output)

        outputs = torch.stack(outputs[step_look_back:], 1).squeeze(2)
        return outputs

    def forward_test_with_past(self, input, step_look_back, hidden_size, dropout_rate, dropout_type):
        outputs = []
        device = input.device
        h_t = torch.zeros(input.size(0), input_size, dtype=torch.float, device=device)
        c_t = torch.zeros(input.size(0), input_size, dtype=torch.float, device=device)

        if dropout_type == 1:
            mask1 = torch.bernoulli(torch.ones(4*hidden_size, input_size, dtype=torch.float)*dropout_rate)/dropout_rate  
            mask2 = torch.bernoulli(torch.ones(4*hidden_size, hidden_size, dtype=torch.float)*dropout_rate)/dropout_rate
        elif dropout_type == 2:
            para = torch.bernoulli(torch.ones(4*hidden_size, input_size, dtype=torch.float)*dropout_rate)/dropout_rate 
            mask1 = para
            mask2 = para.expand(-1, hidden_size)
        elif dropout_type == 3:
            p = math.sqrt((1-dropout_rate)/dropout_rate)
            mask1 = torch.normal(1, torch.ones(4*hidden_size, input_size, dtype=torch.float)*p)
            mask2 = torch.normal(1, torch.ones(4*hidden_size, hidden_size, dtype=torch.float)*p)
        elif dropout_type == 4:
            p = math.sqrt((1-dropout_rate)/dropout_rate)
            para = torch.normal(1, torch.ones(4*hidden_size, input_size, dtype=torch.float)*p)
            mask1 = para
            mask2 = para.expand(-1, hidden_size)
        else:
            print("Please select the correct DROPOUT_TYPE: 1-4")
        mask  = (mask1.to(device),mask2.to(device))
        
        sequence_length = input.size(1)
        predict_length = sequence_length-step_look_back
        for i in range(input.size(1)):
          if i<=step_look_back-1:
            h_t, c_t = self.lstm.forward_with_mask(input[:, i, :], mask, (h_t, c_t))
            output = self.linear(h_t)
            outputs.append(output)
          else:
            h_t, c_t = self.lstm.forward_with_mask(outputs[i-1], mask, (h_t, c_t))
            output = self.linear(h_t)
            outputs.append(output)

        outputs = torch.stack(outputs[step_look_back:], 1).squeeze(2)
        return outputs

def train(lstm_model, train_loader, criterion, optimizer, device, batch_size, hidden_size, dropout_rate, step_look_back):
  # train model
  lstm_model.train()
  train_loss = 0
  for batch_idx, datasample in enumerate(train_loader):
      data,target0 = datasample[0], datasample[1]
      target = []
      for target_data in datasample[1]:
          target.append(target_data[step_look_back:,:])

      target = torch.stack(target,0)
      data, target = data.to(device), target.to(device)
      output = lstm_model(data, step_look_back, hidden_size, dropout_rate)
      loss = criterion(output, target)
      train_loss += loss.item()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  train_loss /= len(train_loader.dataset)
  return train_loss

def test_model(lstm_model, test_loader, criterion, device, predict_length, N_MC,batch_size,hidden_size,dropout_rate,dropout_type,step_look_back):
  # test model
  batch_len = 0
  lstm_model.eval()
  test_loss = 0
  test_loss_mean = 0
  with torch.no_grad():
      for batch_idx, datasample in enumerate(test_loader):
          data,target0 = datasample[0], datasample[1]
          target = []
          for target_data in datasample[1]:
            target.append(target_data[step_look_back:,:])
          target = torch.stack(target,0)
          data, target = data.to(device), target.to(device)
          if target0.size(0)==batch_size:
            batch_len = batch_size
          else:
            batch_len = target0.size(0)
          output = torch.zeros(N_MC, batch_len, predict_length, hidden_size)
          for i in range(N_MC):
            output[i] = lstm_model.forward_test_with_past(data, step_look_back, hidden_size, dropout_rate, dropout_type).cpu()
            test_loss += criterion(output[i], target.cpu()).item()
          test_loss_mean += criterion(output.mean(dim = 0), target.cpu()).item()

  test_loss /= (len(test_loader.dataset)*N_MC)
  test_loss_mean /= (len(test_loader.dataset))
  return test_loss, test_loss_mean

def loss_func_for_one_flowpipe(rho_truth, rho_predict, ground_truth_trace, predicted_future_mean_trace, predicted_future_std_trace, conf, beta_value=0.3):
  '''
  loss func Lqt
  rho_truth: rho interval for ground truth trace
  rho_predict: rho interval for predicted flowpipe
  '''
  pedicted_len = len(predicted_future_mean_trace)
  # calculate upper and lower bound using given mean, std, and conf
  p = 1 - (1 - conf) / 2
  lower = predicted_future_mean_trace - norm.ppf(p) * predicted_future_std_trace # norm.ppf(p):get the x value for certain confidence level p
  upper = predicted_future_mean_trace + norm.ppf(p) * predicted_future_std_trace
  # calculate d_rho
  d_rho = rho_predict[0] if rho_truth[0]>=0 else -rho_predict[1]
  # calculate d_i
  d_i = 0
  for t in range(pedicted_len):
    if lower[t]<=ground_truth_trace[t]<=upper[t]:
      d_i += 0
    elif ground_truth_trace[t]<lower[t]:
      d_i += lower[t] - ground_truth_trace[t]
    elif ground_truth_trace[t]>upper[t]:
      d_i += ground_truth_trace[t] - upper[t]
  # calculate loss for single sample
  loss = beta_value*(-d_rho) + (1-beta_value)*d_i
  return loss.item()

# evaluation with STL-U and dropout
def eval_model(device, segment_result_col, dataset_loader, lstm_model, criterion,
               seq_length, step_look_back, epoch, learning_rate, N_MC, batch_size, hidden_size,
               dropout_rate, dropout_type, requirement_func, conf, mean_df, std_df):
    # evaluate the trained model with dataset
    lstm_model.eval()
    test_loss = 0
    test_loss_mean = 0
    loss_one_sample_average = 0
    loss_L_SAT = 0
    batch_len = 0
    predict_length = seq_length - step_look_back
    segment_result_df = pd.DataFrame(columns=segment_result_col)

    # CGM_mean_value = mean_df['CGM']
    # CGM_std_value = std_df['CGM']
    # CHO_mean_value = mean_df['CHO']
    # CHO_std_value = std_df['CHO']
    # insulin_mean_value = mean_df['insulin']
    # insulin_std_value = std_df['insulin']

    acc_mean_value = mean_df['acc']
    acc_std_value = std_df['acc']
    delta_acc_mean_value = mean_df['delta_acc']
    delta_acc_std_value = std_df['delta_acc']
    speed_mean_value = mean_df['speed']
    speed_std_value = std_df['speed']
    #
    # selected_feature_col = ['lateral_dis', '-delta_yaw', 'vehicle_front', 'acc_x', 'acc_y', 'acc_z',
    #                         'ego_yaw', 'relative_velocity_mean', 'relative_velocity_std', 'relative_distance_mean',
    #                         'relative_distance_std',
    #                         'throttle', 'steer', 'brake', 'speed', 'acc', 'delta_acc']

    with torch.no_grad():
        for batch_idx, datasample in enumerate(dataset_loader):
            data, target0 = datasample[0], datasample[1]
            target = []
            for target_data in datasample[1]:
                target.append(target_data[step_look_back:, :])
            target = torch.stack(target, 0)
            data, target = data.to(device), target.to(device)
            if target0.size(0) == batch_size:
                batch_len = batch_size
            else:
                batch_len = target0.size(0)
            output = torch.zeros(N_MC, batch_len, predict_length, hidden_size)
            # real_whole_CGM_all_batch = target0[:, :, -3:-2] * CGM_std_value + CGM_mean_value
            # real_whole_CHO_all_batch = target0[:, :, -2:-1] * CHO_std_value + CHO_mean_value
            # real_whole_insulin_all_batch = target0[:, :, -1:] * insulin_std_value + insulin_mean_value
            # real_whole_CGM_future_batch = real_whole_CGM_all_batch[:, step_look_back:, :]
            # speed, acc, delta_acc
            real_whole_speed_all_batch = target0[:, :, -3:-2] * speed_std_value + speed_mean_value
            real_whole_acc_all_batch = target0[:, :, -2:-1] * acc_std_value + acc_mean_value
            real_whole_delta_acc_all_batch = target0[:, :, -1:] * delta_acc_std_value + delta_acc_mean_value

            # apply dropout and loop N_MC times to get a series of predicted traces
            new_mean_df = mean_df.reset_index(drop=True)
            new_std_df = std_df.reset_index(drop=True)
            # change target data back to original scale
            for feature_index in range(target.shape[2]):
                target[:, :, feature_index] = (
                            target[:, :, feature_index] * new_std_df.loc[feature_index] + new_mean_df.loc[
                        feature_index]).cpu()
            for i in range(N_MC):
                output[i] = (lstm_model.forward_test_with_past(data, step_look_back, hidden_size, dropout_rate,
                                                               dropout_type)).cpu()
                # change predictions back to original scale
                for feature_index in range(output[i].shape[2]):
                    output[i][:, :, feature_index] = (
                                output[i][:, :, feature_index] * new_std_df.loc[feature_index] + new_mean_df.loc[feature_index]).cpu()
                test_loss += criterion(output[i], target.cpu()).item()
            test_loss_mean += criterion(output.mean(dim=0), target.cpu()).item()

            # output_CGM_mean = output.mean(dim=0)[:, :, -3]
            # output_CGM_std = output.std(dim=0)[:, :, -3]
            # output_CGM_target = target[:, :, -3]

            output_acc_mean = output.mean(dim=0)[:, :, -2]
            output_acc_std = output.std(dim=0)[:, :, -2]
            output_acc_target = target[:, :, -2]

            # trace_CGM = torch.stack((output_CGM_mean, output_CGM_std, output_CGM_target.cpu(),
            #                          torch.zeros(output_CGM_target.size())), dim=-1)
            trace_acc = torch.stack((output_acc_mean, output_acc_std, output_acc_target.cpu(),
                                     torch.zeros(output_acc_target.size())), dim=-1)
            # for LSAT
            loss_L_SAT += lsat(trace_acc[:, :, :-2], trace_acc[:, :, -2:], requirement_func, predict_length, conf)
            # record mean and std value for each sample after applying dropout and loop N_MC times
            for b in range(batch_len):

                current_trace_real_acc_all_segment = output_acc_target[b, :]
                for i in range(current_trace_real_acc_all_segment.size(0)):
                    current_trace_real_acc = current_trace_real_acc_all_segment[i].item()
                # get rho interval for ground truth and predictions
                # rho_orig_acc = requirement_func(trace_acc[b, :, -2:], predict_length, conf, lower_acc=-5.0,
                #                                 upper_acc=4.5, func='monitor')
                # rho_set_acc = requirement_func(trace_acc[b, :, :-2], predict_length, conf, lower_acc=-5.0,
                #                                upper_acc=4.5, func='monitor')
                rho_orig_acc = requirement_func(trace_acc[b, :, -2:], predict_length, conf)
                rho_set_acc = requirement_func(trace_acc[b, :, :-2], predict_length, conf)
                ground_truth_trace = trace_acc[b, :, -2:-1]
                predicted_future_mean_trace = trace_acc[b, :, :1]
                predicted_future_std_trace = trace_acc[b, :, 1:2]
                loss_one_sample = loss_func_for_one_flowpipe(rho_orig_acc, rho_set_acc, ground_truth_trace,
                                                             predicted_future_mean_trace,
                                                             predicted_future_std_trace, conf)
                loss_one_sample_average += loss_one_sample

                # record mean and std value for each sample in one batch
                # each_real_CHO_in_batch = real_whole_CHO_all_batch[b, :, :][:, 0]
                # each_real_insulin_in_batch = real_whole_insulin_all_batch[b, :, :][:, 0]
                # each_real_CGM_in_batch = real_whole_CGM_all_batch[b, :, :][:, 0]
                # CGM_predicted_mean = output.mean(dim=0)[b, :, -3]
                # CGM_predicted_std = output.std(dim=0)[b, :, -3]
                # segment_loss = criterion(CGM_predicted_mean,
                #                                each_real_CGM_in_batch[step_look_back:].cpu()).item()

                each_real_acc_in_batch = real_whole_acc_all_batch[b, :, :][:, 0]
                each_real_delta_acc_in_batch = real_whole_delta_acc_all_batch[b, :, :][:, 0]
                each_real_speed_in_batch = real_whole_speed_all_batch[b, :, :][:, 0]
                acc_predicted_mean = output.mean(dim=0)[b, :, -2]
                acc_predicted_std = output.std(dim=0)[b, :, -2]
                segment_loss = criterion(acc_predicted_mean,
                                         each_real_acc_in_batch[step_look_back:].cpu()).item()

                # segment_result_dict = {'learning_rate': learning_rate, 'epoch': epoch,
                #                        'batch_idx': batch_idx, 'batch_num': b, 'N_MC': N_MC, 'conf': conf,
                #                        'dropout_type': dropout_type, 'dropout_rate': dropout_rate,
                #                        'each_real_CGM_in_batch': each_real_CGM_in_batch.tolist(),
                #                        'CGM_predicted_mean': CGM_predicted_mean.tolist(),
                #                        'CGM_predicted_std': CGM_predicted_std.tolist(),
                #                        'segment_real_CHO': each_real_CHO_in_batch.tolist(),
                #                        'segment_real_insulin': each_real_insulin_in_batch.tolist(),
                #                        'rho_set_CGM': rho_set_CGM.tolist(),
                #                        'rho_orig_CGM': rho_orig_CGM.tolist(), 'loss_one_sample': loss_one_sample}

                segment_result_dict = {'learning_rate': learning_rate, 'epoch': epoch,
                                       'batch_idx': batch_idx, 'batch_num': b, 'N_MC': N_MC, 'conf': conf,
                                       'dropout_type': dropout_type, 'dropout_rate': dropout_rate,
                                       'each_real_acc_in_batch': each_real_acc_in_batch.tolist(),
                                       'acc_predicted_mean': acc_predicted_mean.tolist(),
                                       'acc_predicted_std': acc_predicted_std.tolist(),
                                       'segment_real_acc': each_real_acc_in_batch.tolist(),
                                       'segment_real_delta_acc': each_real_delta_acc_in_batch.tolist(),
                                       'segment_real_speed': each_real_speed_in_batch.tolist(),
                                       'rho_set_acc': rho_set_acc.tolist(),
                                       'rho_orig_acc': rho_orig_acc.tolist(), 'loss_one_sample': loss_one_sample}
                segment_result_df = segment_result_df.append(segment_result_dict, ignore_index=True)

    test_loss /= (len(dataset_loader.dataset) * N_MC)
    test_loss_mean /= (len(dataset_loader.dataset))
    loss_one_sample_average /= (len(dataset_loader.dataset))
    loss_L_SAT /= len(dataset_loader.dataset)
    # print("test loss: %.5f, test loss_mean: %.5f" % (test_loss, test_loss_mean))
    # return loss_one_sample_average, loss_L_SAT, test_loss, test_loss_mean, segment_result_df
    return loss_one_sample_average, test_loss, test_loss_mean, segment_result_df


# eval function for choosing dropout type and rate with Loss Lqt
def eval_model_new_loss(device, dataset_loader, lstm_model, criterion,
                        seq_length, step_look_back, learning_rate, N_MC, batch_size, hidden_size,
                        dropout_rate, dropout_type, requirement_func, conf, mean_df, std_df):
    # evaluate the trained model with dataset
    lstm_model.eval()
    test_loss = 0
    test_loss_mean = 0
    loss_one_sample_average = 0
    loss_L_SAT = 0
    batch_len = 0
    predict_length = seq_length - step_look_back
    # CGM_mean_value = mean_df['CGM']
    # CGM_std_value = std_df['CGM']
    # CHO_mean_value = mean_df['CHO']
    # CHO_std_value = std_df['CHO']
    # insulin_mean_value = mean_df['insulin']
    # insulin_std_value = std_df['insulin']

    acc_mean_value = mean_df['acc']
    acc_std_value = std_df['acc']
    delta_acc_mean_value = mean_df['delta_acc']
    delta_acc_std_value = std_df['delta_acc']
    speed_mean_value = mean_df['speed']
    speed_std_value = std_df['speed']

    with torch.no_grad():
        for batch_idx, datasample in enumerate(dataset_loader):
            data, target0 = datasample[0], datasample[1]
            target = []
            for target_data in datasample[1]:
                target.append(target_data[step_look_back:, :])
            target = torch.stack(target, 0)
            data, target = data.to(device), target.to(device)
            if target0.size(0) == batch_size:
                batch_len = batch_size
            else:
                batch_len = target0.size(0)
            output = torch.zeros(N_MC, batch_len, predict_length, hidden_size)

            # real_whole_CGM_all_batch = target0[:, :, -3:-2] * CGM_std_value + CGM_mean_value
            # real_whole_CGM_future_batch = real_whole_CGM_all_batch[:, step_look_back:, :]
            real_whole_acc_all_batch = target0[:, :, -2:-1] * acc_std_value + acc_mean_value

            new_mean_df = mean_df.reset_index(drop=True)
            new_std_df = std_df.reset_index(drop=True)
            # change target data back to original scale
            for feature_index in range(target.shape[2]):
                target[:, :, feature_index] = (
                            target[:, :, feature_index] * new_std_df.loc[feature_index] + new_mean_df.loc[
                        feature_index]).cpu()

            # apply dropout and loop N_MC times to get a series of predicted traces
            for i in range(N_MC):
                output[i] = (lstm_model.forward_test_with_past(data, step_look_back, hidden_size, dropout_rate,
                                                               dropout_type)).cpu()
                for feature_index in range(output[i].shape[2]):
                    output[i][:, :, feature_index] = (
                                output[i][:, :, feature_index] * new_std_df.loc[feature_index] + new_mean_df.loc[
                            feature_index]).cpu()
                test_loss += criterion(output[i], target.cpu()).item()
            test_loss_mean += criterion(output.mean(dim=0), target.cpu()).item()

            # output_CGM_mean = output.mean(dim=0)[:, :, -3]
            # output_CGM_std = output.std(dim=0)[:, :, -3]
            # output_CGM_target = target[:, :, -3]  # *CGM_std_value+CGM_mean_value
            # trace_CGM = torch.stack((output_CGM_mean, output_CGM_std, output_CGM_target.cpu(),
            #                          torch.zeros(output_CGM_target.size())), dim=-1)

            output_acc_mean = output.mean(dim=0)[:, :, -2]
            output_acc_std = output.std(dim=0)[:, :, -2]
            output_acc_target = target[:, :, -2]  # *CGM_std_value+CGM_mean_value

            trace_acc = torch.stack((output_acc_mean, output_acc_std, output_acc_target.cpu(),
                                     torch.zeros(output_acc_target.size())), dim=-1)
            # record mean and std value for each sample after applying dropout and loop N_MC times
            for b in range(batch_len):

                current_trace_real_acc_all_segment = output_acc_target[b, :]
                for i in range(current_trace_real_acc_all_segment.size(0)):
                    current_trace_real_acc = current_trace_real_acc_all_segment[i].item()
                # get rho interval for ground truth and predictions
                # rho_orig_acc = requirement_func(trace_acc[b, :, -2:], predict_length, conf, lower_acc=-3.4,
                #                                 upper_acc=2.0, func='monitor')
                # rho_set_acc = requirement_func(trace_acc[b, :, :-2], predict_length, conf, lower_acc=-3.4,
                #                                upper_acc=2.0, func='monitor')
                rho_orig_acc = requirement_func(trace_acc[b, :, -2:], predict_length, conf)
                rho_set_acc = requirement_func(trace_acc[b, :, :-2], predict_length, conf)
                ground_truth_trace = trace_acc[b, :, -2:-1]
                predicted_future_mean_trace = trace_acc[b, :, :1]
                predicted_future_std_trace = trace_acc[b, :, 1:2]
                loss_one_sample = loss_func_for_one_flowpipe(rho_orig_acc, rho_set_acc, ground_truth_trace,
                                                             predicted_future_mean_trace,
                                                             predicted_future_std_trace, conf)
                loss_one_sample_average += loss_one_sample
            loss_L_SAT += lsat(trace_acc[:, :, :-2], trace_acc[:, :, -2:], requirement_func, predict_length, conf)

    test_loss /= (len(dataset_loader.dataset) * N_MC)
    test_loss_mean /= (len(dataset_loader.dataset))
    loss_one_sample_average /= (len(dataset_loader.dataset))
    loss_L_SAT /= len(dataset_loader.dataset)
    # print('dropout_rate: ', dropout_rate, ' dropout_type: ', dropout_type,
    #       " New loss_one_sample_average: %.5f" % (loss_L_SAT))
    # return loss_one_sample_average, loss_L_SAT, test_loss, test_loss_mean
    print('dropout_rate: ', dropout_rate, ' dropout_type: ', dropout_type, " New loss_one_sample_average: %.5f" % (loss_one_sample_average))
    return loss_one_sample_average, test_loss, test_loss_mean


# LSAT
## functions for L_SAT
def getdist(tr_pred, tr_orig, conf=0.95):
    ppf = ustl.get_ppf(conf)
    lower = tr_pred[:, :, 0] - ppf * tr_pred[:, :, 1]
    upper = tr_pred[:, :, 0] + ppf * tr_pred[:, :, 1]
    dist = torch.max(torch.max(lower - tr_orig, tr_orig - upper), torch.zeros(lower.size()))
    return dist

def getconfloss(tr_pred, requirement, flag=True):
    # print(requirement(tr_pred, func='eq'))
    strong = confidencelevel.calculatecf_strong(requirement(tr_pred, func='eq'), 0)
    weak = confidencelevel.calculatecf_weak(requirement(tr_pred, func='eq'), 0)
    if flag:
        if strong[1] > 0:
            return 1 - strong[1]
        else:
            return weak[0] + 1
    else:
        strong_ = [1 - weak[1], 1 - weak[0]]
        weak_ = [1 - strong[1], 1 - strong[0]]
        if strong_[1] > 0:
            return 1 - strong_[1]
        else:
            return weak_[0] + 1

def getconfdistloss(tr_pred, tr_orig):
    cdf = scipy.stats.norm.cdf((torch.abs(tr_orig - tr_pred[:, :, 0]) / tr_pred[:, :, 1]))
    return cdf * 2 - 1
def lsat(trace_uncertain, trace_origin, requirement, predict_length, conf, beta_1=0, beta_2=0):
    # for LACC, beta_1=beta_2=0, for LSAT, beta_1=beta_2=0.25
    # For trace, dimension 0 is batchsize.
    # Trace origin is not a pipeline so the std is 0.
    r3 = (getdist(trace_uncertain, trace_origin[:, :, 0]) == 0).float().mean(dim=1).sum().item()
    r1 = 0
    r2 = 0
    for j in range(trace_uncertain.size(0)):
        rho_set = requirement(trace_uncertain[j], trace_len=predict_length, conf=conf)
        rho_orig = requirement(trace_origin[j], trace_len=predict_length, conf=conf)
        if rho_orig[0] < 0:
            flag = False
        else:
            flag = True
        if (rho_set[0] >= 0 and rho_orig[0] >= 0) or (rho_set[1] <= 0 and rho_orig[0] <= 0):
            r1 += 1
        if (rho_set[1] >= 0 and rho_orig[0] >= 0) or (rho_set[0] <= 0 and rho_orig[0] <= 0):
            r2 += 1
    return 1 - (beta_1 * r1 + beta_2 * r2 + (1 - beta_1 - beta_2) * r3)

def comp_tpfp(trace_uncertain, trace_origin, requirement):
    tp, fp, tn, fn = 0, 0, 0, 0
    for j in range(trace_uncertain.size(0)):
        rho_set = requirement(trace_uncertain[j])
        rho_orig = requirement(trace_origin[j])
        if rho_orig[0] < 0:
            flag = False
        else:
            flag = True
        if (rho_set[0] >= 0 and rho_orig[0] >= 0):
            tp += 1
        if (rho_set[1] <= 0 and rho_orig[0] <= 0):
            tn += 1
        if (rho_set[0] <= 0 and rho_orig[0] >= 0):
            fn += 1
        if (rho_set[1] >= 0 and rho_orig[0] <= 0):
            fp += 1
    return tp, fp, tn, fn


# save overall training results and prediction for each sample
def save_results(lstm_model, num_epochs, train_loss, valid_loss, test_loader, requirement_func, dropout_rate_for_test,
                 dropout_type_for_test, conf, learning_rate, output_folder, mean_value_df, std_value_df):
    result_col = ['selected_feature_col', 'seq_length', 'step_look_back', 'predict_len', 'conf',
                  'batch_size', 'N_MC', 'dropout_type', 'dropout_rate', 'learning_rate', 'num_epochs', 'train_loss',
                  'valid_loss',
                  'test_loss', 'loss_one_sample_average']
    result_output_path = '{output_folder}/results_b_{batch_size}_seqlen_{seq_length}_stepback_{step_look_back}_e_{num_epochs}_lr_{learning_rate}_f_{selected_feature_num}_conf_{conf}_dt_{dropout_type}_dr_{dropout_rate}.csv'.format(
        output_folder=output_folder, batch_size=batch_size, seq_length=seq_length, step_look_back=step_look_back,
        num_epochs=num_epochs,
        learning_rate=learning_rate, selected_feature_num=selected_feature_num, dropout_type=dropout_type_for_test,
        dropout_rate=dropout_rate_for_test, conf=conf)
    result_df = pd.DataFrame(columns=result_col)
    result_df.to_csv(result_output_path)

    segment_result_col = ['learning_rate', 'epoch', 'batch_idx', 'batch_num']
    segment_output_file_path = '{output_folder}/segment_results_b_{batch_size}_seqlen_{seq_length}_stepback_{step_look_back}_e_{num_epochs}_lr_{learning_rate}_f_{selected_feature_num}_conf_{conf}_dt_{dropout_type}_dr_{dropout_rate}.csv'.format(
        output_folder=output_folder, batch_size=batch_size, seq_length=seq_length, step_look_back=step_look_back,
        num_epochs=num_epochs, learning_rate=learning_rate, selected_feature_num=selected_feature_num,
        dropout_type=dropout_type_for_test, dropout_rate=dropout_rate_for_test, conf=conf)

    total_segment_result_df = pd.DataFrame(columns=segment_result_col)
    total_segment_result_df.to_csv(segment_output_file_path)
    # evaluate trained model with test set
    print('dropout_type_for_test: ', dropout_type_for_test, ' dropout_rate_for_test: ', dropout_rate_for_test)
    loss_one_sample_average, test_loss, test_loss_mean, segment_result_df = eval_model(device, segment_result_col,
                                                                                       test_loader, lstm_model,
                                                                                       criterion,
                                                                                       seq_length, step_look_back,
                                                                                       num_epochs, learning_rate, N_MC,
                                                                                       batch_size, hidden_size,
                                                                                       dropout_rate_for_test,
                                                                                       dropout_type_for_test,
                                                                                       requirement_func, conf,
                                                                                       mean_value_df, std_value_df)
    result_dict = {'type': 'with_dropout',
                   'selected_feature_col': selected_feature_col, 'seq_length': seq_length,
                   'step_look_back': step_look_back,
                   'predict_len': predict_length, 'conf': conf,
                   'batch_size': batch_size,
                   'N_MC': N_MC, 'dropout_type': dropout_type_for_test, 'dropout_rate': dropout_rate_for_test,
                   'learning_rate': learning_rate, 'num_epochs': num_epochs,
                   'train_loss': train_loss, 'valid_loss': valid_loss,
                   'test_loss': test_loss, 'loss_one_sample_average': loss_one_sample_average,
                   'acc_mean_value': mean_value_df['acc'], 'acc_std_value': std_value_df['acc']}
    result_df = result_df.append(result_dict, ignore_index=True)
    result_df.to_csv(result_output_path)
    total_segment_result_df = total_segment_result_df.append(segment_result_df)
    total_segment_result_df.to_csv(segment_output_file_path)
    # calculate Accuracy & F1
    print('Calculating F1.')
    get_satisfaction_type_all_STLU(segment_output_file_path)
    get_if_warning_label_each_segment_STLU(segment_output_file_path)
    results_dict_one_side = calculate_metrics_STLU(segment_output_file_path)
    results_df_one_side = pd.DataFrame()
    results_one_side_output_path = '{output_folder}/metrics_results_one_side_b_{batch_size}_seqlen_{seq_length}_stepback_{step_look_back}_e_{num_epochs}_lr_{learning_rate}_f_{selected_feature_num}_conf_{conf}_dt_{dropout_type}_dr_{dropout_rate}.csv'.format(
        output_folder=output_folder, batch_size=batch_size, seq_length=seq_length, step_look_back=step_look_back,
        num_epochs=num_epochs, learning_rate=learning_rate, selected_feature_num=selected_feature_num,
        dropout_type=dropout_type_for_test, dropout_rate=dropout_rate_for_test, conf=conf)
    results_df_one_side = results_df_one_side.append(results_dict_one_side, ignore_index=True)
    results_df_one_side.to_csv(results_one_side_output_path)
    return loss_one_sample_average


# STL formula
def requirement_func_always_acc_in_range(signal, trace_len, conf=0.95, lower_acc=-8.0, upper_acc=8.0,
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


# funcs for saving models
def _get_config_file(model_path, model_name):
    # Name of the file for storing hyperparameter details
    return os.path.join(model_path, model_name + ".config")

def _get_model_file(model_path, model_name):
    # Name of the file for storing network parameters
    return os.path.join(model_path, model_name + ".tar")

def load_model(model_path, model_name, net=None):
    """
    Loads a saved model from disk.

    Inputs:
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
        net - (Optional) If given, the state dict is loaded into this model. Otherwise, a new model is created.
    """
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    assert os.path.isfile(config_file), f"Could not find the config file \"{config_file}\". Are you sure this is the correct path and you have your model config stored here?"
    assert os.path.isfile(model_file), f"Could not find the model file \"{model_file}\". Are you sure this is the correct path and you have your model stored here?"
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    if net is None:
        act_fn_name = config_dict["act_fn"].pop("name").lower()
        act_fn = act_fn_by_name[act_fn_name](**config_dict.pop("act_fn"))
        net = BaseNetwork(act_fn=act_fn, **config_dict)
    net.load_state_dict(torch.load(model_file, map_location=device))
    return net

def save_model(model, model_path, model_name):
    """
    Given a model, we save the state_dict and hyperparameters.

    Inputs:
        model - Network object to save parameters from
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
    """
    config_dict = model.config
    os.makedirs(model_path, exist_ok=True)
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    with open(config_file, "w") as f:
        json.dump(config_dict, f)
    torch.save(model.state_dict(), model_file)

# calculate evaluation metrics-with dropout & STLU rho interval
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

def get_satisfaction_type_all_STLU(segment_output_file_path):
    # get all satisfaction type for all rho_set/rho_orig for left and right traces
    # rho set: rho interval for predicted sample
    # rho orig: rho interval for ground truth sample
    segment_df = pd.read_csv(segment_output_file_path, index_col=0)
    satisfaction_type_orig_right = []
    satisfaction_type_predict_right = []
    satisfaction_type_predict_central_signal_right = []
    satisfaction_type_orig_central_signal_right = []

    for index, row in segment_df.iterrows():
        rho_set_central_sigal_right = [float(x) for x in row['rho_set_acc'][1:-1].split(',')]
        rho_orig_central_signal_right = [float(x) for x in row['rho_orig_acc'][1:-1].split(',')]
        satisfaction_type_predict_central_signal_right.append(get_satisfaction_type_one_segment_STLU(rho_set_central_sigal_right))
        satisfaction_type_orig_central_signal_right.append(get_satisfaction_type_one_segment_STLU(rho_orig_central_signal_right))

    segment_df['satisfaction_type_real_acc'] = satisfaction_type_orig_central_signal_right
    segment_df['satisfaction_type_predict_acc'] = satisfaction_type_predict_central_signal_right
    segment_df.to_csv(segment_output_file_path)
    return

def get_if_warning_label_each_segment_STLU(segment_output_file_path):
  # add warning labels according to qual way: 
  # warning: violation & weak satisfaction
  # no warning: strong satisfaction
    segment_df = pd.read_csv(segment_output_file_path, index_col=0)
    if_warning_orig_list = []
    if_warning_predict_list = []
    for index, row in segment_df.iterrows():
        st_central_predict_right = int(row['satisfaction_type_predict_acc'])
        st_central_orig_right = int(row['satisfaction_type_real_acc'])
        if (st_central_orig_right==1 or st_central_orig_right==3):
            if_warning_orig = 1 # warning for st=violation & weak satisfaction
        else:
            if_warning_orig = 0  # no warning for st=strong satisfy
        if (st_central_predict_right==1 or st_central_predict_right==3):
            if_warning_predict = 1 # warning for st=violation & weak satisfaction
        else:
            if_warning_predict = 0  # no warning for st=strong satisfy
        if_warning_orig_list.append(if_warning_orig)
        if_warning_predict_list.append(if_warning_predict)

    segment_df['if_warning_real'] = if_warning_orig_list
    segment_df['if_warning_predict'] = if_warning_predict_list
    segment_df.to_csv(segment_output_file_path)
    return

def calculate_metrics_STLU(segment_output_file_path):
    # calculate F1 and Accuracy with predictions on all samples
    segment_df = pd.read_csv(segment_output_file_path, index_col=0)
    warning_label_df = pd.DataFrame(segment_df,columns=['if_warning_real', 'if_warning_predict'])
    total_num = len(warning_label_df)
    if_warning_real_sum = warning_label_df['if_warning_real'].sum()
    real_warning_percentage = if_warning_real_sum/total_num
    TP, TN, FP, FN = 0, 0, 0, 0
    TPR, TNR, FPR, FNR = 0, 0, 0, 0
    accuracy = 0
    for item, row in warning_label_df.iterrows():
        if_warning_real = int(row['if_warning_real'])
        if_warning_predict = int(row['if_warning_predict'])
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

    results_dict = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'TPR': TPR, 'TNR': TNR, 'FPR': FPR, 'FNR': FNR,
                    'accuracy': accuracy, 'F1_score': F1_score, 'real_warning_percentage':real_warning_percentage}
    print('Metrics STLU one_side: ', results_dict)
    return results_dict

###############################
gpu_avail = torch.cuda.is_available()
print(f"Is the GPU available? {gpu_avail}")
print("Device", device)

input_size = selected_feature_num  # total feature number
hidden_size = selected_feature_num
output_size = selected_feature_num
num_layers = 1
num_classes = selected_feature_num  # total feature number
N_MC = 30  # dropout iteration times
criterion = nn.MSELoss()

dropout_type_for_train = 4
dropout_rate_for_train = 0.9

output_folder = '/home/cpsgroup/predictive_monitor_new_stl/with_dropout_new_stl/new_testing_predict_with_chosen_dropout_LQT/results/behavior_type_{behavior_type}_f_{feature_num}'.format(
    behavior_type=behavior_type, feature_num=selected_feature_num)
mkdir(output_folder)
print('output_folder: ', output_folder)

# load trained lstm model
# behavior type 0 model
b0_lstm_path = '/home/cpsgroup/predictive_monitor_new_stl/lstm_checkpoint/behavior_type_0/lstm_s1_with_dropout_lr_0.001_dt_4_dr_0.9_e_500_behavior_type_0_f_17_state_dict.pt'
# behavior type 2 model
b2_lstm_path = ''
lstm_path_dict = {0: b0_lstm_path, 2:b2_lstm_path}
#dropout_rate_dict = {0: 0.7, 2: 0.6}
#dropout_type_dict = {0: 1, 2: 4}
#dropout_rate_dict = {0: 0.9, 2: 0.9}
#dropout_type_dict = {0: 4, 2: 4}

save_folder = '{CHECKPOINT_PATH}/with_dropout_new_stl'.format(CHECKPOINT_PATH=CHECKPOINT_PATH)
all_mean_value_file_path = '{save_folder}/all_mean_value_{behavior_type}.csv'.format(save_folder=save_folder, behavior_type=behavior_type)
all_std_value_file_path = '{save_folder}/all_std_value_{behavior_type}.csv'.format(save_folder=save_folder,behavior_type=behavior_type)
all_mean_df = pd.read_csv(all_mean_value_file_path, index_col=0)
all_std_df = pd.read_csv(all_std_value_file_path, index_col=0)

print('behavior_type: ', behavior_type)
#dropout_type_list = [4,3,2,1]
#dropout_rate_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
dropout_type_list = [1]
dropout_rate_list = [0.7]
for dropout_type in dropout_type_list:
    for dropout_rate in dropout_rate_list:
        lstm_path = lstm_path_dict[behavior_type]
        with_dropout_model = LSTM(num_classes, input_size, hidden_size, num_layers, step_look_back, train_dropout_type=4)
        with_dropout_model.load_state_dict(torch.load(lstm_path, map_location=device))
        with_dropout_model = with_dropout_model.to(device)
        with_dropout_model.eval()
        #dropout_rate = dropout_rate_dict[behavior_type]
        #dropout_type = dropout_type_dict[behavior_type]
        mean_df = all_mean_df['{behavior_type}'.format(behavior_type=behavior_type)]
        std_df = all_std_df['{behavior_type}'.format(behavior_type=behavior_type)]

        conf=0.95
        learning_rate=0.001
        num_epochs=1
        train_loss=0
        valid_loss=0
        loss_one_sample_average = save_results(with_dropout_model, num_epochs, train_loss, valid_loss, test_loader, requirement_func_always_acc_in_range, dropout_rate,
                         dropout_type, conf, learning_rate, output_folder, mean_df, std_df)



