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
# import STL-U monitor
STLU_path = '/Medical_case/stlu_monitor'
sys.path.append(STLU_path)
import ustlmonitor as ustl
import confidencelevel
import argparse
from collections import namedtuple

# lstm model for prediction
class LSTMCellWithMask(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCellWithMask, self).__init__(input_size, hidden_size, bias=True)
        
    def forward_with_mask(self, input, mask, hx=None):
        (mask_ih, mask_hh) = mask
        # type: (Tensor, Optional[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]
        # self.check_forward_input(input)
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
        self.input_size = input_size
        self.output_size = input_size
        print('train dropout type: ', train_dropout_type)
        # store all hyperparameters in a dictionary for saving and loading of the model
        self.config = {"input_size": self.input_size, "output_size": self.output_size, "hidden_size": self.hidden_size}
      
    def forward(self, input, step_look_back, hidden_size, dropout_rate):
        # funcation for train, dropout type using train_dropout_type
        outputs = []
        device = input.device
        h_t = torch.zeros(input.size(0), self.input_size, dtype=torch.float, device=device)
        c_t = torch.zeros(input.size(0), self.input_size, dtype=torch.float, device=device)

        if self.train_dropout_type == 1:
            #print('type 1: ')
            mask1 = torch.bernoulli(torch.ones(4*hidden_size, self.input_size, dtype=torch.float)*dropout_rate)/dropout_rate  
            mask2 = torch.bernoulli(torch.ones(4*hidden_size, hidden_size, dtype=torch.float)*dropout_rate)/dropout_rate 
        elif self.train_dropout_type == 2:
            #print('type 2: ')
            para = torch.bernoulli(torch.ones(4*hidden_size, self.input_size, dtype=torch.float)*dropout_rate)/dropout_rate 
            mask1 = para
            mask2 = para.expand(-1, hidden_size)
        elif self.train_dropout_type == 3:
            #print('type 3: ')
            p = math.sqrt((1-dropout_rate)/dropout_rate)
            mask1 = torch.normal(1, torch.ones(4*hidden_size, self.input_size, dtype=torch.float)*p)
            mask2 = torch.normal(1, torch.ones(4*hidden_size, hidden_size, dtype=torch.float)*p)
        elif self.train_dropout_type == 4:
            #print('type 4: ')
            p = math.sqrt((1-dropout_rate)/dropout_rate)
            para = torch.normal(1, torch.ones(4*hidden_size, self.input_size, dtype=torch.float)*p)
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
        h_t = torch.zeros(input.size(0), self.input_size, dtype=torch.float, device=device)
        c_t = torch.zeros(input.size(0), self.input_size, dtype=torch.float, device=device)

        if dropout_type == 1:
            #print('type 1: ')
            mask1 = torch.bernoulli(torch.ones(4*hidden_size, self.input_size, dtype=torch.float)*dropout_rate)/dropout_rate  
            mask2 = torch.bernoulli(torch.ones(4*hidden_size, hidden_size, dtype=torch.float)*dropout_rate)/dropout_rate 
        elif dropout_type == 2:
            #print('type 2: ')
            para = torch.bernoulli(torch.ones(4*hidden_size, self.input_size, dtype=torch.float)*dropout_rate)/dropout_rate 
            mask1 = para
            mask2 = para.expand(-1, hidden_size)
        elif dropout_type == 3:
            #print('type 3: ')
            p = math.sqrt((1-dropout_rate)/dropout_rate)
            mask1 = torch.normal(1, torch.ones(4*hidden_size, self.input_size, dtype=torch.float)*p)
            mask2 = torch.normal(1, torch.ones(4*hidden_size, hidden_size, dtype=torch.float)*p)
        elif dropout_type == 4:
            #print('type 4: ')
            p = math.sqrt((1-dropout_rate)/dropout_rate)
            para = torch.normal(1, torch.ones(4*hidden_size, self.input_size, dtype=torch.float)*p)
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

# STL formula for checking CGM: 70<CGM<180
def requirement_func_always_BG_in_range(signal, trace_len, conf, lower_BG=70, upper_BG=180, func='monitor'):
    # signal: CGM trace, keep BG in range
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

from numpy.core.fromnumeric import mean
from simglucose.simulation.user_interface import simulate
from simglucose.controller.base import Controller, Action
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
import numpy as np
import pandas as pd
import pkg_resources
import logging
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
from datetime import timedelta
from datetime import datetime
torch.set_printoptions(precision=4,sci_mode=False)

Observation = namedtuple('Observation', ['CGM'])

logger = logging.getLogger(__name__)
CONTROL_QUEST = pkg_resources.resource_filename('simglucose','params/Quest.csv')
PATIENT_PARA_FILE = pkg_resources.resource_filename('simglucose', 'params/vpatient_params.csv')

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('New folder ok.')
    else:
        print('There is this folder')

    return

def get_ppf(p:float):
	return norm.ppf(p) # get the x value for certain confidence level p

def normalconf(mean, sigma, conf):
	# calculate upper and lower bound using given mean, std, and conf
	p = 1 - (1 - conf) / 2 
	lower = mean - get_ppf(p) * sigma
	upper = mean + get_ppf(p) * sigma
	return (lower, upper)
 
# get upper and lower bound for predicted flowpipe
def get_upper_lower_bound_flowpipe(mean, sigma, conf):
  # calculate upper and lower bound using given mean, std, and conf
  p = 1 - (1 - conf) / 2 
  lower = mean - norm.ppf(p) * sigma # norm.ppf(p):get the x value for certain confidence level p
  upper = mean + norm.ppf(p) * sigma
  return (lower, upper)

# self-defined controller
class NewController(Controller):
    def __init__(self, args_basal_bolus, meal_time_list_df, meal_amount_list_df, target=140):
        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)
        self.target = target
        self.state_list = []
        self.trace_type='history'
        self.correction_bolus_parameter = float(args_basal_bolus.correction_bolus_parameter)
        self.arg_last_low_basal = float(args_basal_bolus.arg_last_low_basal)
        self.arg_last_high_basal = float(args_basal_bolus.arg_last_high_basal)
        self.arg_low_basal = float(args_basal_bolus.arg_low_basal)
        self.arg_high_basal = float(args_basal_bolus.arg_high_basal)
        self.arg_consecutive_step = int(args_basal_bolus.arg_consecutive_step)
        self.arg_violate_threshold = float(args_basal_bolus.arg_violate_threshold)
        self.arg_max_bolus_amount_patient_type = float(args_basal_bolus.arg_max_bolus_amount_patient_type)
        self.arg_past_step_num_severe_low = int(args_basal_bolus.arg_past_step_num_severe_low)
        self.arg_past_step_num_correction_bolus = int(args_basal_bolus.arg_past_step_num_correction_bolus)
        self.arg_severe_high_basal = float(args_basal_bolus.arg_severe_high_basal)
        self.arg_correction_bolus_ahead_step = int(args_basal_bolus.arg_correction_bolus_ahead_step)
        self.arg_start_ahead_step_next_meal_bolus = int(args_basal_bolus.arg_start_ahead_step_next_meal_bolus)
        self.arg_end_ahead_step_next_meal_bolus = int(args_basal_bolus.arg_end_ahead_step_next_meal_bolus)

        self.meal_time_list_df = meal_time_list_df
        self.meal_amount_list_df = meal_amount_list_df
        self.prev_correction_bolus = 0
        self.done_meal_time_list = []
        self.done_meal_amount_list = []

    def get_state_by_rho_and_avegcgm(self, rho_interval, pred_CGM_mean, delta_k_low=20, delta_k_high=70):
      # define states by rho interval and predicted mean value trace
      # s0:strong satisfy
      # s1:weak satisfy
      # s2:mild violation for low BG
      # s3:severe violation for low BG
      # s4:mild violation for high BG
      # s5:severe violation for high BG
      # s6: saved for abnormal case in the trace
      # delta_k_low=20: mild low: 50~70, severe low: <50, delta_k_high=70: mile high: 180~250, severe high: >250
      state = 0
      rho_left = rho_interval[0]
      rho_right = rho_interval[1]
      average_cgm_mean = torch.mean(pred_CGM_mean) # the average of predicted mean values
      if rho_left>self.arg_violate_threshold: # strong satisfy
        state = 0
      elif rho_right<=0 or (rho_left<=self.arg_violate_threshold and rho_right>0): # violation
        if average_cgm_mean<=125: #125=(70+180)/2 # low bg
          if abs(rho_left)<=delta_k_low: # mild violation
            state = 2 
          elif abs(rho_left)>delta_k_low: # severe violation
            state = 4
        elif average_cgm_mean>125: # high bg
          if abs(rho_left)<=delta_k_high: # mild violation
            state = 3 
          elif abs(rho_left)>delta_k_high: # severe violation
            state = 5
      else:
        print('need to check this rho for state.')
        print('rho_interval: ', rho_interval)
        state = 6
      return state  

    def get_violation_state_type(self, predicted_cgm_mean, low_bg=70, high_bg=180):
      # decide the state is Hypo(<70) or Hyper(>180) when the prediction is s2/s3 (violation)
      # predicted_cgm_mean: mean value trace for predicted flowpipe at this step
      # temporary function, need refine
      state_type = 0 # 0: low bg, 1: high bg
      current_CGM = 0 # cgm value calculated from the predicted flowpipe used for making control decidions 
      if abs(torch.mean(predicted_cgm_mean, dim=0)-low_bg)<abs(torch.mean(predicted_cgm_mean, dim=0)-high_bg):
        # low bg
        state_type = 0
        current_CGM = min(predicted_cgm_mean) # how to define the CGM used for future adjustment?
      elif abs(torch.mean(predicted_cgm_mean, dim=0)-low_bg)>=abs(torch.mean(predicted_cgm_mean, dim=0)-high_bg):
        # high bg
        state_type = 1
        current_CGM = max(predicted_cgm_mean)
      return state_type, current_CGM

    def policy(self, past_trace_df, current_rho_set_CGM, current_pred_CGM_mean, current_pred_CGM_std,
               current_pred_CGM_low, current_pred_CGM_up, obs_real, reward, done, **kwargs):
        '''
        past_trace_df: past trace at each step(time, bg, cgm, insulin, risk index, predicted flowpipe mean/std, rho interval for predicted flowpipe)
        current_rho_set_CGM: rho interval for current predicted flowpipe
        current_pred_CGM_mean: mean value for current predicted flowpipe
        obs_real: real observation at current step
        reward, done, **kwargs: parameters from env at current step
        '''
        sample_time = kwargs.get('sample_time', 1)
        pname = kwargs.get('patient_name')
        meal = kwargs.get('meal')  # unit: g/min
        current_time = kwargs.get('time')
        if_warning_this_step = 0
        if_warning_eat = 0
        basal_severe_low_adjust = 0
        basal_severe_high_not_last_adjust = 0
        basal_severe_high_last_adjust = 0
        basal_mild_low_not_last_adjust = 0
        basal_mild_low_last_adjust = 0
        basal_mild_high_not_last_adjust = 0
        basal_mild_high_last_adjust = 0
        bolus_correction_adjust = 0
        last_bolus_time = 0
        if meal > 0:
            meal_bolus_adjust = 1
        else:
            meal_bolus_adjust = 0
        # get current/last states
        past_step_num = self.arg_consecutive_step  # steps for consecutive check
        past_step_num_severe = 10  # steps for consecutive check on severe case
        past_step_num_for_eat = 2  # steps for consecutive check on severe case
        low_bg = 70
        high_bg = 180
        last_rho_set_CGM = [float(x) for x in past_trace_df['rho_set_CGM'].to_list()[
            -1]]  # rho interval for predicted flowpipe at last step
        last_CGM_mean = torch.Tensor([float(x) for x in past_trace_df['predicted_CGM_mean'].to_list()[
            -1]])  # cgm mean value trace for predicted flowpipe at last step
        current_state = self.get_state_by_rho_and_avegcgm(current_rho_set_CGM, current_pred_CGM_mean)
        self.state_list.append(current_state)
        past_state_list = self.state_list[-past_step_num:]  # states of last 5 steps
        past_state_list_severe = self.state_list[-past_step_num_severe:]
        past_state_list_for_eat = self.state_list[-past_step_num_for_eat:]
        # set actions by state transition, some parts need further refine
        if current_state == 4:  # servere low violation, take new action immediately
            if_warning_eat = 1
            if_warning_this_step = 4
            # decrease basal to 0 only one time in 5 steps
            past_basal_severe_low_adjust_list = past_trace_df[-self.arg_past_step_num_severe_low:]['basal_severe_low_adjust'].to_list()
            if past_basal_severe_low_adjust_list.count(1)!=0:
                basal_severe_low_adjust = 0
                current_CGM = torch.Tensor([obs_real.CGM])
                action, (basal, correction_bolus, meal_bolus,max_total_bolus_trigger_flag) = self._new_policy(pname, meal, current_CGM, sample_time,
                                                                                                              past_trace_df,current_time,current_state,
                                          bg_type='normal')  # adjust insulin with self-defined new policy
            else:
                basal_severe_low_adjust = 1
                current_CGM = torch.Tensor([obs_real.CGM])
                action, (basal, correction_bolus, meal_bolus,max_total_bolus_trigger_flag) = self._new_policy(pname, meal, current_CGM, sample_time,
                                                                                                              past_trace_df,current_time,current_state,
                                          bg_type='severe_low')  # adjust insulin with self-defined new policy
        elif current_state == 5:  # servere high violation, take new action by checking on 2 steps
            # if_warning_this_step = 5
            if_warning_eat = 0
            past_state_list_severe = self.state_list[-past_step_num_severe:]  # states of last 5 steps
            current_CGM = torch.Tensor([obs_real.CGM])
            basal_severe_high_last_adjust = 0
            basal_severe_high_not_last_adjust = 1
            bolus_correction_adjust = 0
            if_warning_this_step = 5
            action, (basal, correction_bolus, meal_bolus, max_total_bolus_trigger_flag) = self._new_policy(pname, meal,
                                                                                                           current_CGM,
                                                                                                           sample_time,
                                                                                                           past_trace_df,current_time,current_state,
                                                                                                           bg_type='severe_high')  # adjust insulin with self-defined new policy

        elif current_state == 2:  # mild violation, low bg, consecutive monitor
            if_warning_eat = 0
            current_CGM = torch.Tensor([obs_real.CGM])
            basal_mild_low_not_last_adjust = 1
            if_warning_this_step = 2
            action, (basal, correction_bolus, meal_bolus,max_total_bolus_trigger_flag) = self._new_policy(pname, meal, current_CGM, sample_time,past_trace_df,current_time,current_state,
                                      bg_type='low')  # adjust insulin with self-defined new policy
        elif current_state == 3:  # mild violation, high bg, consecutive monitor
            if_warning_eat = 0
            current_CGM = torch.Tensor([obs_real.CGM])
            basal_mild_high_not_last_adjust = 1
            if_warning_this_step = 3
            action, (basal, correction_bolus, meal_bolus,max_total_bolus_trigger_flag) = self._new_policy(pname, meal, current_CGM, sample_time,past_trace_df,current_time,current_state,
                                      bg_type='high')  # adjust insulin with self-defined new policy
        elif current_state == 6:
            print('Need check on this state.')
            if_warning_this_step = 7
            if_warning_eat = 7
            action = None
        else:  # other state transitions, take normal control(bb_controller)
            current_CGM = torch.Tensor([obs_real.CGM])
            action, (basal, correction_bolus, meal_bolus,max_total_bolus_trigger_flag) = self._new_policy(pname, meal, current_CGM, sample_time,past_trace_df,current_time,current_state,
                                      bg_type='normal')  # adjust insulin with self-defined new policy
            if_warning_this_step = 0  # if_warning_this_step_bb
            if_warning_eat = 0

        basal_adjust_mark_dict = {'basal_severe_low_adjust': basal_severe_low_adjust,
                                  'basal_severe_high_not_last_adjust': basal_severe_high_not_last_adjust,
                                  'basal_severe_high_last_adjust': basal_severe_high_last_adjust,
                                  'basal_mild_low_not_last_adjust': basal_mild_low_not_last_adjust,
                                  'basal_mild_low_last_adjust': basal_mild_low_last_adjust,
                                  'basal_mild_high_not_last_adjust': basal_mild_high_not_last_adjust,
                                  'basal_mild_high_last_adjust': basal_mild_high_last_adjust,
                                  'bolus_correction_adjust': bolus_correction_adjust,
                                  'meal_bolus_adjust': meal_bolus_adjust}

        return action, if_warning_this_step, if_warning_eat, basal_adjust_mark_dict, (basal, correction_bolus, meal_bolus,max_total_bolus_trigger_flag)

    def _new_policy(self, name, meal, glucose, env_sample_time, past_trace_df, current_time, current_state, bg_type='low'):

        if any(self.quest.Name.str.match(name)):
            quest = self.quest[self.quest.Name.str.match(name)]
            params = self.patient_params[self.patient_params.Name.str.match(
                name)]
            u2ss = params.u2ss.values.item()  # unit: pmol/(L*kg)
            BW = params.BW.values.item()  # unit: kg
        else:
            quest = pd.DataFrame([['Average', 1 / 15, 1 / 50, 50, 30]],
                                 columns=['Name', 'CR', 'CF', 'TDI', 'Age'])
            u2ss = 1.43  # unit: pmol/(L*kg) #this is the steady state insulin rate per kg
            BW = 57.0  # unit: kg

        default_basal = u2ss * BW / 6000  # unit: U/min
        bolus = 0

        # Calculate basal based on conditions
        if bg_type == 'severe_low':
            # temporary basal suspension
            basal = 0
        elif bg_type == 'severe_high':
            basal = default_basal * self.arg_severe_high_basal
        elif bg_type == 'lasting_low':
            print("eat")  # tell patient to eat
            # reduce basal slightly
            basal = default_basal * self.arg_last_low_basal
        elif bg_type == 'lasting_high':
            # increase insulin rate more severely by 20%
            basal = default_basal * self.arg_last_high_basal
        elif bg_type == 'low':
            # print("eat")  # tell patient to eat
            basal = default_basal * self.arg_low_basal
        elif bg_type == 'high':
            # increase insulin rate by 20%
            basal = default_basal * self.arg_high_basal
        else:  # normal control
            basal = default_basal

        # adapt meal bolus ahead step
        orig_meal_time_list = self.meal_time_list_df[name].to_list()
        orig_meal_amount_list = self.meal_amount_list_df[name].to_list()
        meal_time_list = [x for x in orig_meal_time_list if x == x]
        meal_amount_list = [x for x in orig_meal_amount_list if x == x]
        # get time and amount of next meal
        rest_meal_time_list = list(set(meal_time_list).difference(set(self.done_meal_time_list)))
        rest_meal_time_list.sort(key=meal_time_list.index)
        if rest_meal_time_list != []:
            next_meal_time = datetime.strptime(rest_meal_time_list[0], "%Y-%m-%d %H:%M:%S")
            next_meal_time_str = next_meal_time.strftime("%Y-%m-%d %H:%M:%S")
            index_orig = meal_time_list.index(next_meal_time_str)
            next_meal_amount = meal_amount_list[index_orig]
        else:
            next_meal_time = current_time + timedelta(days=30)
            next_meal_time_str = next_meal_time.strftime("%Y-%m-%d %H:%M:%S")
            next_meal_amount = 0

        # get start & end time for giving meal bolus for next meal
        start_time_next_meal_bolus = next_meal_time - timedelta(minutes=self.arg_start_ahead_step_next_meal_bolus * 3)
        end_time_next_meal_bolus = next_meal_time - timedelta(minutes=self.arg_end_ahead_step_next_meal_bolus * 3)
        if start_time_next_meal_bolus <= current_time <= end_time_next_meal_bolus:
            if ((glucose.item()>=180 or (current_state==3 or current_state==5)) or (70<glucose.item()<180 and current_state==0) or (current_time == end_time_next_meal_bolus)):
                meal_bolus = (
                        (next_meal_amount * env_sample_time) / quest.CR.values + self.correction_bolus_parameter * (
                        glucose.item() > 150) *
                        (glucose.item() - self.target) / quest.CF.values).item()  # unit: U
                meal_bolus = meal_bolus / env_sample_time  # unit: U/min
                self.done_meal_time_list.append(next_meal_time_str)
                self.done_meal_amount_list.append(next_meal_amount)
            else:
                meal_bolus = 0
        else:
            meal_bolus = 0

        # constrain on max total bolus in past 1h
        past_meal_bolus_list = past_trace_df['meal_bolus'][-20:].to_list()
        past_correction_bolus_list = past_trace_df['correction_bolus'][-20:].to_list()
        sum_past_meal_bolus = sum(past_meal_bolus_list)
        sum_past_correction_bolus = sum(past_correction_bolus_list)
        final_bolus = bolus + meal_bolus
        if self.arg_max_bolus_amount_patient_type - (sum_past_meal_bolus+sum_past_correction_bolus)>= final_bolus:
            final_bolus = final_bolus
            max_total_bolus_trigger_flag = 0
        else:
            final_bolus = self.arg_max_bolus_amount_patient_type - (sum_past_meal_bolus+sum_past_correction_bolus)
            meal_bolus = 0
            bolus = final_bolus
            max_total_bolus_trigger_flag = 1

        return Action(basal=basal, bolus=final_bolus), (basal, bolus, meal_bolus,max_total_bolus_trigger_flag)

    def _bb_policy(self, name, meal, glucose, env_sample_time, current_state_from_main_controller):
        """
        Provided by simglucose, used for normal control, temporary
        Helper function to compute the basal and bolus amount.

        The basal insulin is based on the insulin amount to keep the blood
        glucose in the steady state when there is no (meal) disturbance. 
               basal = u2ss (pmol/(L*kg)) * body_weight (kg) / 6000 (U/min)
        
        The bolus amount is computed based on the current glucose level, the
        target glucose level, the patient's correction factor and the patient's
        carbohydrate ratio.
               bolus = ((carbohydrate / carbohydrate_ratio) + 
                       (current_glucose - target_glucose) / correction_factor)
                       / sample_time
        NOTE the bolus computed from the above formula is in unit U. The
        simulator only accepts insulin rate. Hence the bolus is converted to
        insulin rate.
        """
        if any(self.quest.Name.str.match(name)):
            quest = self.quest[self.quest.Name.str.match(name)]
            params = self.patient_params[self.patient_params.Name.str.match(
                name)]
            u2ss = params.u2ss.values.item()  # unit: pmol/(L*kg)
            BW = params.BW.values.item()  # unit: kg
        else:
            quest = pd.DataFrame([['Average', 1 / 15, 1 / 50, 50, 30]],
                                 columns=['Name', 'CR', 'CF', 'TDI', 'Age'])
            u2ss = 1.43  # unit: pmol/(L*kg)
            BW = 57.0  # unit: kg

        basal = u2ss * BW / 6000  # unit: U/min
        if meal > 0:
            logger.info('Calculating bolus ...')
            logger.info(f'Meal = {meal} g/min')
            logger.info(f'glucose = {glucose.CGM}')
            bolus = (
                (meal * env_sample_time) / quest.CR.values + (glucose.CGM.item() > 150) *
                (glucose.CGM.item() - self.target) / quest.CF.values).item()  # unit: U
        else:
            bolus = 0  # unit: U

        # This is to convert bolus in total amount (U) to insulin rate (U/min).
        # The simulation environment does not treat basal and bolus
        # differently. The unit of Action.basal and Action.bolus are the same
        # (U/min).
        bolus = bolus / env_sample_time  # unit: U/min

        if 50<glucose.CGM<=70: # mild low bg
          if_warning_this_step = 2
          current_state_bb = 2
        elif glucose.CGM<=50: # severe low bg
          if_warning_this_step = 4
          current_state_bb = 4
        elif 250>glucose.CGM>=180: # mild high bg
          if_warning_this_step = 3
          current_state_bb = 3
        elif glucose.CGM>=250: # severe high bg
          if_warning_this_step = 5
          current_state_bb = 5
        elif 180>glucose.CGM>70: # in range
          if_warning_this_step = 0
          current_state_bb = 0
        if self.trace_type=='history':
          self.state_list.append(current_state_bb)
        elif self.trace_type=='future':
          pass
        if meal>0:
          meal_bolus_adjust=1
        else:
          meal_bolus_adjust=0
        basal_adjust_mark_dict = {'basal_severe_low_adjust':0,
        'basal_severe_high_not_last_adjust':0,
        'basal_severe_high_last_adjust':0, 
        'basal_mild_low_not_last_adjust':0,
        'basal_mild_low_last_adjust':0,
        'basal_mild_high_not_last_adjust':0,
        'basal_mild_high_last_adjust':0,
        'bolus_correction_adjust': 0,
        'meal_bolus_adjust':meal_bolus_adjust}
        if_warning_eat_this_step = 0
        return Action(basal=basal, bolus=bolus), if_warning_this_step, if_warning_eat_this_step, basal_adjust_mark_dict, (basal, 0, bolus, 0)

    def reset(self):
        pass


# get basal bolus parameters
parser = argparse.ArgumentParser(description='basal bolus argparse')
parser.add_argument('--arg_id', '-arg_id', help='id of this set of params', required=True)
parser.add_argument('--arg_cuda', '-arg_cuda', help='cuda id', required=True)
parser.add_argument('--arg_patient', '-arg_patient', help='patient type', required=True)
parser.add_argument('--arg_day', '-arg_day', help='simulation days', required=True)
parser.add_argument('--arg_consecutive_step', '-arg_consecutive_step', help='consecutive step number for lasting case', required=True)
parser.add_argument('--arg_violate_threshold', '-arg_violate_threshold', help='threshold for deciding violating states', default=0, required=True)
parser.add_argument('--arg_last_low_basal', '-last_low_basal', help='reduce basal for lasting low case', required=True)
parser.add_argument('--arg_last_high_basal', '-last_high_basal', help='increase basal for lasting high case', required=True)
parser.add_argument('--arg_low_basal', '-low_basal', help='reduce basal for low case', required=True)
parser.add_argument('--arg_high_basal', '-high_basal', help='increase basal for high case', required=True)
parser.add_argument('--correction_bolus_parameter', '-correction_bolus', help='correction bolus param for severe high case', required=True)
parser.add_argument('--arg_max_bolus_amount_patient_type', '-arg_max_bolus_amount_patient_type', help='max total bolus past 1h for this patient type', required=True)
parser.add_argument('--arg_past_step_num_severe_low', '-arg_past_step_num_severe_low', help='past steps for setting basal=0 for severe low case', required=True)
parser.add_argument('--arg_past_step_num_correction_bolus', '-arg_past_step_num_correction_bolus', help='past steps for checking correction bolus', required=True)
parser.add_argument('--arg_severe_high_basal', '-arg_severe_high_basal', help='reduce basal for severe high case', required=True)
parser.add_argument('--arg_correction_bolus_ahead_step', '-arg_correction_bolus_ahead_step', help='ahead steps for giving correction bolus', required=True)
parser.add_argument('--arg_start_ahead_step_next_meal_bolus', '-arg_start_ahead_step_next_meal_bolus', help='ahead steps (start) for giving meal bolus', required=True)
parser.add_argument('--arg_end_ahead_step_next_meal_bolus', '-arg_end_ahead_step_next_meal_bolus', help='ahead steps (end) for giving meal bolus', required=True)

args_basal_bolus = parser.parse_args()

# set parameters
lstm_type_1='lstm_with_monitor'
lstm_type_2='lstm_no_monitor'
lstm_type_3='no_lstm'

lstm_type_list = ['lstm_with_monitor']

patient_type_list = [args_basal_bolus.arg_patient]
patient_id_num_list = ["%03d" % x for x in range(1,11)]

selected_feature_col = ['hour', 'minute', 'LBGI', 'HBGI', 'Risk', 'CGM', 'CHO', 'insulin']
num_feature = len(selected_feature_col)
num_classes=num_feature
input_size=num_feature
hidden_size=num_feature
num_layers=1
step_look_back=10 
train_dropout_type=4
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:{cuda_id}".format(cuda_id=args_basal_bolus.arg_cuda))
batch_size = 1
predict_length = 10
N_MC = 30
day_num = int(args_basal_bolus.arg_day)
t_history = 20 #step, 1h
conf = 0.95

# load trained lstm model
# adult model
adult_lstm_path = '/Medical_case/lstm_model/lstm_s1_with_dropout_lr_0.01_dt_4_dr_0.9_e_50_3_m_8_f_adult_patient_state_dict.pt'
# child model
child_lstm_path = '/Medical_case/lstm_model/lstm_s1_with_dropout_lr_0.01_dt_4_dr_0.9_e_50_3_m_8_f_child_patient_state_dict.pt'
# adolescent model
adolescent_lstm_path = '/Medical_case/lstm_model/lstm_s1_with_dropout_lr_0.001_dt_4_dr_0.9_e_50_3_m_8_f_adolescent_patient_state_dict.pt'
lstm_path_dict = {'adult': adult_lstm_path, 'child':child_lstm_path, 'adolescent':adolescent_lstm_path}
dropout_rate_dict = {'adult':0.8, 'child':0.9, 'adolescent':0.9}
dropout_type_dict = {'adult':2, 'child':3, 'adolescent':2}

all_mean_value_file_path = '/Medical_case/medical_case_pipeline/all_mean_value.csv'
all_std_value_file_path = '/Medical_case/medical_case_pipeline/all_std_value.csv'
all_meal_time_value_file_path = '/Medical_case/medical_case_pipeline/all_meal_time.csv'
all_meal_amount_value_file_path = '/Medical_case/medical_case_pipeline/all_meal_amount.csv'
all_mean_df = pd.read_csv(all_mean_value_file_path, index_col=0)
all_std_df = pd.read_csv(all_std_value_file_path, index_col=0)
all_meal_time_df = pd.read_csv(all_meal_time_value_file_path, index_col=0)
all_meal_amount_df = pd.read_csv(all_meal_amount_value_file_path, index_col=0)

for patient_type in patient_type_list:
  for patient_id in patient_id_num_list: 
    print('patient_type: ', patient_type, ' patient_id: ', patient_id)
    lstm_path = lstm_path_dict[patient_type]
    with_dropout_model = LSTM(num_classes, input_size, hidden_size, num_layers, step_look_back, train_dropout_type)
    with_dropout_model.load_state_dict(torch.load(lstm_path, map_location=torch.device('cpu')))
    with_dropout_model.eval()
    with_dropout_lstm_path_1 = lstm_path
    with_dropout_model_1 = LSTM(num_classes, input_size, hidden_size, num_layers, step_look_back, train_dropout_type)
    with_dropout_model_1.load_state_dict(torch.load(lstm_path, map_location=torch.device('cpu')))
    with_dropout_model_1.eval()
    dropout_rate = dropout_rate_dict[patient_type]
    dropout_type = dropout_type_dict[patient_type]
    mean_df = all_mean_df[patient_type]
    std_df = all_std_df[patient_type]
    for lstm_type in lstm_type_list:
          print('lstm_type: ', lstm_type)
          simglucose_test_results_path = '/Medical_case/medical_case_pipeline/results/{patient_type}_patient/{patient_num}_patient_{day}_day_{arg_id}/{lstm_type}'.format(lstm_type=lstm_type,
                                                                    patient_num=len(patient_id_num_list),day=day_num, arg_id=args_basal_bolus.arg_id, patient_type=patient_type)
          mkdir(simglucose_test_results_path)

          # specify start_time as the beginning of today
          now = datetime(2022, 10, 3)
          print('date now: ', now)
          start_time = datetime.combine(now.date(), datetime.min.time())

          # --------- Create Random Scenario --------------
          # Specify results saving path
          path = simglucose_test_results_path

          # Create a simulation environment
          # set scenario seed for different patients
          if patient_type=='adult':
              scenario_seed = int(patient_id[-2:])
          elif patient_type=='child':
              scenario_seed = int(patient_id[-2:])+10
          elif patient_type=='adolescent':
              scenario_seed = int(patient_id[-2:])+20

          patient = T1DPatient.withName('{patient_type}#{patient_id}'.format(patient_type=patient_type, patient_id=patient_id))
          sensor = CGMSensor.withName('Dexcom', seed=1)
          pump = InsulinPump.withName('Insulet')
          scenario = RandomScenario(start_time=start_time, seed=scenario_seed)
          env = T1DSimEnv(patient, sensor, pump, scenario)

          # Create a controller
          print('Basal Bolus param setting: ', args_basal_bolus)
          new_controller_1 = NewController(args_basal_bolus,all_meal_time_df,all_meal_amount_df)
          # Choose model
          if lstm_type=='lstm_with_monitor':
            lstm_model=with_dropout_model
          elif lstm_type=='lstm_no_monitor':
            lstm_model=with_dropout_model_1
          elif lstm_type=='no_lstm':
            lstm_model=with_dropout_model_1 # but not use the model
          else:
            print('check lstm_type.')
            lstm_model = None
          # Put them together to create a simulation object
          s1 = SimObj(env, new_controller_1, timedelta(days=day_num), animate=True, path=path, lstm_model=lstm_model, requirement_func=requirement_func_always_BG_in_range)
          results1 = sim(s1, simglucose_test_results_path, t_history, mean_df, std_df, dropout_rate, dropout_type, conf, N_MC, lstm_type)
          print(results1)

