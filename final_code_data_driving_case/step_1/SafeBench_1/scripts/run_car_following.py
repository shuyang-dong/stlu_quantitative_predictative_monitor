''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-22 18:45:01
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import traceback
import os.path as osp

import torch 

from safebench.util.run_util import load_config
from safebench.util.torch_util import set_seed, set_torch_variable
from safebench.carla_runner import CarlaRunner

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import torch._VF as _VF

import pandas as pd
import os
import math
from scipy.stats import norm

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
        super(LSTM, self).__init__()
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
            # print('type 1: ')
            mask1 = torch.bernoulli(
                torch.ones(4 * hidden_size, self.input_size, dtype=torch.float) * dropout_rate) / dropout_rate
            mask2 = torch.bernoulli(
                torch.ones(4 * hidden_size, hidden_size, dtype=torch.float) * dropout_rate) / dropout_rate
        elif self.train_dropout_type == 2:
            # print('type 2: ')
            para = torch.bernoulli(
                torch.ones(4 * hidden_size, self.input_size, dtype=torch.float) * dropout_rate) / dropout_rate
            mask1 = para
            mask2 = para.expand(-1, hidden_size)
        elif self.train_dropout_type == 3:
            # print('type 3: ')
            p = math.sqrt((1 - dropout_rate) / dropout_rate)
            mask1 = torch.normal(1, torch.ones(4 * hidden_size, self.input_size, dtype=torch.float) * p)
            mask2 = torch.normal(1, torch.ones(4 * hidden_size, hidden_size, dtype=torch.float) * p)
        elif self.train_dropout_type == 4:
            # print('type 4: ')
            p = math.sqrt((1 - dropout_rate) / dropout_rate)
            para = torch.normal(1, torch.ones(4 * hidden_size, self.input_size, dtype=torch.float) * p)
            mask1 = para
            mask2 = para.expand(-1, hidden_size)
        else:
            print("Please select the correct DROPOUT_TYPE: 1-4")
        mask = (mask1.to(device), mask2.to(device))

        sequence_length = input.size(1)
        predict_length = sequence_length - step_look_back
        for i in range(input.size(1)):
            if i <= step_look_back - 1:
                h_t, c_t = self.lstm.forward_with_mask(input[:, i, :], mask, (h_t, c_t))
                output = self.linear(h_t)
                outputs.append(output)
            else:
                h_t, c_t = self.lstm.forward_with_mask(outputs[i - 1], mask, (h_t, c_t))
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
            # print('type 1: ')
            mask1 = torch.bernoulli(
                torch.ones(4 * hidden_size, self.input_size, dtype=torch.float) * dropout_rate) / dropout_rate
            mask2 = torch.bernoulli(
                torch.ones(4 * hidden_size, hidden_size, dtype=torch.float) * dropout_rate) / dropout_rate
        elif dropout_type == 2:
            # print('type 2: ')
            para = torch.bernoulli(
                torch.ones(4 * hidden_size, self.input_size, dtype=torch.float) * dropout_rate) / dropout_rate
            mask1 = para
            mask2 = para.expand(-1, hidden_size)
        elif dropout_type == 3:
            # print('type 3: ')
            p = math.sqrt((1 - dropout_rate) / dropout_rate)
            mask1 = torch.normal(1, torch.ones(4 * hidden_size, self.input_size, dtype=torch.float) * p)
            mask2 = torch.normal(1, torch.ones(4 * hidden_size, hidden_size, dtype=torch.float) * p)
        elif dropout_type == 4:
            # print('type 4: ')
            p = math.sqrt((1 - dropout_rate) / dropout_rate)
            para = torch.normal(1, torch.ones(4 * hidden_size, self.input_size, dtype=torch.float) * p)
            mask1 = para
            mask2 = para.expand(-1, hidden_size)
        else:
            print("Please select the correct DROPOUT_TYPE: 1-4")
        mask = (mask1.to(device), mask2.to(device))

        sequence_length = input.size(1)
        predict_length = sequence_length - step_look_back
        for i in range(input.size(1)):
            if i <= step_look_back - 1:
                h_t, c_t = self.lstm.forward_with_mask(input[:, i, :], mask, (h_t, c_t))
                output = self.linear(h_t)
                outputs.append(output)
            else:
                h_t, c_t = self.lstm.forward_with_mask(outputs[i - 1], mask, (h_t, c_t))
                output = self.linear(h_t)
                outputs.append(output)

        outputs = torch.stack(outputs[step_look_back:], 1).squeeze(2)
        return outputs

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('New folder ok.')
    else:
        print('There is this folder')

    return

def get_ppf(p: float):
    return norm.ppf(p)  # get the x value for certain confidence level p

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
    lower = mean - norm.ppf(p) * sigma  # norm.ppf(p):get the x value for certain confidence level p
    upper = mean + norm.ppf(p) * sigma
    return (lower, upper)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--output_dir', type=str, default='log')
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__)))))
    print(osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__)))))

    parser.add_argument('--max_episode_step', type=int, default=500)
    parser.add_argument('--auto_ego', action='store_true')
    parser.add_argument('--mode', '-m', type=str, default='eval', choices=['train_agent', 'train_scenario', 'eval'])
    parser.add_argument('--agent_cfg', nargs='*', type=str, default='behavior.yaml')
    parser.add_argument('--scenario_cfg', nargs='*', type=str, default='standard.yaml')
    parser.add_argument('--continue_agent_training', '-cat', type=bool, default=False)
    parser.add_argument('--continue_scenario_training', '-cst', type=bool, default=False)

    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')   

    parser.add_argument('--num_scenario', '-ns', type=int, default=1, help='num of scenarios we run in one episode')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--frame_skip', '-fs', type=int, default=1, help='skip of frame in each step')
    parser.add_argument('--port', type=int, default=2000, help='port to communicate with carla')
    parser.add_argument('--tm_port', type=int, default=8000, help='traffic manager port')
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)
    parser.add_argument('--behavior_type', type=int, default=0) #behavior agent mode, 0:cautious, 2:aggressive
    parser.add_argument('--if_new_controller', type=int,default=0) # if using predictive monitor and new controller, 1 yes, 0 no
    parser.add_argument('--control_type', type=str, default='no_lstm')  # no_lstm, lstm_with_controller, lstm_no_controller
    args = parser.parse_args()
    args_dict = vars(args)

    # set predictive monitor params
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device(
        "cuda:{cuda_id}".format(cuda_id=0))

    # load trained lstm model
    behavior_type = args.behavior_type
    # cautious behavior agent lstm model
    cautious_lstm_path = '/home/safebench/predictive_monitor/lstm_checkpoint/behavior_type_0/lstm_s1_with_dropout_lr_0.001_dt_4_dr_0.9_e_150_behavior_type_0_f_17_state_dict.pt'
    # aggressive behavior agent lstm model
    aggressive_lstm_path = '/home/safebench/predictive_monitor/lstm_checkpoint/behavior_type_2/lstm_s1_with_dropout_lr_0.001_dt_4_dr_0.9_e_250_behavior_type_2_f_17_state_dict.pt'
    lstm_path_dict = {0: cautious_lstm_path, 2: aggressive_lstm_path}
    dropout_rate_dict = {0: 0.7, 2: 0.6}
    dropout_type_dict = {0: 1, 2: 4}

    selected_feature_col = ['v_x', 'v_y', 'lateral_dis', '-delta_yaw', 'vehicle_front', 'acc_x', 'acc_y', 'acc_z',
                            'ego_yaw', 'radar_obj_depth_aveg', 'radar_obj_velocity_aveg',
                            'throttle', 'steer', 'brake', 'speed', 'acc', 'delta_acc']
    learning_rate_list = [0.01]
    input_size = len(selected_feature_col)  # total feature number
    hidden_size = len(selected_feature_col)
    output_size = len(selected_feature_col)
    num_layers = 1
    num_classes = len(selected_feature_col)  # total feature number
    N_MC = 30  # dropout iteration times
    batch_size = 1

    criterion = nn.MSELoss()
    mean_df = None
    std_df = None
    # List for choosing dropout parameters with Lqt
    train_dropout_type = 4
    dropout_type_for_train = 4
    dropout_rate_for_train = 0.9
    conf_list = [0.95]
    conf = 0.95
    dropout_type_list = [4, 3, 2, 1]
    dropout_rate_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    dropout_rate = 0.9
    dropout_type = 4

    seq_length = 50
    step_look_back = 30
    predict_length = seq_length-step_look_back

    lstm_type = args.control_type
    start_episode = 0
    if_new_controller = args.if_new_controller

    err_list = []
    for agent_cfg in args.agent_cfg:
        for scenario_cfg in args.scenario_cfg:
            # set global parameters
            set_torch_variable(args.device)
            torch.set_num_threads(args.threads)
            set_seed(args.seed)

            # load agent config
            agent_config_path = osp.join(args.ROOT_DIR, 'safebench/agent/config', agent_cfg)
            agent_config = load_config(agent_config_path)

            # load scenario config
            scenario_config_path = osp.join(args.ROOT_DIR, 'safebench/scenario/config', scenario_cfg)
            scenario_config = load_config(scenario_config_path)

            # main entry with a selected mode
            agent_config.update(args_dict)
            scenario_config.update(args_dict)
            agent_config['behavior_type'] = args.behavior_type
            if scenario_config['policy_type'] == 'scenic':
                from safebench.scenic_runner import ScenicRunner
                assert scenario_config['num_scenario'] == 1, 'the num_scenario can only be one for scenic now'
                runner = ScenicRunner(agent_config, scenario_config)
            else:
                print('scenario_config: ', scenario_config)
                ####
                if if_new_controller:
                    all_mean_value_file_path = '/home/safebench/predictive_monitor/lstm_checkpoint/behavior_type_{behavior_type}/with_dropout/all_mean_value_{behavior_type}.csv'.format(
                        behavior_type=behavior_type)
                    all_std_value_file_path = '/home/safebench/predictive_monitor/lstm_checkpoint/behavior_type_{behavior_type}/with_dropout/all_std_value_{behavior_type}.csv'.format(
                        behavior_type=behavior_type)
                    all_mean_df = pd.read_csv(all_mean_value_file_path, index_col=0)
                    all_std_df = pd.read_csv(all_std_value_file_path, index_col=0)

                    lstm_path = lstm_path_dict[behavior_type]
                    with_dropout_model = LSTM(num_classes, input_size, hidden_size, num_layers, step_look_back,
                                              train_dropout_type)
                    with_dropout_model.load_state_dict(torch.load(lstm_path, map_location=torch.device('cpu')))
                    with_dropout_model.eval()
                    with_dropout_lstm_path_1 = lstm_path
                    with_dropout_model_1 = LSTM(num_classes, input_size, hidden_size, num_layers, step_look_back,
                                                train_dropout_type)
                    with_dropout_model_1.load_state_dict(torch.load(lstm_path, map_location=torch.device('cpu')))
                    with_dropout_model_1.eval()
                    dropout_rate = dropout_rate_dict[behavior_type]
                    dropout_type = dropout_type_dict[behavior_type]

                    mean_df = all_mean_df['0'] # col named "0"
                    std_df = all_std_df['0']

                    # Choose model
                    if lstm_type == 'lstm_with_monitor':
                        lstm_model = with_dropout_model
                    elif lstm_type == 'lstm_no_monitor':
                        lstm_model = with_dropout_model_1
                    elif lstm_type == 'no_lstm':
                        lstm_model = with_dropout_model_1  # but not using the model
                    else:
                        print('check lstm_type.')
                        lstm_model = None
                    ####
                    runner = CarlaRunner(agent_config, scenario_config,lstm_model)
                else:
                    mean_df = None
                    std_df = None
                    lstm_model = None
                    runner = CarlaRunner(agent_config, scenario_config, lstm_model)


            # start running
            try:
                runner.run(mean_df, std_df, dropout_rate, dropout_type, conf, N_MC, hidden_size, lstm_type,
                            start_episode, selected_feature_col,if_new_controller, step_look_back, predict_length)
            except:
                runner.close()
                traceback.print_exc()
                err_list.append([agent_cfg, scenario_cfg, traceback.format_exc()])

    for err in err_list:
        print(err[0], err[1], 'failed!')
        print(err[2])
