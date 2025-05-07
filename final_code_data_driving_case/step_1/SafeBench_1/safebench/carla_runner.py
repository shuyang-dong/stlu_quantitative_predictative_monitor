''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-03 22:35:17
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import copy

import matplotlib.pyplot as plt
import numpy as np
import carla
import pygame
from tqdm import tqdm

from safebench.gym_carla.env_wrapper import VectorWrapper
from safebench.gym_carla.envs.render import BirdeyeRender
from safebench.gym_carla.replay_buffer import RouteReplayBuffer, PerceptionReplayBuffer

from safebench.agent import AGENT_POLICY_LIST
from safebench.scenario import SCENARIO_POLICY_LIST

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_data_loader import ScenarioDataLoader
from safebench.scenario.tools.scenario_utils import scenario_parse

from safebench.util.logger import Logger, setup_logger_kwargs
from safebench.util.metric_util import get_route_scores, get_perception_scores

import pandas as pd
import warnings
import json
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

import sys
sys.path.append('your/path/to/stlu_monitor')
import ustlmonitor as ustl
import confidencelevel
from collections import namedtuple
from scipy.stats import norm
import math
import os
#from statistics import mean

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('New folder ok.')
    else:
        print('There is this folder')

    return

class CarlaRunner:
    def __init__(self, agent_config, scenario_config,lstm_model=None):
        self.scenario_config = scenario_config
        self.agent_config = agent_config

        self.seed = scenario_config['seed']
        self.exp_name = scenario_config['exp_name']
        self.output_dir = scenario_config['output_dir']
        self.mode = scenario_config['mode']
        self.save_video = scenario_config['save_video']

        self.render = scenario_config['render']
        self.num_scenario = scenario_config['num_scenario']
        self.fixed_delta_seconds = scenario_config['fixed_delta_seconds']
        self.scenario_category = scenario_config['scenario_category']
        self.behavior_type = agent_config['behavior_type']
        self.scenario_type = scenario_config['scenario_type']
        self.scenario_type_dir = scenario_config['scenario_type_dir']

        scenario_json_file = '/home/safebench/SafeBench/{scenario_type_dir}/{scenario_type}'.format(scenario_type_dir=self.scenario_type_dir, scenario_type=self.scenario_type)
        with open(scenario_json_file, 'r') as file:
            scenario_data = json.load(file)
            print(scenario_data)
            self.scenario_id = scenario_data[0]['scenario_id']
            self.route_id = scenario_data[0]['route_id']

        # continue training flag
        self.continue_agent_training = scenario_config['continue_agent_training']
        self.continue_scenario_training = scenario_config['continue_scenario_training']

        # apply settings to carla
        self.client = carla.Client('localhost', scenario_config['port'])
        self.client.set_timeout(10.0)
        self.world = None
        self.env = None

        # params for predictive monitor
        # mean_df = None
        # std_df = None
        # dropout_rate = 0
        # dropout_type = 0
        # conf = 0
        # N_MC = 0
        # hidden_size = 0
        # lstm_type = 'lstm_with_monitor'
        # start_episode = 0
        # select_feature_col = []
        # if_new_controller = 0
        # step_look_back = 10
        # predict_length = 10
        self.lstm_model = lstm_model
        self.trace_df = None

        self.env_params = {
            'auto_ego': scenario_config['auto_ego'],
            'obs_type': agent_config['obs_type'],
            'scenario_category': self.scenario_category,
            'ROOT_DIR': scenario_config['ROOT_DIR'],
            'warm_up_steps': 9,                                        # number of ticks after spawning the vehicles
            'disable_lidar': False,                                     # show bird-eye view lidar or not
            'display_size': 400, #128,                                       # screen size of one bird-eye view window
            'obs_range': 32,                                           # observation range (meter)
            'd_behind': 12,                                            # distance behind the ego vehicle (meter)
            'max_past_step': 1,                                        # the number of past steps to draw
            'discrete': False,                                         # whether to use discrete control space
            'discrete_acc': [-4.0, 0.0, 4.0],                          # discrete value of accelerations
            'discrete_steer': [-0.2, 0.0, 0.2],                        # discrete value of steering angles
            'continuous_accel_range': [-4.0, 4.0],                     # continuous acceleration range
            'continuous_steer_range': [-0.3, 0.3],                     # continuous steering angle range
            'max_episode_step': scenario_config['max_episode_step'],   # maximum timesteps per episode
            'max_waypt': 12,                                           # maximum number of waypoints
            'lidar_bin': 0.125,                                        # bin size of lidar sensor (meter)
            'out_lane_thres': 4,                                       # threshold for out of lane (meter)
            'desired_speed': 8,                                        # desired speed (m/s)
            'image_sz': 1024,                                          # TODO: move to config of od scenario
        }

        # pass config from scenario to agent
        agent_config['mode'] = scenario_config['mode']
        agent_config['ego_action_dim'] = scenario_config['ego_action_dim']
        agent_config['ego_state_dim'] = scenario_config['ego_state_dim']
        agent_config['ego_action_limit'] = scenario_config['ego_action_limit']

        #print('max_episode_step: ', scenario_config['max_episode_step'])

        # define logger
        logger_kwargs = setup_logger_kwargs(
            self.exp_name, 
            self.output_dir, 
            self.seed,
            agent=agent_config['policy_type'],
            scenario=scenario_config['policy_type'],
            scenario_category=self.scenario_category
        )
        self.logger = Logger(**logger_kwargs)
        
        # prepare parameters
        if self.mode == 'train_agent':
            self.buffer_capacity = agent_config['buffer_capacity']
            self.eval_in_train_freq = agent_config['eval_in_train_freq']
            self.save_freq = agent_config['save_freq']
            self.train_episode = agent_config['train_episode']
            self.logger.save_config(agent_config)
            self.logger.create_training_dir()
        elif self.mode == 'train_scenario':
            self.buffer_capacity = scenario_config['buffer_capacity']
            self.eval_in_train_freq = scenario_config['eval_in_train_freq']
            self.save_freq = scenario_config['save_freq']
            self.train_episode = scenario_config['train_episode']
            self.logger.save_config(scenario_config)
            self.logger.create_training_dir()
        elif self.mode == 'eval':
            self.save_freq = scenario_config['save_freq']
            self.eval_episode = agent_config['eval_episode']
            self.logger.log('>> Evaluation Mode, skip config saving', 'yellow')
            self.logger.create_eval_dir(load_existing_results=True)
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}.")

        # define agent and scenario
        self.logger.log('>> Agent Policy: ' + agent_config['policy_type'])
        self.logger.log('>> Scenario Policy: ' + scenario_config['policy_type'])

        if self.scenario_config['auto_ego']:
            self.logger.log('>> Using auto-polit for ego vehicle, action of policy will be ignored', 'yellow')
        if scenario_config['policy_type'] == 'ordinary' and self.mode != 'train_agent':
            self.logger.log('>> Ordinary scenario can only be used in agent training', 'red')
            raise Exception()
        self.logger.log('>> ' + '-' * 40)

        # define agent and scenario policy
        self.agent_policy = AGENT_POLICY_LIST[agent_config['policy_type']](agent_config, logger=self.logger)
        self.scenario_policy = SCENARIO_POLICY_LIST[scenario_config['policy_type']](scenario_config, logger=self.logger)
        if self.save_video:
            assert self.mode == 'eval', "only allow video saving in eval mode"
            self.logger.init_video_recorder()

    def _init_world(self, town):
        self.logger.log(f">> Initializing carla world: {town}")
        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(self.scenario_config['tm_port'])
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def _init_renderer(self):
        self.logger.log(">> Initializing pygame birdeye renderer")
        pygame.init()
        flag = pygame.HWSURFACE | pygame.DOUBLEBUF
        if not self.render:
            flag = flag | pygame.HIDDEN
        if self.scenario_category == 'planning': 
            # [bird-eye view, Lidar, front view] or [bird-eye view, front view]
            if self.env_params['disable_lidar']:
                window_size = (self.env_params['display_size'] * 2, self.env_params['display_size'] * self.num_scenario)
            else:
                window_size = (self.env_params['display_size'] * 3, self.env_params['display_size'] * self.num_scenario)
        else:
            window_size = (self.env_params['display_size'], self.env_params['display_size'] * self.num_scenario)
        self.display = pygame.display.set_mode(window_size, flag)

        # initialize the render for generating observation and visualization
        pixels_per_meter = self.env_params['display_size'] / self.env_params['obs_range']
        pixels_ahead_vehicle = (self.env_params['obs_range'] / 2 - self.env_params['d_behind']) * pixels_per_meter
        self.birdeye_params = {
            'screen_size': [self.env_params['display_size'], self.env_params['display_size']],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle,
        }
        self.birdeye_render = BirdeyeRender(self.world, self.birdeye_params, logger=self.logger)

    def train(self, data_loader, start_episode=0):
        # general buffer for both agent and scenario
        Buffer = RouteReplayBuffer if self.scenario_category == 'planning' else PerceptionReplayBuffer
        replay_buffer = Buffer(self.num_scenario, self.mode, self.buffer_capacity)

        for e_i in tqdm(range(start_episode, self.train_episode)):
            # sample scenarios
            sampled_scenario_configs, _ = data_loader.sampler()
            # reset the index counter to create endless loader
            data_loader.reset_idx_counter()

            # get static obs and then reset with init action 
            static_obs = self.env.get_static_obs(sampled_scenario_configs)
            scenario_init_action, additional_dict = self.scenario_policy.get_init_action(static_obs)
            obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)
            replay_buffer.store_init([static_obs, scenario_init_action], additional_dict=additional_dict)

            # get ego vehicle from scenario
            self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)

            # start loop
            episode_reward = []
            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions = self.agent_policy.get_action(obs, infos, deterministic=False)
                scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=False)

                # apply action to env and get obs
                next_obs, rewards, dones, infos = self.env.step(ego_actions=ego_actions, scenario_actions=scenario_actions)
                replay_buffer.store([ego_actions, scenario_actions, obs, next_obs, rewards, dones], additional_dict=infos)
                obs = copy.deepcopy(next_obs)
                episode_reward.append(np.mean(rewards))

                # train off-policy agent or scenario
                if self.mode == 'train_agent' and self.agent_policy.type == 'offpolicy':
                    self.agent_policy.train(replay_buffer)
                elif self.mode == 'train_scenario' and self.scenario_policy.type == 'offpolicy':
                    self.scenario_policy.train(replay_buffer)

            # end up environment
            self.env.clean_up()
            replay_buffer.finish_one_episode()
            self.logger.add_training_results('episode', e_i)
            self.logger.add_training_results('episode_reward', np.sum(episode_reward))
            self.logger.save_training_results()

            # train on-policy agent or scenario
            if self.mode == 'train_agent' and self.agent_policy.type == 'onpolicy':
                self.agent_policy.train(replay_buffer)
            elif self.mode == 'train_scenario' and self.scenario_policy.type in ['init_state', 'onpolicy']:
                self.scenario_policy.train(replay_buffer)

            # eval during training
            if (e_i+1) % self.eval_in_train_freq == 0:
                #self.eval(env, data_loader)
                pass

            # save checkpoints
            if (e_i+1) % self.save_freq == 0:
                if self.mode == 'train_agent':
                    self.agent_policy.save_model(e_i)
                if self.mode == 'train_scenario':
                    self.scenario_policy.save_model(e_i)

    def preprocess_driving_data(self, driving_trace_df):
        # get acc, and the change of acc
        # df = pd.read_csv(driving_trace_df, index_col=0)
        # change time format, save hour and minute
        acc_list = []
        delta_acc_list = []
        for item, row in driving_trace_df.iterrows():
            acc_x = row['acc_x']
            acc_y = row['acc_y']
            acc_z = row['acc_z']
            acc = math.sqrt(math.pow(acc_x, 2) + math.pow(acc_y, 2) + math.pow(acc_z, 2))
            acc_list.append(acc)
            if item == 0:
                delta_acc_list.append(0)
            else:
                delta_acc_list.append(acc - acc_list[item - 1])

        driving_trace_df['acc'] = acc_list
        driving_trace_df['delta_acc'] = delta_acc_list
        return driving_trace_df
    # get upper and lower bound for predicted flowpipe
    def get_upper_lower_bound_flowpipe(self, mean, sigma, conf):
        # calculate upper and lower bound using given mean, std, and conf
        p = 1 - (1 - conf) / 2
        lower = mean - norm.ppf(p) * sigma  # norm.ppf(p):get the x value for certain confidence level p
        upper = mean + norm.ppf(p) * sigma
        return (lower, upper)
    # STL formula for left lane distance-using one side
    def requirement_func_always_speed_in_range(self, signal, trace_len, conf, lower_speed=2.0, upper_speed=100.0,
                                               func='monitor'):
        # signal: Speed trace, Keep speed in range
        # G[0,t](speed > lower_speed & speed < upper_speed)
        # STL: G[0,10](signal>lower_speed) and (signal<upper_speed)
        # convert to:
        # G[0,10] (signal>lower_speed) and neg(signal>upper_speed)
        threshold_1 = lower_speed
        threshold_2 = upper_speed
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

    def requirement_func_always_acc_in_range(self, signal, trace_len, conf, lower_acc=-3.4, upper_acc=2.0,
                                               func='monitor'):
        # signal: Speed trace, Keep speed in range
        # G[0,t](speed > lower_speed & speed < upper_speed)
        # STL: G[0,10](signal>lower_speed) and (signal<upper_speed)
        # convert to:
        # G[0,10] (signal>lower_speed) and neg(signal>upper_speed)
        # https://safety.fhwa.dot.gov/speedmgt/ref_mats/fhwasa12022/chap_2.cfm
        # https://www.sciencedirect.com/science/article/pii/S0003687022002046

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

    def predict_with_lstm_with_dropout(self, dataset_loader, lstm_model, predict_length, hidden_size, N_MC,
                step_look_back, dropout_rate, dropout_type, requirement_func_acc, mean_df, std_df, conf):
      # evaluate the trained model with dataset
      lstm_model.eval()
      batch_len=1
      with torch.no_grad():
        for batch_idx, datasample in enumerate(dataset_loader):  # tqdm(enumerate(test_loader)):
            data,target0 = datasample[0], datasample[1]
            target = []
            output = torch.zeros(N_MC, batch_len, predict_length, hidden_size)
            # apply dropout and loop N_MC times to get a series of predicted traces
            new_mean_df = mean_df.reset_index(drop=True)
            new_std_df = std_df.reset_index(drop=True)
            for i in range(N_MC):
                output[i] = (lstm_model.forward_test_with_past(data, step_look_back, hidden_size, dropout_rate, dropout_type)).cpu()
                # change predictions back to original scale
                for feature_index in range(output[i].shape[2]):
                  output[i][:, :, feature_index] = (output[i][:, :, feature_index]*new_std_df.loc[feature_index]+new_mean_df.loc[feature_index]).cpu()
            output_acc_mean = output.mean(dim=0)[:, :, -2]
            output_acc_std = output.std(dim=0)[:, :, -2]
            trace_acc = torch.stack((output_acc_mean, output_acc_std), dim=-1)
            # record mean and std value for each sample after applying dropout and loop N_MC times
            for b in range(batch_len):
                # get rho interval for predictions with STL-U
                rho_set_acc = requirement_func_acc(trace_acc[b, :, :], predict_length-1, conf=conf, lower_acc=-3.4, upper_acc=2.0,
                                               func='monitor')
        return output_acc_mean.squeeze(0), output_acc_std.squeeze(0), rho_set_acc

    def new_control_with_predictive_monitor_0(self, actions, rho_set_acc_this_step, predict_acc_low_list, predict_acc_up_list,
                                            distance_to_lead,
                                            current_speed,
                                            past_step_to_check=5):
        # improve actions with the monitor result
        # action: throttle, steer, brake, hand_brake, reverse, manual_gear_shift, gear
        # adjust brake and throttle values according to rho_set_acc
        # if the prediction is weak satisfy/or if the lower bound is lower than certain value, start to decelerate, and avoid sudden accelerate
        # how to smooth the acceleration/deceleration, and keep a steady speed at the same time
        # check on the rho_set intervals continuously, refer to previous values
        max_distance_to_lead = 20
        max_throttle = 0.6
        min_throttle = 0.4
        max_brake = 0.6
        min_brake = 0.4
        min_speed = 5.0
        rho_low_no_change = 3.0
        #throttle_adjust_param = 0.7
        #brake_adjust_param = 1.3
        throttle_this_step = actions[0][0]
        brake_this_step = actions[0][2]
        throttle_past_list = self.trace_df['throttle'].tolist()[-past_step_to_check:]
        brake_past_list = self.trace_df['brake'].tolist()[-past_step_to_check-1:]
        acc_past_list = self.trace_df['acc'].tolist()[-past_step_to_check-1:]
        throttle_past_list.append(throttle_this_step)
        brake_past_list.append(brake_this_step)
        #print('throttle_this_step: ', throttle_this_step, ' brake_this_step: ', brake_this_step)
        #print('throttle_past_list: ', throttle_past_list, ' brake_past_list: ', brake_past_list)
        rho_low = rho_set_acc_this_step[0]
        rho_high = rho_set_acc_this_step[1]
        if distance_to_lead>max_distance_to_lead:
            #print('Longer distance, slight adjustment.')
            mean_throttle = sum(throttle_past_list) / len(throttle_past_list)
            # if mean_throttle >= min([min_throttle, throttle_this_step]):
            #     new_throttle = min_throttle
            #     new_brake = min([min_brake, brake_this_step])
            # else:
            #     new_throttle = min([max_throttle, throttle_this_step])
            #     new_brake = min([max_brake, brake_this_step])
            ########
            # if rho_low >= rho_low_no_change:
            #     new_throttle = min([max_throttle, throttle_this_step])
            #     new_brake = min([max_brake, brake_this_step])
            # else:
            #     if mean_throttle >= min([min_throttle, throttle_this_step]):
            #         new_throttle = min_throttle
            #         new_brake = min([min_brake, brake_this_step])
            #     else:
            #         new_throttle = min([max_throttle, throttle_this_step])
            #         new_brake = min([max_brake, brake_this_step])
            if current_speed>=min_speed:
                if rho_low >= rho_low_no_change:
                    # no change in actions if strong satisfy
                    print('LD, Strong satisfy, no adjustment.')
                    new_throttle = min([max_throttle, throttle_this_step])
                    new_brake = min([max_brake, brake_this_step])
                elif 0 <= rho_low < rho_low_no_change:
                    print('LD, Need adjustment.')
                    # reduce throttle to prevent getting to close to lead veh
                    throttle_adjust_param = 0.7
                    mean_throttle = sum(throttle_past_list) / len(throttle_past_list)
                    # print('mean_throttle: ', mean_throttle)
                    new_throttle = min([mean_throttle * throttle_adjust_param, max_throttle])
                    new_brake = 0
                else:
                    # adjust throttle and brake if there will be violation or weak satisfy
                    if min(predict_acc_low_list) <= -3.4 and max(predict_acc_up_list) < 2.0:
                        # predict a hard brake, smooth the brake
                        brake_adjust_param = 1 + 1/abs(rho_low) # the more violation, the smaller this param
                        mean_brake = sum(brake_past_list)/len(brake_past_list)
                        #print('mean_brake: ', mean_brake)
                        new_brake = min([mean_brake * brake_adjust_param, max_brake])
                        print('LD, Predict a hard brake, mean and new brake are: ', mean_brake, new_brake)
                        new_throttle = 0
                    elif max(predict_acc_up_list) >= 2.0 and min(predict_acc_low_list) > -3.4:
                        # Predict a sharp acc, smooth the throttle
                        throttle_adjust_param = 1 + 1 / abs(rho_low) # the more violation, the smaller this param
                        mean_throttle = sum(throttle_past_list) / len(throttle_past_list)
                        #print('mean_throttle: ', mean_throttle)
                        new_throttle = min([mean_throttle * throttle_adjust_param,  max_throttle])
                        new_brake = 0
                        print('LD, Predict a sharp acc, mean and new throttle are: ', mean_throttle, new_throttle)
                    else:
                        # both sides violate the requirement, prevent a sharp acceleration by conducting smooth brake, which can prevent a hard brake later
                        brake_adjust_param = max(1 - 1 / abs(rho_low), 0)
                        mean_brake = sum(brake_past_list) / len(brake_past_list)
                        new_brake = min([mean_brake * brake_adjust_param, max_brake])
                        new_throttle = 0
                        print('LD,Both side violation. Mean and new brake are: ', mean_brake, new_brake)
                        #print('Check on acc_past_list values: ', predict_acc_low_list, predict_acc_up_list)
            else:
                # provide throttle to avoid speed being too low
                throttle_adjust_param = (1 - 1 / abs(rho_low)) #?
                mean_throttle = sum(throttle_past_list) / len(throttle_past_list)
                # print('mean_throttle: ', mean_throttle)
                new_throttle = min([mean_throttle * throttle_adjust_param, max_throttle])
                new_brake = 0
                print('LD, Avoid speed reduce to 0, adjust: ', mean_throttle, new_throttle)
        else:
            #print('Near lead veh.')
            if current_speed>=min_speed:
                if rho_low >= rho_low_no_change:
                    # no change in actions if strong satisfy
                    print('NV, Strong satisfy, no adjustment.')
                    new_throttle = min([max_throttle, throttle_this_step])
                    new_brake = min([max_brake, brake_this_step])
                elif 0 <= rho_low < rho_low_no_change:
                    print('NV, Need adjustment.')
                    # reduce throttle to prevent getting to close to lead veh
                    throttle_adjust_param = 0.7
                    mean_throttle = sum(throttle_past_list) / len(throttle_past_list)
                    # print('mean_throttle: ', mean_throttle)
                    new_throttle = min([mean_throttle * throttle_adjust_param, max_throttle])
                    new_brake = 0
                else:
                    # adjust throttle and brake if there is violation or weak satisfy
                    if min(predict_acc_low_list) <= -3.4 and max(predict_acc_up_list) < 2.0:
                        # predict a hard brake, smooth the brake
                        brake_adjust_param = 1 + 1 / abs(rho_low)  # the more violation, the smaller this param
                        mean_brake = sum(brake_past_list)/len(brake_past_list)
                        #print('mean_brake: ', mean_brake)
                        new_brake = min([mean_brake * brake_adjust_param, max_brake])
                        #print('NV, Predict a hard brake, mean and new brake are: ', mean_brake, new_brake)
                        new_throttle = 0
                    elif max(predict_acc_up_list) >= 2.0 and min(predict_acc_low_list) > -3.4:
                        # if acc prediction is lower than lower bound, <-3.4, smooth the brake
                        throttle_adjust_param = 1 + 1 / abs(rho_low)  # the more violation, the smaller this param
                        mean_throttle = sum(throttle_past_list) / len(throttle_past_list)
                        #print('mean_throttle: ', mean_throttle)
                        new_throttle = min([mean_throttle * throttle_adjust_param,  max_throttle])
                        new_brake = 0
                        #print('NV, Predict a sharp acc, mean and new throttle are: ', mean_throttle, new_throttle)
                    else:
                        # both sides violate the requirement, either a hard brake or a sharp acc, both need to conduct brake, to prevent a sharp acceleration first
                        brake_adjust_param = max(1 - 1 / abs(rho_low), 0)
                        mean_brake = sum(brake_past_list) / len(brake_past_list)
                        new_brake = min([mean_brake * brake_adjust_param, max_brake])
                        new_throttle = 0
                        print('NV, Both side violation. Mean and new brake are: ', mean_brake, new_brake)
                        #print('Check on acc_past_list values: ', predict_acc_low_list, predict_acc_up_list)
            else:
                # give a moderate throttle to avoid speed being too low, but not too high for avoiding collision
                throttle_adjust_param = 1 - 1 / abs(rho_low) #?
                mean_throttle = sum(throttle_past_list) / len(throttle_past_list)
                # print('mean_throttle: ', mean_throttle)
                new_throttle = min([mean_throttle * throttle_adjust_param, max_throttle])
                new_brake = 0
                print('NV,Avoid speed reduce to 0, adjust: ', mean_throttle, new_throttle)
        actions[0][0] = max([new_throttle, 0])
        actions[0][2] = max([new_brake, 0])
        actions[0][1] = 0 # set steer to 0, avoid collision with wall
        return actions

    def new_control_with_predictive_monitor(self, actions, rho_set_acc_this_step, predict_acc_low_list, predict_acc_up_list,
                                            distance_to_lead,
                                            current_speed,
                                            past_step_to_check=5):
        # improve actions with the monitor result
        # action: throttle, steer, brake, hand_brake, reverse, manual_gear_shift, gear
        # adjust brake and throttle values according to rho_set_acc
        # if the prediction is weak satisfy/or if the lower bound is lower than certain value, start to decelerate, and avoid sudden accelerate
        # how to smooth the acceleration/deceleration, and keep a steady speed at the same time
        # check on the rho_set intervals continuously, refer to previous values
        max_distance_to_lead = 20
        max_throttle = 0.6
        min_throttle = 0.4
        max_brake = 0.6
        min_brake = 0.4
        min_speed = 5.0
        rho_low_no_change = 0
        throttle_this_step = actions[0][0]
        brake_this_step = actions[0][2]
        throttle_past_list = self.trace_df['throttle'].tolist()[-past_step_to_check:]
        brake_past_list = self.trace_df['brake'].tolist()[-past_step_to_check-1:]
        acc_past_list = self.trace_df['acc'].tolist()[-past_step_to_check-1:]
        throttle_past_list.append(throttle_this_step)
        brake_past_list.append(brake_this_step)
        #print('throttle_this_step: ', throttle_this_step, ' brake_this_step: ', brake_this_step)
        #print('throttle_past_list: ', throttle_past_list, ' brake_past_list: ', brake_past_list)
        rho_low = rho_set_acc_this_step[0]
        rho_high = rho_set_acc_this_step[1]
        mean_throttle = sum(throttle_past_list) / len(throttle_past_list)
        if current_speed >= min_speed:
            if rho_low >= rho_low_no_change:
                # no change in actions if strong satisfy
                print('Strong satisfy, no adjustment.')
                new_throttle = min([max_throttle, throttle_this_step])
                new_brake = min([max_brake, brake_this_step])
            # elif 0 <= rho_low < rho_low_no_change:
            #     print('Need adjustment.')
            #     # reduce throttle to prevent getting to close to lead veh
            #     throttle_adjust_param = 0.7
            #     mean_throttle = sum(throttle_past_list) / len(throttle_past_list)
            #     # print('mean_throttle: ', mean_throttle)
            #     new_throttle = min([mean_throttle * throttle_adjust_param, max_throttle])
            #     new_brake = 0
            else:
                # adjust throttle and brake if there will be violation or weak satisfy
                if min(predict_acc_low_list) <= -3.4 and max(predict_acc_up_list) < 2.0:
                    # predict a hard brake, smooth the brake
                    brake_adjust_param = 1 + 1 / abs(rho_low)  # the more violation, the smaller this param
                    mean_brake = sum(brake_past_list) / len(brake_past_list)
                    # print('mean_brake: ', mean_brake)
                    new_brake = min([mean_brake * brake_adjust_param, max_brake])
                    print('Predict a hard brake, mean and new brake are: ', mean_brake, new_brake)
                    new_throttle = 0
                elif max(predict_acc_up_list) >= 2.0 and min(predict_acc_low_list) > -3.4:
                    # Predict a sharp acc, smooth the throttle
                    throttle_adjust_param = 1 + 1 / abs(rho_low)  # the more violation, the smaller this param
                    mean_throttle = sum(throttle_past_list) / len(throttle_past_list)
                    # print('mean_throttle: ', mean_throttle)
                    new_throttle = min([mean_throttle * throttle_adjust_param, max_throttle])
                    new_brake = 0
                    print('Predict a sharp acc, mean and new throttle are: ', mean_throttle, new_throttle)
                else:
                    # both sides violate the requirement, prevent a sharp acceleration by conducting smooth brake, which can prevent a hard brake later
                    brake_adjust_param = max(1 - 1 / abs(rho_low), 0)
                    mean_brake = sum(brake_past_list) / len(brake_past_list)
                    new_brake = min([mean_brake * brake_adjust_param, max_brake])
                    new_throttle = 0
                    #print('Both side violation. Mean and new brake are: ', mean_brake, new_brake, brake_adjust_param)
                    # print('Check on acc_past_list values: ', predict_acc_low_list, predict_acc_up_list)
        else:
            # provide moderate throttle to avoid speed being too low
            # if abs(rho_low)<1:
            #     throttle_adjust_param = (1 - abs(rho_low))
            # else:
            #     throttle_adjust_param = (1 - 1 / abs(rho_low))
            throttle_adjust_param = max(1 - 1 / abs(rho_low), 0)
            mean_throttle = sum(throttle_past_list) / len(throttle_past_list)
            # print('mean_throttle: ', mean_throttle)
            new_throttle = min([mean_throttle * throttle_adjust_param, max_throttle])
            new_brake = 0
            #print('Avoid speed reduce to 0, adjust throttle: ', mean_throttle, new_throttle, throttle_adjust_param)
        actions[0][0] = max([new_throttle, 0])
        actions[0][2] = max([new_brake, 0])
        actions[0][1] = 0 # set steer to 0, avoid collision with wall
        return actions

    def eval(self, data_loader,
             mean_df, std_df, dropout_rate, dropout_type, conf, N_MC, hidden_size, lstm_type,
             start_episode=0, select_feature_col=[],if_new_controller=0, step_look_back=30, predict_length=20):
        # new dataframe for saving trace
        feature_columns = ['episode','step','seed','scenario_id','route_id','lateral_dis','acc_x','acc_y','acc_z','acc','delta_acc',
                   'speed','v_x', 'v_y',
                   'brake', 'throttle','steer', 'gear','hand_brake','manual_gear_shift',
                   'radar_obj_depth_aveg', 'radar_obj_depth_std',
                   'radar_obj_velocity_aveg', 'radar_obj_velocity_std',
                    'reverse','cost','done','ego_x', 'ego_y','ego_yaw','-delta_yaw','if_vehicle_front',
                   'obs_state_prev','scenario_actions','vehicle_front','waypoints','rewards']
        self.trace_df_list = []
        #self.trace_df = pd.DataFrame(columns=feature_columns)
        ###
        for e_i in tqdm(range(start_episode, self.eval_episode)):
            # sample scenarios
            sampled_scenario_configs, _ = data_loader.sampler()
            # reset the index counter to create endless loader
            data_loader.reset_idx_counter()
            num_finished_scenario = 0
            #data_loader.reset_idx_counter()

            while len(data_loader) > 0:
                self.trace_df = pd.DataFrame(columns=feature_columns)
                # sample scenarios
                sampled_scenario_configs, num_sampled_scenario = data_loader.sampler()
                num_finished_scenario += num_sampled_scenario

                # reset envs with new config, get init action from scenario policy, and run scenario
                static_obs = self.env.get_static_obs(sampled_scenario_configs)
                self.scenario_policy.load_model(sampled_scenario_configs)
                scenario_init_action, _ = self.scenario_policy.get_init_action(static_obs, deterministic=False)
                obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)

                # get ego vehicle from scenario
                self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)
                step_count = 0
                step_dict = {'episode':e_i}

                score_list = {s_i: [] for s_i in range(num_sampled_scenario)}
                while not self.env.all_scenario_done():
                    if if_new_controller==1: # use the new controller with predictive monitor
                        if len(self.trace_df)<step_look_back:
                            #trace_df = self.preprocess_driving_data(trace_df)
                            print('Run step without predict.')
                            # get action from agent policy and scenario policy (assume using one batch)
                            ego_actions = self.agent_policy.get_action(obs, infos, deterministic=False)
                            scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=False)
                            # apply action to env and get obs
                            obs, rewards, _, infos = self.env.step(ego_actions=ego_actions,
                                                                   scenario_actions=scenario_actions)
                        else:
                            #print('Run step with predict')
                            #trace_df = self.preprocess_driving_data(trace_df)
                            # pass history trace to lstm, predict future steps
                            history_trace = self.trace_df[select_feature_col]
                            # get last 30 step trace, used as historical data for prediction
                            past_df_x = history_trace[-step_look_back:]
                            past_df_y = history_trace[-step_look_back - 1:-1]
                            # standardlize
                            new_mean_df = mean_df
                            new_std_df = std_df
                            for feature in select_feature_col:
                                a_x = past_df_x[feature] - new_mean_df.loc[feature]
                                b_x = a_x / new_std_df.loc[feature]
                                a_y = past_df_x[feature] - new_mean_df.loc[feature]
                                b_y = a_x / new_std_df.loc[feature]
                                past_df_x[feature] = b_x
                                past_df_y[feature] = b_y
                            x = np.array(past_df_x)
                            y = np.array(past_df_y)
                            x = np.concatenate((x, np.zeros((predict_length, len(select_feature_col)))), axis=0)
                            y = np.concatenate((y, np.zeros((predict_length, len(select_feature_col)))), axis=0)
                            x = x.astype(float)
                            y = y.astype(float)
                            dataX = Variable(torch.from_numpy(x))
                            dataY = Variable(torch.from_numpy(y))
                            dataX = dataX.unsqueeze(0)
                            dataY = dataY.unsqueeze(0)
                            testX = Variable(torch.Tensor(np.array(dataX)))
                            testY = Variable(torch.Tensor(np.array(dataY)))
                            test_set = torch.utils.data.TensorDataset(testX, testY)
                            test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=0,
                                                                      shuffle=False)
                            # get mean value trace, std trace and rho interval for predicted flowpipe
                            if lstm_type == 'lstm_with_monitor':
                                #print('lstm_with_monitor')
                                #self.controller.trace_type = 'future'
                                # connect lstm with STL-U, using monitor results for self-defined controller to decide action
                                output_acc_mean, output_acc_std, rho_set_acc = self.predict_with_lstm_with_dropout(
                                    test_loader, self.lstm_model, predict_length, hidden_size, N_MC,
                                    step_look_back, dropout_rate, dropout_type,
                                    self.requirement_func_always_acc_in_range, new_mean_df, new_std_df, conf)
                                #print('output_acc_mean, output_acc_std, rho_set_acc: ',output_acc_mean, output_acc_std, rho_set_acc)
                                # print('lstm_with_monitor, rho_set_acc: ', rho_set_acc)
                                output_acc_low, output_acc_up = self.get_upper_lower_bound_flowpipe(output_acc_mean,
                                                                                                    output_acc_std,
                                                                                                    conf)
                                #print('output_acc_low, output_acc_up: ', output_acc_low, output_acc_up)
                                # Todo:
                                # pass rho interval to new controller to decide new control actions
                                # get action from modified agent policy and scenario policy (assume using one batch)
                                ego_actions = self.agent_policy.get_action(obs, infos, deterministic=False)
                                scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=False)
                                #print('ego action in carla runner: ', ego_actions)
                                distance_to_lead = obs_state_list[12]
                                current_speed = obs_state_list[2]
                                new_ego_actions = self.new_control_with_predictive_monitor(ego_actions, rho_set_acc,
                                                                                           output_acc_low,output_acc_up,
                                                                                           distance_to_lead,
                                                                                           current_speed)
                                #print('Action after adjustment: ', new_ego_actions)
                                # apply action to env and get obs
                                obs, rewards, _, infos = self.env.step(ego_actions=new_ego_actions,
                                                                       scenario_actions=scenario_actions)

                            elif lstm_type == 'lstm_no_monitor':
                                print('lstm_no_monitor')
                                #self.controller.trace_type = 'history'
                                # not using monitor, pass predicted CGM to BB control policy to decide action
                                '''output_CGM_mean, output_CGM_std, rho_set_CGM = self.predict_with_lstm_no_dropout(test_loader, self.lstm_model, device, 
                                                batch_size, predict_length, hidden_size, N_MC, 
                                                step_look_back, self.requirement_func)'''
                                output_acc_mean, output_acc_std, rho_set_acc = self.predict_with_lstm_with_dropout(
                                    test_loader, self.lstm_model, predict_length, hidden_size, N_MC,
                                    step_look_back, dropout_rate, dropout_type,
                                    self.requirement_func_always_acc_in_range, new_mean_df, new_std_df, conf)
                                print('output_acc_mean, output_acc_std, rho_set_acc: ', output_acc_mean,
                                      output_acc_std, rho_set_acc)
                                output_speed_low, output_speed_up = self.get_upper_lower_bound_flowpipe(output_acc_mean,
                                                                                                    output_acc_std,
                                                                                                    conf)
                                print('output_acc_low, output_acc_up: ', output_acc_low, output_acc_up)
                                # Todo: not implemented, not used in the final evaluation
                                # get action from modified agent policy and scenario policy (assume using one batch)
                                # pass predicted trace to stl-u and get rho interval
                                # pass rho interval to new controller to decide new control actions
                                ego_actions = self.agent_policy.get_action(obs, infos, deterministic=False)
                                scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=False)
                                # apply action to env and get obs
                                obs, rewards, _, infos = self.env.step(ego_actions=ego_actions,
                                                                       scenario_actions=scenario_actions)

                    else: # conduct actions with origin agent policy
                        # get action from agent policy and scenario policy (assume using one batch)
                        ego_actions = self.agent_policy.get_action(obs, infos, deterministic=False)
                        scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=False)
                        # apply action to env and get obs
                        obs, rewards, _, infos = self.env.step(ego_actions=ego_actions,
                                                               scenario_actions=scenario_actions)

                    # ####### original code
                    # # get action from agent policy and scenario policy (assume using one batch)
                    # ego_actions = self.agent_policy.get_action(obs, infos, deterministic=False)
                    # scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=False)
                    # #print('ego_actions: ', ego_actions)
                    #
                    # # store previous obs
                    # #print('obs: ', obs)
                    # #step_dict['obs_state_prev'] = obs[0]['states'].tolist()
                    # #step_dict['obs_img_prev'] = obs[0]['img'].tolist()
                    # #step_dict['obs_lidar_prev'] = obs[0]['lidar'].tolist()
                    # # print('obs states: ', obs[0]['states'].tolist(), type(obs[0]['states'].tolist()))
                    # # print('obs_img_prev: ', obs[0]['img'].tolist(), type(obs[0]['img'].tolist()))
                    # # print('obs_lidar_prev: ', obs[0]['lidar'].tolist(), type(obs[0]['lidar'].tolist()))
                    #
                    # # apply action to env and get obs
                    # obs, rewards, _, infos = self.env.step(ego_actions=ego_actions, scenario_actions=scenario_actions)
                    # #######
                    # store trace info
                    # print('obs in carla runner: ', obs)
                    obs_state_list = obs[0]['states'].tolist() # how many state items is decided in env_wrapper.py
                    ego_actions_list = ego_actions[0].tolist()
                    # print(obs_state_list, len(obs_state_list))
                    # print(ego_actions_list, len(ego_actions_list))
                    # state = np.array([lateral_dis, -delta_yaw, speed, self.vehicle_front,
                    #                   acc.x, acc.y, acc.z,
                    #                   ego_x, ego_y, ego_yaw,
                    #                   v.x, v.y,
                    #                   radar_obj_depth_aveg, radar_obj_depth_std, radar_obj_velocity_aveg,
                    #                   radar_obj_velocity_std])
                    step_dict['lateral_dis'] = obs_state_list[0]
                    step_dict['-delta_yaw'] = obs_state_list[1]
                    step_dict['speed'] = obs_state_list[2]
                    step_dict['vehicle_front'] = obs_state_list[3]
                    step_dict['acc_x'] = obs_state_list[4]
                    step_dict['acc_y'] = obs_state_list[5]
                    step_dict['acc_z'] = obs_state_list[6]
                    step_dict['acc'] = math.sqrt(math.pow(step_dict['acc_x'], 2) + math.pow(step_dict['acc_y'], 2) + math.pow(step_dict['acc_z'], 2))
                    step_dict['ego_x'] = obs_state_list[7]
                    step_dict['ego_y'] = obs_state_list[8]
                    step_dict['ego_yaw'] = obs_state_list[9]
                    step_dict['v_x'] = obs_state_list[10]
                    step_dict['v_y'] = obs_state_list[11]
                    step_dict['radar_obj_depth_aveg'] = obs_state_list[12]
                    step_dict['radar_obj_depth_std'] = obs_state_list[13]
                    step_dict['radar_obj_velocity_aveg'] = obs_state_list[14]
                    step_dict['radar_obj_velocity_std'] = obs_state_list[15]
                    #step_dict['img'] = obs[0]['img'].tolist()
                    #step_dict['lidar'] = obs[0]['lidar'].tolist()
                    # VehicleControl(throttle=0.750000, steer=0.000001, brake=0.000000, hand_brake=False, reverse=False,
                    #                manual_gear_shift=False, gear=0)
                    step_dict['throttle'] = ego_actions_list[0]
                    step_dict['steer'] = ego_actions_list[1]
                    step_dict['brake'] = ego_actions_list[2]
                    step_dict['hand_brake'] = ego_actions_list[3]
                    step_dict['reverse'] = ego_actions_list[4]
                    step_dict['manual_gear_shift'] = ego_actions_list[5]
                    step_dict['gear'] = ego_actions_list[6]
                    step_dict['scenario_actions'] = scenario_actions
                    step_dict['rewards'] = rewards[0]
                    step_dict['done'] = _[0]
                    step_dict['waypoints'] = infos[0]['waypoints']
                    step_dict['step'] = step_count
                    step_dict['if_vehicle_front'] = infos[0]['vehicle_front']
                    step_dict['scenario_id'] = self.scenario_id
                    step_dict['route_id'] = self.route_id
                    step_dict['seed'] = self.seed
                    step_dict['cost'] = infos[0]['cost'] # mark for collision
                    # step_dict['radar'] = infos[0]['radar']
                    if len(self.trace_df) == 0:
                        step_dict['delta_acc'] = 0
                    else:
                        step_dict['delta_acc'] = step_dict['acc'] - self.trace_df['acc'].tolist()[-1]

                    self.trace_df = self.trace_df.append(step_dict, ignore_index=True)

                    # save video
                    if self.save_video:
                        if self.scenario_category == 'planning':
                            self.logger.add_frame(pygame.surfarray.array3d(self.display).transpose(1, 0, 2))
                        else:
                            self.logger.add_frame({s_i['scenario_id']: ego_actions[n_i]['annotated_image'] for n_i, s_i in enumerate(infos)})

                    # accumulate scores of corresponding scenario
                    reward_idx = 0
                    for s_i in infos:
                        score = rewards[reward_idx] if self.scenario_category == 'planning' else 1-infos[reward_idx]['iou_loss']
                        score_list[s_i['scenario_id']].append(score)
                        reward_idx += 1

                    step_count += 1

                # save trace df for this simulation
                self.trace_df_list.append(self.trace_df)

                # clean up all things
                self.logger.log(">> All scenarios are completed. Clearning up all actors")
                self.env.clean_up()

                # save video
                if self.save_video:
                    data_ids = [config.data_id for config in sampled_scenario_configs]
                    self.logger.save_video(data_ids=data_ids)

                # print score for ranking
                self.logger.log(f'[{num_finished_scenario}/{data_loader.num_total_scenario}] Ranking scores for batch scenario:', 'yellow')
                for s_i in score_list.keys():
                    self.logger.log('\t Env id ' + str(s_i) + ': ' + str(np.mean(score_list[s_i])), 'yellow')

        # calculate evaluation results
        score_function = get_route_scores if self.scenario_category == 'planning' else get_perception_scores
        all_running_results = self.logger.add_eval_results(records=self.env.running_results)
        all_scores = score_function(all_running_results)
        self.logger.add_eval_results(scores=all_scores)
        self.logger.print_eval_results()
        if len(self.env.running_results) % self.save_freq == 0:
            self.logger.save_eval_results()
        self.logger.save_eval_results()

        # save trace to file
        if lstm_type=='no_lstm':
            trace_file_folder = '/home/safebench/SafeBench/safebench/predictive_monitor_trace/trace_file_orig'
        else:
            trace_file_folder = '/home/safebench/SafeBench/safebench/predictive_monitor_trace/trace_file_close_loop_simulation/behavior_type_{behavior_type}/{lstm_type}'.format(behavior_type=self.behavior_type, lstm_type=lstm_type)
        mkdir(trace_file_folder)
        trace_file_path = '{folder}/eval_agent_{agent_mode}_type_{behavior_type}_scenario_{scenario_id}_route_{route_id}_episode_{e_i}_seed_{seed}.csv'.format(
            folder=trace_file_folder, agent_mode=self.agent_config['policy_type'],behavior_type=self.behavior_type,
            scenario_id=self.scenario_id, route_id=self.route_id, e_i=self.eval_episode, seed=self.seed)
        self.trace_df = pd.concat(self.trace_df_list)
        self.trace_df.to_csv(trace_file_path)


    def run(self, mean_df, std_df, dropout_rate, dropout_type, conf, N_MC, hidden_size, lstm_type,
             start_episode, select_feature_col,if_new_controller, step_look_back, predict_length):
        # get scenario data of different maps
        config_by_map = scenario_parse(self.scenario_config, self.logger)
        for m_i in config_by_map.keys():
            # initialize map and render
            self._init_world(m_i)
            self._init_renderer()

            # create scenarios within the vectorized wrapper
            self.env = VectorWrapper(
                self.env_params, 
                self.scenario_config, 
                self.world, 
                self.birdeye_render, 
                self.display, 
                self.logger
            )

            # prepare data loader and buffer
            data_loader = ScenarioDataLoader(config_by_map[m_i], self.num_scenario, m_i, self.world)

            # run with different modes
            if self.mode == 'eval':
                self.agent_policy.load_model()
                # self.scenario_policy.load_model()
                self.agent_policy.set_mode('eval')
                self.scenario_policy.set_mode('eval')
                self.eval(data_loader, mean_df, std_df, dropout_rate, dropout_type, conf, N_MC, hidden_size, lstm_type,
                            start_episode, select_feature_col,if_new_controller, step_look_back, predict_length)
            elif self.mode == 'train_agent':
                start_episode = self.check_continue_training(self.agent_policy)
                self.scenario_policy.load_model()
                self.agent_policy.set_mode('train')
                self.scenario_policy.set_mode('eval')
                self.train(data_loader, start_episode)
            elif self.mode == 'train_scenario':
                start_episode = self.check_continue_training(self.scenario_policy)
                self.agent_policy.load_model()
                self.agent_policy.set_mode('eval')
                self.scenario_policy.set_mode('train')
                self.train(data_loader, start_episode)
            else:
                raise NotImplementedError(f"Unsupported mode: {self.mode}.")

    def check_continue_training(self, policy):
        # load previous checkpoint
        policy.load_model()
        if policy.continue_episode == 0:
            start_episode = 0
            self.logger.log('>> Previous checkpoint not found. Training from scratch.')
        else:
            start_episode = policy.continue_episode
            self.logger.log('>> Continue training from previous checkpoint.')
        return start_episode

    def close(self):
        pygame.quit() 
        if self.env:
            self.env.clean_up()
