''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-03 19:01:03
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import numpy as np

from safebench.agent.base_policy import BasePolicy
from carla.agents.navigation.behavior_agent import BehaviorAgent 


class CarlaBehaviorAgent(BasePolicy):
    name = 'behavior'
    type = 'unlearnable'

    """ This is just an example for testing, whcih always goes straight. """
    def __init__(self, config, logger):
        self.logger = logger
        self.num_scenario = config['num_scenario']
        self.ego_action_dim = config['ego_action_dim']
        self.model_path = config['model_path']
        self.mode = 'train'
        self.continue_episode = 0
        self.route = None
        self.controller_list = []
        behavior_list = ["cautious", "normal", "aggressive"]
        self.behavior = behavior_list[config['behavior_type']]

    def set_ego_and_route(self, ego_vehicles, info):
        self.ego_vehicles = ego_vehicles
        self.controller_list = []
        for e_i in range(len(ego_vehicles)):
            controller = BehaviorAgent(self.ego_vehicles[e_i], behavior=self.behavior)
            dest_waypoint = info[e_i]['route_waypoints'][-1]
            location = dest_waypoint.transform.location
            controller.set_destination(location) # set route for each controller
            self.controller_list.append(controller)

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, obs, infos, deterministic=False):
        actions = []
        for e_i in infos:
            #print('In behavior agent, e_i: ', e_i, ' infos: ', infos)
            # select the controller that matches the scenario_id
            control = self.controller_list[e_i['scenario_id']].run_step()
            #print('car controller: ', self.controller_list[e_i['scenario_id']])
            # VehicleControl(throttle=0.750000, steer=0.000001, brake=0.000000, hand_brake=False, reverse=False,
            #                manual_gear_shift=False, gear=0)
            throttle = control.throttle
            steer = control.steer
            brake = control.brake
            hand_brake = control.hand_brake
            reverse = control.reverse
            manual_gear_shift = control.manual_gear_shift
            gear = control.gear
            actions.append([throttle, steer, brake, hand_brake, reverse, manual_gear_shift, gear])
        actions = np.array(actions, dtype=np.float32)
        #print('Action in behavior agent: ', actions)
        return actions

    def load_model(self):
        pass

    def save_model(self):
        pass
