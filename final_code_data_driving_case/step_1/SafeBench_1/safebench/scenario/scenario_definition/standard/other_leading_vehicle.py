''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 16:48:53
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import carla

from safebench.scenario.tools.scenario_operation import ScenarioOperation
from safebench.scenario.tools.scenario_utils import calculate_distance_transforms
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.tools.scenario_helper import get_waypoint_in_distance
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from carla.agents.navigation.behavior_agent import BehaviorAgent, BasicAgent


class OtherLeadingVehicle(BasicScenario):
    """
        Ego vehicle follows a leading car driving down a given road. At some point the leading car has to decelerate.
        The ego vehicle has to react accordingly by changing lane to avoid a collision and follow the leading car in other lane. 
        The scenario ends either via a timeout, or if the ego vehicle drives some distance. (Traffic Scenario 05)
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        super(OtherLeadingVehicle, self).__init__("OtherLeadingVehicle", config, world)
        self.ego_vehicle = ego_vehicle
        self.timeout = timeout

        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)

        self._first_vehicle_location = 35
        self._second_vehicle_location = self._first_vehicle_location + 1

        self.dece_distance = 5
        self.dece_target_speed = 5  # 3 will be safe
        self.need_decelerate = False
        self.scenario_operation = ScenarioOperation()
        self.trigger_distance_threshold = 35
        self.ego_max_driven_distance = 200
        self.behavior_type = 'cautious'
        self.basic_target_speed = 10

    def initialize_actors(self):
        first_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        second_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
        second_vehicle_waypoint = second_vehicle_waypoint.get_left_lane()
        # second_vehicle_waypoint = second_vehicle_waypoint.get_right_lane()
        first_vehicle_transform = carla.Transform(first_vehicle_waypoint.transform.location, first_vehicle_waypoint.transform.rotation)
        second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location, second_vehicle_waypoint.transform.rotation)

        self.actor_type_list = ['vehicle.nissan.patrol', 'vehicle.audi.tt']
        self.actor_transform_list = [first_vehicle_transform, second_vehicle_transform]
        self.actor_speed_list = [12, 12]
        self.other_actors = self.scenario_operation.initialize_vehicle_actors(self.actor_transform_list, self.actor_type_list)
        self.reference_actor = self.other_actors[0] # used for triggering this scenario
        #self.behavior_agent = BehaviorAgent(self.reference_actor, self.behavior_type) # set a behavior agent for controlling leading vehicle
        self.basic_agent = BasicAgent(self.reference_actor, self.basic_target_speed)  # set basic agent for controlling leading vehicle
        #print('self.reference_actor: ', first_vehicle_waypoint, type(first_vehicle_waypoint))
        #print('self.reference_actor waypoints: ', self.reference_actor, type(self.reference_actor))

        
    def create_behavior(self, scenario_init_action):
        assert scenario_init_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'

    def update_behavior(self, scenario_action):
        """
            Just make two vehicles move forward with specific speed
            At specific point, vehicle in front of ego will decelerate other_actors[0] is the vehicle before the ego
        """
        assert scenario_action is None, f'{self.name} should receive [None] action. A wrong scenario policy is used.'
        
        cur_distance = calculate_distance_transforms(self.actor_transform_list[0], CarlaDataProvider.get_transform(self.other_actors[0]))
        if cur_distance > self.dece_distance:
            self.need_decelerate = True
        for i in range(len(self.other_actors)):
            if i == 0 and self.need_decelerate:
               self.scenario_operation.go_straight(self.dece_target_speed, i)
            # if i == 0: # let leading vehicle drive as behavior agent
            #     #self.scenario_operation.drive_as_behavior_agent(self.behavior_agent, i)
            #     self.scenario_operation.drive_as_basic_agent(self.basic_agent, i)
            else:
                self.scenario_operation.go_straight(self.actor_speed_list[i], i)

    def check_stop_condition(self):
        pass
