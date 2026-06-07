# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import time

from kesslergame import Scenario, KesslerGame, GraphicsType
#from test_controller import TestController
#from jamie_controller import JamieController
from TE_vtwo import TEController
from adversarial_scenarios_for_jie import *
xfc2024 = [
    adv_random_small_1,
    adv_random_small_1_2,
    adv_multi_wall_left_easy,
    adv_multi_four_corners,
    adv_multi_wall_top_easy,
    adv_multi_2wall_closing,
    adv_wall_bottom_staggered,
    adv_multi_wall_right_hard,
    adv_moving_corridor_angled_1,
    adv_moving_corridor_angled_1_mines,
    adv_multi_ring_closing_left,
    adv_multi_ring_closing_left2,
    adv_multi_ring_closing_both2,
    adv_multi_ring_closing_both_inside_fast,
    adv_multi_two_rings_closing
]

asteroid_states=[
                              #{"position": (101, 657), "angle": 1, "speed": 151, "size": 1},
                              {"position": (800, 57), "angle": 1, "speed": 565,"size": 1},
                            ]
# Define game scenario
my_test_scenario = Scenario(name='Test Scenario',
                           asteroid_states=asteroid_states,
                            ship_states=[
                                {'position': (500, 400), 'angle': 180, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                # {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
                            ],
                            map_size=(1000, 800),
                            time_limit=100000000,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)

# Define Game Settings
game_settings = {'perf_tracker': True,
                 'graphics_type': GraphicsType.Tkinter,
                 'realtime_multiplier': 1,
                 'graphics_obj': None,
                 'frequency': 30}

game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario
# game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

# Evaluate the game
pre = time.perf_counter()
score, perf_data = game.run(scenario=my_test_scenario, controllers=[TEController()])

# Print out some general info about the result
print('Scenario eval time: '+str(time.perf_counter()-pre))
print(score.stop_reason)
print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
print('Deaths: ' + str([team.deaths for team in score.teams]))
print('Accuracy: ' + str([team.accuracy for team in score.teams]))
print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))