# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import time

from kesslergame import Scenario, KesslerGame, GraphicsType
from test_controller import TestController


# Define game scenario
my_test_scenario = Scenario(name='Test Scenario',
                            #num_asteroids=20,
                            asteroid_states=[
                                {'position': (00, 50), 'angle': 2, 'size': 3, 'speed': 200},
                                {'position': (100, 750), 'angle': 21, 'size': 1, 'speed': 320},
                                {'position': (600, 512), 'angle': 29, 'size': 2, 'speed': 120},
                                {'position': (500, 100), 'angle': 0, 'size': 1, 'speed': 500},

                               
                                
                            ],
                            ship_states=[
                                {'position': (800, 400), 'angle': 0, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                # {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
                            ],
                            map_size=(1500, 900),
                            time_limit=100000,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)

# Define Game Settings
game_settings = {'perf_tracker': True,
                 'graphics_type': GraphicsType.Tkinter,
                 'realtime_multiplier': 0.5,
                 'graphics_obj': None,
                 'frequency': 30}

game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario
# game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

# Evaluate the game
pre = time.perf_counter()
score, perf_data = game.run(scenario=my_test_scenario, controllers=[TestController()])

# Print out some general info about the result
print('Scenario eval time: '+str(time.perf_counter()-pre))
print(score.stop_reason)
print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
print('Deaths: ' + str([team.deaths for team in score.teams]))
print('Accuracy: ' + str([team.accuracy for team in score.teams]))
print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))
