
import time
import matplotlib.pyplot as plt

from kesslergame import Scenario, KesslerGame, GraphicsType
from jamie_controller import JamieController
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

asteroid_states = [
    {"position": (800, 57), "angle": 1, "speed": 50, "size": 3},
]

# Optional custom test scenario, not used in the tournament loop below
my_test_scenario = Scenario(
    name='Test Scenario',
    num_asteroids=7,
    ship_states=[
        {'position': (500, 400), 'angle': 180, 'lives': 3, 'team': 1, "mines_remaining": 3},
    ],
    map_size=(1000, 800),
    time_limit=100000000,
    ammo_limit_multiplier=0,
    stop_if_no_ammo=False
)

# Define Game Settings
game_settings = {
    'perf_tracker': True,
    'graphics_type': GraphicsType.NoGraphics,
    # If you want the full tournament to run faster with no visuals, use:
    # 'graphics_type': GraphicsType.NoGraphics,
    'realtime_multiplier': 0.0,
    'graphics_obj': None,
    'frequency': 30
}

# Controller labels must match the order used in the controllers list below.
controller_names = ["TEController", "JamieController"]

# Store asteroid-hit stats for each scenario
scenario_names = []
te_hits_by_scenario = []
jamie_hits_by_scenario = []

# Store totals
total_hits = {
    "TEController": 0,
    "JamieController": 0
}

# Run all scenarios in order
overall_start = time.perf_counter()

for scenario_index, scenario in enumerate(xfc2024):
    print(f"\nRunning scenario {scenario_index + 1}/{len(xfc2024)}: {scenario.name}")

    # Make a fresh game instance each run
    game = KesslerGame(settings=game_settings)

    # Make fresh controller instances each run
    controllers = [
        TEController(),
        JamieController()
    ]

    pre = time.perf_counter()
    score, perf_data = game.run(
        scenario=scenario,
        controllers=controllers
    )
    eval_time = time.perf_counter() - pre

    # Assumes score.teams order matches the controller/team order.
    asteroids_hit = [team.asteroids_hit for team in score.teams]
    deaths = [team.deaths for team in score.teams]
    accuracy = [team.accuracy for team in score.teams]
    mean_eval_time = [team.mean_eval_time for team in score.teams]

    te_hits = asteroids_hit[0]
    jamie_hits = asteroids_hit[1]

    scenario_names.append(scenario.name)
    te_hits_by_scenario.append(te_hits)
    jamie_hits_by_scenario.append(jamie_hits)

    total_hits["TEController"] += te_hits
    total_hits["JamieController"] += jamie_hits

    print('Scenario eval time: ' + str(eval_time))
    print(score.stop_reason)
    print('Asteroids hit: ' + str(asteroids_hit))
    print('Deaths: ' + str(deaths))
    print('Accuracy: ' + str(accuracy))
    print('Mean eval time: ' + str(mean_eval_time))

overall_time = time.perf_counter() - overall_start

print("\n================ Tournament Results ================")
print(f"Total eval time: {overall_time:.3f} seconds")
print(f"Total asteroids hit by TEController: {total_hits['TEController']}")
print(f"Total asteroids hit by JamieController: {total_hits['JamieController']}")

if total_hits["TEController"] > total_hits["JamieController"]:
    print("Winner by asteroid hits: TEController")
elif total_hits["JamieController"] > total_hits["TEController"]:
    print("Winner by asteroid hits: JamieController")
else:
    print("Tie by asteroid hits!")