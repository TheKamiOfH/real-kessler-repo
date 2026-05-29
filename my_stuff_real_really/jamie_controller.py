# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

#THIS PROGRAM IS FREE SOFTWARE.

from kesslergame import KesslerController
from typing import Dict, Tuple
import math
import numpy as np

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import copy

# Turn on and off explanation
print_explanation = False
do_log_explanation = False

mine_preventative_lookahead_frames = 8
super_mario_fudge_factor_64_for_gnuke_mode = 17
size_thresh_closing_ring = 3

# Distance (0 to 1000 px)
distance = ctrl.Antecedent(np.arange(0, 1001, 1), 'distance')
# Speed (-240 to 240 px/s)
speed = ctrl.Antecedent(np.arange(-240, 241, 1), 'speed')
# Thrust (-480 to 480 px/s²)
thrust = ctrl.Consequent(np.arange(-480, 481, 1), 'thrust')

# Distance memberships
distance['very_close'] = fuzz.trapmf(distance.universe, [0, 0, 40, 80])
distance['close'] = fuzz.trimf(distance.universe, [60, 120, 180])
distance['medium'] = fuzz.trimf(distance.universe, [160, 350, 550])
distance['far'] = fuzz.trapmf(distance.universe, [500, 700, 1000, 1000])

# Speed memberships
speed['reverse'] = fuzz.trapmf(speed.universe, [-240, -240, -200, -140])
speed['normal_reverse'] = fuzz.trimf(speed.universe, [-180, -120, -60])
speed['zero'] = fuzz.trimf(speed.universe, [-80, 0, 80])
speed['normal_forward'] = fuzz.trimf(speed.universe, [60, 120, 180])
speed['forward'] = fuzz.trapmf(speed.universe, [140, 180, 240, 240])

# Thrust memberships
thrust['strong_reverse'] = fuzz.trapmf(thrust.universe, [-480, -480, -360, -240])
thrust['normal_reverse'] = fuzz.trimf(thrust.universe, [-300, -180, -60])
thrust['coast'] = fuzz.trimf(thrust.universe, [-80, 0, 80])
thrust['normal_forward'] = fuzz.trimf(thrust.universe, [60, 180, 300])
thrust['strong_forward'] = fuzz.trapmf(thrust.universe, [240, 360, 480, 480])

rules = [
    ctrl.Rule(distance['far'] & speed['reverse'], thrust['strong_forward']),
    ctrl.Rule(distance['far'] & speed['normal_reverse'], thrust['strong_forward']),
    ctrl.Rule(distance['far'] & speed['zero'], thrust['strong_forward']),
    ctrl.Rule(distance['far'] & speed['normal_forward'], thrust['normal_forward']),
    ctrl.Rule(distance['far'] & speed['forward'], thrust['normal_forward']),

    ctrl.Rule(distance['medium'] & speed['reverse'], thrust['strong_forward']),
    ctrl.Rule(distance['medium'] & speed['zero'], thrust['strong_forward']),
    ctrl.Rule(distance['medium'] & speed['normal_forward'], thrust['strong_forward']),
    ctrl.Rule(distance['medium'] & speed['forward'], thrust['normal_forward']),

    ctrl.Rule(distance['close'] & speed['normal_forward'], thrust['coast']), #what
    ctrl.Rule(distance['close'] & speed['forward'], thrust['coast']), #what
    ctrl.Rule(distance['close'] & speed['zero'], thrust['coast']),
    ctrl.Rule(distance['close'] & speed['normal_reverse'], thrust['normal_forward']),

    ctrl.Rule(distance['very_close'] & speed['forward'], thrust['strong_reverse']),
    ctrl.Rule(distance['very_close'] & speed['normal_forward'], thrust['normal_reverse']),
    ctrl.Rule(distance['very_close'] & speed['zero'], thrust['strong_reverse']),
    ctrl.Rule(distance['very_close'] & speed['reverse'], thrust['coast']),
]
thrust_ctrl = ctrl.ControlSystem(rules)
thrust_sim = ctrl.ControlSystemSimulation(thrust_ctrl)

#fig, axs = plt.subplots(3, 1, figsize=(12, 12))
#distance.view(ax=axs[0])
#axs[0].set_title('Distance Membership Functions')

#speed.view(ax=axs[1])
#axs[1].set_title('Speed Membership Functions')

#thrust.view(ax=axs[2])
#axs[2].set_title('Thrust Membership Functions')

#plt.tight_layout()
#plt.show()

last_message = ""
def fudge_game_state(game_state):
    # Convert immutabledict to regular dict if needed
    game_state = dict(game_state)

    fudged_asteroids = []

    for asteroid in game_state['asteroids']:
        if asteroid['size'] == 4:
            x, y = asteroid['position']

            # Original asteroid
            fudged_asteroids.append(asteroid)

            # Three "twin" asteroids with tweaked positions
            twin1 = asteroid.copy()
            twin1['position'] = (x + 1, y)
            fudged_asteroids.append(twin1)

            twin2 = asteroid.copy()
            twin2['position'] = (x - 1, y)
            fudged_asteroids.append(twin2)

            twin3 = asteroid.copy()
            twin3['position'] = (x, y + 1)
            fudged_asteroids.append(twin3)

            twin4 = asteroid.copy()
            twin4['position'] = (x -1, y + 1 )
            fudged_asteroids.append(twin4)

            twin5 = asteroid.copy()
            twin5['position'] = (x + 1, y + 1)
            fudged_asteroids.append(twin5)

            twin6 = asteroid.copy()
            twin6['position'] = (x - 1, y - 1)
            fudged_asteroids.append(twin6)

            twin7 = asteroid.copy()
            twin7['position'] = (x + 1, y - 1)
            fudged_asteroids.append(twin7)

            twin8 = asteroid.copy()
            twin8['position'] = (x, y - 1)
            fudged_asteroids.append(twin8)


        else:
            # Keep non-size 4 asteroids as is
            fudged_asteroids.append(asteroid)

    game_state['asteroids'] = fudged_asteroids
    return game_state

def log_explanation(message: str):
    global last_message
    if last_message == message:
        return
    else:
        if print_explanation:
            print(message)
        if do_log_explanation:
            with open('explanations_jamie.txt', 'a+') as file:
                file.write(message + '\n')
        last_message = message

def wrapped_distance(x1, y1, x2, y2, map_size):
    width, height = map_size
    dx = min(abs(x1 - x2), width - abs(x1 - x2))
    dy = min(abs(y1 - y2), height - abs(y1 - y2))
    return math.sqrt(dx**2 + dy**2)

def time_to_collision_wrapped(ship_pos, ship_radius, asteroid_pos, asteroid_velocity, asteroid_radius, map_size):
    x_s, y_s = ship_pos
    v_ax, v_ay = asteroid_velocity
    r_s = ship_radius
    r_a = asteroid_radius

    wrapped_positions = [
        (asteroid_pos[0] + dx, asteroid_pos[1] + dy)
        for dx in [-map_size[0], 0, map_size[0]]
        for dy in [-map_size[1], 0, map_size[1]]
    ]

    best_time = float('inf')

    for x_a, y_a in wrapped_positions:
        a = v_ax**2 + v_ay**2
        if a == 0:
            distance = math.sqrt((x_s - x_a)**2 + (y_s - y_a)**2)
            if distance <= r_s + r_a:
                return 0
            continue

        b = 2 * ((x_a - x_s) * v_ax + (y_a - y_s) * v_ay)
        c = (x_s - x_a)**2 + (y_s - y_a)**2 - (r_s + r_a)**2
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            continue

        t1 = (-b - math.sqrt(discriminant)) / (2 * a)
        t2 = (-b + math.sqrt(discriminant)) / (2 * a)
        if t1 > 0:
            best_time = min(best_time, t1)
        elif t2 > 0:
            best_time = min(best_time, t2)

    return best_time

def canonicalize_asteroid(asteroid, game_time, map_size):
    # Take me back to the start babyyyyyy
    x, y = asteroid['position']
    vx, vy = asteroid['velocity']
    width, height = map_size
    x0 = (x - vx * game_time) % width
    y0 = (y - vy * game_time) % height
    return (round(x0, 5), round(y0, 5))

def predict_imminent_collision(ship_state: Dict, game_state: Dict, delta_time: float) -> bool:
    ship_pos = ship_state['position']
    ship_radius = 20
    map_size = game_state['map_size']
    ship_speed = ship_state['speed']
    ship_heading_rad = math.radians(ship_state['heading'])
    ship_vel = (
        ship_speed * math.cos(ship_heading_rad),
        ship_speed * math.sin(ship_heading_rad)
    )

    for asteroid in game_state['asteroids']:
        ast_pos = asteroid['position']
        ast_vel = asteroid['velocity']
        ast_radius = asteroid['radius']
        #rel_vel = (ast_vel[0] - ship_vel[0], ast_vel[1] - ship_vel[1])
        # jk we aint movin
        future_x = ast_pos[0]
        future_y = ast_pos[1]
        for _ in range(mine_preventative_lookahead_frames):
            future_x += ast_vel[0] * delta_time
            future_y += ast_vel[1] * delta_time
            if (future_x - ship_pos[0])**2 + (future_y - ship_pos[1])**2 <= (ship_radius + ast_radius)**2:
                return True
    return False

def prioritize_imminent_collision(ship_state: Dict, game_state: Dict, closing_ring: bool) -> Dict:
    ship_pos = ship_state['position']
    ship_radius = ship_state['radius']
    map_size = game_state['map_size']
    bullet_speed = 800
    imminent_asteroid = None
    shortest_time = float('inf')

    for asteroid in game_state['asteroids']:
        if closing_ring and asteroid['size'] <= size_thresh_closing_ring and game_state['sim_frame'] < 5*30:
            continue
        t_col = time_to_collision_wrapped(ship_pos, ship_radius, asteroid['position'], asteroid['velocity'], asteroid['radius'], map_size)
        if t_col < shortest_time:
            shortest_time = t_col
            imminent_asteroid = asteroid
    if len(game_state['asteroids']) > 2 and imminent_asteroid is None:
        # relax bro
        for asteroid in game_state['asteroids']:
            t_col = time_to_collision_wrapped(ship_pos, ship_radius, asteroid['position'], asteroid['velocity'], asteroid['radius'], map_size)
            if t_col < shortest_time:
                shortest_time = t_col
                imminent_asteroid = asteroid
    return imminent_asteroid

class JamieController(KesslerController):
    def __init__(self, fudge_fact = None):
        global super_mario_fudge_factor_64_for_gnuke_mode
        self.first_scen_count = 1
        self.prev_lives = None
        self.last_mine_time = -10
        self.mine_cooldown = 3.0
        self.use_forecasted_fudging = False
        self.asteroids_targeted = {}
        self.last_frame_life = None
        self.last_frame_life_lost = None
        self.fire_this_fram = False
        self.fire_next_fram = False
        self.is_closing_ring = False
        if fudge_fact is not None:
            super_mario_fudge_factor_64_for_gnuke_mode = fudge_fact
        #make better to win 
    
    def gnuke_mode(self, ship_state, game_state) -> tuple[bool, list[int]]:
        global super_mario_fudge_factor_64_for_gnuke_mode
        is_closing_ring = False
        list_of_frames_to_drop_mines = []
        # Do a for loop up to 10 seconds into the future to detect a closing ring. CLosing rings must happen within this time
        asteroids = copy.deepcopy(game_state['asteroids'])
        initial_asts_count = len(asteroids)
        threshold_fraction_of_asteroids_to_signify_closing_ring = 1/3 # 1/2 is prob fine but let's just do this to be safe
        # Find pos of asteroids for each frame
        ship_radius_to_check_closing_ring = 100 # px
        initial_mine_drop_frame = -10000
        #print("This should be executed")
        for frame in range(0, 10*30):
            count_of_asts_within_mine_radius = 0
            for a in asteroids:
                if wrapped_distance(ship_state['position'][0], ship_state['position'][1], a['position'][0], a['position'][1], game_state['map_size']) < ship_radius_to_check_closing_ring:
                #if wrapped_distance(ship_state['position'][0], ship_state['position'][1], a['position'][0], a['position'][1], game_state['map_size']) in [int(ship_radius_to_check_closing_ring - tol,:
                    count_of_asts_within_mine_radius += 1
            if count_of_asts_within_mine_radius/initial_asts_count > threshold_fraction_of_asteroids_to_signify_closing_ring:
                log_explanation("CLOSING RING SCENARIO DETECTED! ACTIVATING GNUKE MODE!!!")
                #print("CLosing ring detected")
                is_closing_ring = True
                break

            for a in asteroids:
                a['position'] = ((a['position'][0] + a['velocity'][0]/30.0)%game_state['map_size'][0], (a['position'][1] + a['velocity'][1]/30.0)%game_state['map_size'][1])
        # Now we iterate through a second time to find when to drop the mines
        if is_closing_ring:
            #print("GOING INTO DEEPER GNUKE CODE")
            asteroids = copy.deepcopy(game_state['asteroids'])
            for frame in range(0, 10*30):
                #print(frame)
                count_of_asts_within_mine_radius = 0
                
                for a in asteroids:
                    if wrapped_distance(ship_state['position'][0], ship_state['position'][1], a['position'][0], a['position'][1], game_state['map_size']) < 150 + a['radius'] - 1:
                    #if wrapped_distance(ship_state['position'][0], ship_state['position'][1], a['position'][0], a['position'][1], game_state['map_size']) in [int(ship_radius_to_check_closing_ring - tol,:
                        count_of_asts_within_mine_radius += 1
                if count_of_asts_within_mine_radius/initial_asts_count > threshold_fraction_of_asteroids_to_signify_closing_ring:
                    log_explanation("GNUKE MODE FOUND FRAME THAT MINE CAN BE DETONATED IN AND FREEZE STUFF")
                    #print(f"{frame} GNUKE MODE FOUND FRAME THAT MINE CAN BE DETONATED IN AND FREEZE STUFF")
                    print("GNUKE MODE ACTIVATING!!!! KABOOM!!!!")
                    initial_mine_drop_frame = frame - 3*30
                    break

                for a in asteroids:
                    a['position'] = ((a['position'][0] + a['velocity'][0]/30.0)%game_state['map_size'][0], (a['position'][1] + a['velocity'][1]/30.0)%game_state['map_size'][1])
        if initial_mine_drop_frame != -10000:
            #print(super_mario_fudge_factor_64_for_gnuke_mode)
            initial_mine_drop_frame += super_mario_fudge_factor_64_for_gnuke_mode
            initial_mine_drop_frame = max(initial_mine_drop_frame, 0)
            list_of_frames_to_drop_mines = [initial_mine_drop_frame, initial_mine_drop_frame + 30, initial_mine_drop_frame + 30 + 30]
        if is_closing_ring:
            log_explanation(f"{list_of_frames_to_drop_mines=}")
            #print(list_of_frames_to_drop_mines)
        return is_closing_ring, list_of_frames_to_drop_mines

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        current_time = game_state['time']
        if current_time == 0:
            count_asts = 0
            count_size_4_asts = 0
            for a in game_state['asteroids']:
                if a['size'] == 4:
                    count_size_4_asts += 1
            count_asts = len(game_state['asteroids'])
            if count_asts < 10 and count_size_4_asts > 2:
                self.use_forecasted_fudging = True

            print(game_state)
        if self.use_forecasted_fudging:
            game_state = fudge_game_state(game_state)
        if game_state["sim_frame"] == 0:
            self.use_forecasted_fudging = False
            self.first_scen_count -= 1
            self.last_frame_life = ship_state['lives_remaining']
            self.last_frame_life_lost = -1
            self.prev_lives = None
            self.last_mine_time = -10
            self.mine_cooldown = 3.0
            self.asteroids_targeted = {}
            self.fire_this_fram = False
            self.fire_next_fram = False
            #print(f"It is timestep {game_state['time']} and we're calling gnuke code")
            self.is_closing_ring, self.list_of_frames_to_drop_mines = self.gnuke_mode(ship_state, game_state)
        if ship_state['lives_remaining'] < self.last_frame_life:
            log_explanation("OUCH life lost")
            self.last_frame_life_lost = game_state['sim_frame']
        self.last_frame_life = ship_state['lives_remaining']
        ship_x, ship_y = ship_state['position'] #(log_explanation = [position])
        ship_heading_deg = ship_state['heading'] 
        ship_heading_rad = math.radians(ship_heading_deg)
        bullet_speed = 800
       
        if current_time == 0:
            print(game_state)
        delta_time = game_state['delta_time']
        map_size = game_state['map_size']
        self.fire_this_fram = self.fire_next_fram
        log_explanation(f"Frame {game_state['sim_frame']}")
        # Expire old targeting info
        self.asteroids_targeted = {
            k: v for k, v in self.asteroids_targeted.items() if v > current_time
        } 
        thrust = 0
        turn = 0
        fire = False
        drop_mine = False
        
        # Mine drop logic
        drop_mine = False
        if not self.is_closing_ring:
            # Do regular mine drop logic
            if (ship_state['can_deploy_mine'] and ship_state['mines_remaining'] > 0 and current_time - self.last_mine_time >= self.mine_cooldown):
                if predict_imminent_collision(ship_state, game_state, delta_time):
                    asts = copy.deepcopy(game_state['asteroids'])
                    # MOve all asts 3 secs into future and check if any of them would get blastsed by a mine i drop right noww
                    blat_count = 0
                    for a in asts:
                        a['position'] = ((a['position'][0] + 3*a['velocity'][0])%game_state['map_size'][0], (a['position'][1] + 3*a['velocity'][1])%game_state['map_size'][1])
                        if wrapped_distance(ship_state['position'][0], ship_state['position'][1], a['position'][0], a['position'][1], game_state['map_size']) < 150 + a['radius']:
                            blat_count += 1
                    if blat_count > 0:
                        log_explanation("Damage incoming, dropping preventitative mine")
                        drop_mine = True
                        self.last_mine_time = current_time
        else:
            # DO GNUKE MODE LOGIC FOR MINEES
            if game_state['sim_frame'] in self.list_of_frames_to_drop_mines:
                drop_mine = True
            else:
                drop_mine = False
        dont_spray = False
        self.prev_lives = ship_state['lives_remaining']
        target_asteroid = None
        if ship_state['bullets_remaining'] != 0:
            most_imminent_ast = prioritize_imminent_collision(ship_state, game_state, self.is_closing_ring)
            for asteroid in game_state['asteroids']:
                if self.is_closing_ring and asteroid['size'] <= size_thresh_closing_ring and game_state['sim_frame'] < 5*30:
                    continue
                # Go through asterdois and foind target
                canon_key = canonicalize_asteroid(asteroid, current_time, map_size)
                if canon_key in self.asteroids_targeted:
                    continue # don't waste bullet, bad for environment badddd

                ast_x, ast_y = asteroid['position']
                ast_vx, ast_vy = asteroid['velocity']
                t = 0
                for _ in range(5):
                    # iterateive search to home in on solution (but never actually get there)
                    future_x = ast_x + ast_vx * t
                    future_y = ast_y + ast_vy * t
                    dist = math.sqrt((future_x - ship_x)**2 + (future_y - ship_y)**2)
                    t = dist / bullet_speed

                future_x = ast_x + ast_vx * (t + 1/30)
                future_y = ast_y + ast_vy * (t + 1/30)
                angle_to = math.atan2(future_y - ship_y, future_x - ship_x)
                angle_diff = ((angle_to - ship_heading_rad + math.pi) % (2 * math.pi)) - math.pi
                angle_diff_deg = abs(math.degrees(angle_diff))
                if asteroid is most_imminent_ast:
                    target_asteroid = (asteroid, angle_diff_deg, t, canon_key, future_x, future_y)
                    break
                if target_asteroid is None or angle_diff_deg < target_asteroid[1]:
                    target_asteroid = (asteroid, angle_diff_deg, t, canon_key, future_x, future_y)

        my_team = ship_state['team']
        enemies = [s for s in game_state['ships'] if s['team'] != my_team]
        if enemies and enemies[0]['lives_remaining'] < ship_state['lives_remaining']:
            log_explanation(f"I have {ship_state['lives_remaining']} lives, which is more than my opponent's {enemies[0]['lives_remaining']} lives. Good time for RAM mode maybe...")
            greater_lives = True
        else:
            greater_lives = False
        #if ((target_asteroid is None and not self.fire_this_fram) or ship_state['lives_remaining'] >= 3)*0  and greater_lives:
        if (greater_lives and ship_state['is_respawning']):
            log_explanation("We're doing RAM mode!")
            ram_mode = True
        else:
            log_explanation("Not doing RAM mode")
            ram_mode = False

        if target_asteroid:
            asteroid, angle_diff_deg, intercept_time, canon_key, future_x, future_y = target_asteroid

            angle_to = math.atan2(future_y - ship_y, future_x - ship_x)
            angle_diff = ((angle_to - ship_heading_rad + math.pi) % (2 * math.pi)) - math.pi
            angle_diff_deg = math.degrees(angle_diff)

            turn = max(-180, min(angle_diff_deg * 30, 180))
            self.fire_next_fram = abs(angle_diff_deg) < 6# or (angle_diff_deg > 18 and ship_state['bullets_remaining'] > 35)

            if abs(angle_diff_deg) < 6: # 6 is deg ship turns in one frame
                log_explanation('I locked onto an asteroid!')
                self.asteroids_targeted[canon_key] = current_time + intercept_time # secs
            elif abs(angle_diff_deg) < 18: # 18 is deg ship turns in one shot cooldown cycle
                dont_spray = True

        # RAM mode
        #ram_mode= True 
        if ram_mode:
            drop_mine = False
            fire = False
            my_team = ship_state['team']
            enemies = [s for s in game_state['ships'] if s['team'] != my_team]
            if enemies:
                try:
                    # Find closest enemy
                    closest = min(
                        enemies,
                        key=lambda s: wrapped_distance(ship_x, ship_y, s['position'][0], s['position'][1], map_size)
                    )
                    target_x, target_y = closest['position']

                    # Calculate turning angle
                    dx = target_x - ship_x
                    dy = target_y - ship_y
                    desired = math.degrees(math.atan2(dy, dx))
                    angle_diff = ((desired - ship_heading_deg + 180) % 360) - 180
                    turn = max(-180, min(angle_diff * 5, 180))

                    # Fuzzy logic inputs
                    distance_val = wrapped_distance(ship_x, ship_y, target_x, target_y, map_size)
                    speed_val = ship_state['speed']

                    # Apply fuzzy wuzzy logic
                    thrust_sim.input['distance'] = max(min(distance_val, 1000), 0)
                    thrust_sim.input['speed'] = max(min(speed_val, 240), -240)
                    thrust_sim.compute()
                    fuzzy_thrust = thrust_sim.output['thrust']

                    log_explanation(f"[RAM] Distance: {distance_val:.1f}, Speed: {speed_val:.1f} => Thrust: {fuzzy_thrust:.1f}, Turn: {turn:.1f}")
                    return fuzzy_thrust, turn, False, drop_mine
                except Exception as e:
                    log_explanation(f"fuzzy ram error: {e}")
                    return 0, 0, False, False
            else:
                return 0, 0, False, False

        if ship_state['is_respawning']:
            iframes_left = 3*30 - (game_state['sim_frame'] - self.last_frame_life_lost)
            asts = copy.deepcopy(game_state['asteroids'])
            # MOve all asts 3 secs into future and check if any of them would get blastsed by a mine i drop right noww
            freedom_flsg = False
            for m in game_state['mines']:
                if wrapped_distance(ship_state['position'][0], ship_state['position'][1], m['position'][0], m['position'][1], game_state['map_size']) <= ship_state['radius'] + 150:
                    #m['time_remaining'] <= iframes_left/30 and 
                    freedom_flsg = True
            if not freedom_flsg:
                for frame in range(iframes_left):
                    if freedom_flsg:
                        break
                    for a in asts:
                        a['position'] = ((a['position'][0] + a['velocity'][0]/30)%game_state['map_size'][0], (a['position'][1] + a['velocity'][1]/30)%game_state['map_size'][1])
                        if wrapped_distance(ship_state['position'][0], ship_state['position'][1], a['position'][0], a['position'][1], game_state['map_size']) <= ship_state['radius'] + a['radius']:
                            # We gon get hit so we need to keep the invincibility
                            freedom_flsg = True
                            break
        
        if self.fire_this_fram is True and (not ship_state['is_respawning'] or not freedom_flsg):
            fire = True
        else:
            fire = False
        if ship_state['can_fire'] and not (0 <= ship_state['bullets_remaining'] <= 35) and not self.fire_next_fram:
            if not ship_state['is_respawning'] and not dont_spray:
                fire = True
            else:
                fire = False          
        return thrust, turn, fire, drop_mine

    @property
    def name(self) -> str:
        return "Nexus"

    #@property
    #def custom_sprite_path(self) -> str:
        #return "prideship.png"