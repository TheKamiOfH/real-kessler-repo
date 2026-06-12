#define imminent asteroids
#better turning logic using only max turns
from kesslergame import KesslerController
#from collisions import circle_line_collision_continuous
from typing import Dict, Tuple
from pprint import pprint
import math
import numpy as np
from scipy.optimize import fsolve
global actions
actions = {0:{"doing":False, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0, "tts": 0  }}

def circle_circle_collision_time_interval(
    ax: float, ay: float, vax: float, vay: float, ra: float,
    bx: float, by: float, vbx: float, vby: float, rb: float
) -> tuple[float, float]:
    """
    Returns (t_enter, t_exit) if the two circles will collide,
    or (nan, nan) if there's no collision in the future.
    Can return (-inf, inf) if the circles collide always and ever.
    """
    # Courtesy of Jie and Kessler Game
    # This linalg version is mathematically the same as setting up a quadratic and solving it, but is faster since it simplifies things
    nan = None
    separation = ra + rb

    dx = ax - bx
    dy = ay - by
    dvx = vax - vbx
    dvy = vay - vby

    dist_sq = dx * dx + dy * dy
    speed_sq = dvx * dvx + dvy * dvy
    dot = dx * dvx + dy * dvy
    sep_sq = separation * separation

    # Both stationary. Either overlapping forever or never
    if math.isclose(speed_sq, 0.0):
        if dist_sq <= sep_sq:
            return -math.inf, math.inf # Always overlapping
        else:
            return nan, nan # Never collide

    # Already outside and moving away (or tangent and moving apart)
    if dot >= 0.0 and dist_sq > sep_sq:
        return nan, nan

    # sin check: if angle too wide, paths never intersect within radius band
    cos_theta_sq = (dot * dot) / (dist_sq * speed_sq)
    sin_theta_sq = 1.0 - cos_theta_sq
    min_sin_sq = sep_sq / dist_sq

    if sin_theta_sq > min_sin_sq:
        return nan, nan  # Will miss each other

    # Compute collision time interval centered around closest approach
    root_term = math.sqrt((sep_sq - dist_sq * sin_theta_sq) / speed_sq)
    t_mid = -dot / speed_sq

    t_enter = t_mid - root_term
    t_exit  = t_mid + root_term

    return t_enter, t_exit

def do_drop_mine(game_state, ship_state, current_frame):
    count = 0
    i_count = 0
    d_count = 0
    d_score = math.inf
    #add a check based on number of asteroids in close proximity
    if ship_state["mines_remaining"] <= 0:
        return (False, 0, 0,0)
    for asteroid in game_state["asteroids"]:
        score = coll_evaluate_asteroid(ship_state, asteroid)
        d_score = math.inf
        for i in range(30):
            a_x = asteroid["position"][0] + asteroid["velocity"][0]*i*(1/30)
            a_y = asteroid["position"][1] + asteroid["velocity"][1]*i*(1/30)
            cache = distance(a_x,a_y,ship_state["position"][0],ship_state["position"][1])
            if cache < d_score:
                d_score = cache
        if score is not None and score < 30:
            count+=1
        if score is not None and score < 1 :
            i_count += 1
        if d_score is not None and d_score < 50:
            d_count+=1
    if count >= 5 or i_count >= 1 or d_score>10:
        return (True, count,i_count,d_count)
    return (False,0,0,0)
def canonize(asteroid,current_frame):
    delta_t = 1/30
    new_a_x = round((asteroid["position"][0] - asteroid["velocity"][0] * current_frame * delta_t)%1000,2)
    new_a_y = round((asteroid["position"][1] - asteroid["velocity"][1] * current_frame * delta_t)%800,2)
    return (new_a_x, new_a_y, asteroid["size"])
def rot_evaluate_asteroid(ship_state, asteroid):
    theta1 = math.atan2(asteroid["position"][1]-ship_state["position"][1], asteroid["position"][0]-ship_state["position"][0]) # account for bullet spawn here
    theta1 = math.degrees(theta1)%360
    angle_diff = (theta1 - ship_state["heading"])%360
    if angle_diff > 180:
        angle_diff -= 360
    frame_diff = math.ceil(abs(angle_diff) / 6)
    return frame_diff

def coll_evaluate_asteroid(ship_state, asteroid):
    tup = circle_circle_collision_time_interval(ship_state["position"][0], ship_state["position"][1],0,0,ship_state["radius"],asteroid["position"][0], asteroid["position"][1], asteroid["velocity"][0], asteroid["velocity"][1], asteroid["radius"])
    check = None
    if tup[0] is not None: 
        if tup[0] < 0:
            if tup[1] is None or tup[1] < 0:
                return None
            check = tup[1]
        else:
            check = tup[0]  
    if check is not None:
        if tup[0] == math.inf or tup[0] == -math.inf:
            return -math.inf
        return check
    return None

def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)
def can_shoot(actions,frame):
    for i in range(frame-3, frame):
        if i in actions.keys() and actions[i]["shooting"]:
            return False
    return True

def aim_bot(s_x,s_y,a_x,a_y,a_vx,a_vy,s_h) -> None:
    "Update the actions dict wih the appropriate turning and shooting actions"
    delta_t = 1/30
    current_frame = -1
    for i in actions.keys():
        if actions[i]["doing"]:
            continue
        else:
            current_frame = i
    if current_frame == -1:
        current_frame = max(actions.keys()) + 1
    for f in  range(0,330):
        last_thing = False
        new_a_x = (a_x + a_vx * f * delta_t)%1000
        new_a_y = (a_y + a_vy * f * delta_t)%800
        theta1 = math.atan2(new_a_y-s_y, new_a_x-s_x) # account for bullet spawn here
        theta1 = math.degrees(theta1)%360
        angle_diff = (theta1 - s_h)%360
        if angle_diff > 180:
            angle_diff -= 360
        sign = 1 if angle_diff > 0 else -1
        #turning_frames = math.ceil(abs(angle_diff) / 6)
        shooting_frames = math.ceil(distance(s_x, s_y, new_a_x, new_a_y) / (800/30))
        #rate = max(-180, min(180, angle_diff*30))
        
        if f < shooting_frames:
            continue
        turning_frames = f - shooting_frames
        if turning_frames <= 0:
            turning_frames = 1
        if abs(angle_diff)/turning_frames > 6:
            continue
        max_turn_frames = int(abs(angle_diff)/6)
        
        #max_turn_frames = math.floor(abs(angle_diff) / 6)
        if 6*max_turn_frames < abs(angle_diff):
            last_thing = True
        if last_thing:
            turning_frames = max_turn_frames + 1
        else:
            turning_frames = max_turn_frames
        if not can_shoot(actions,current_frame + turning_frames + shooting_frames):
            #print("Triggtered no shoot")
            continue
        if turning_frames + shooting_frames <= f:
            #decide what to do for the frames before we start turning
            extra_frames = f - max_turn_frames - shooting_frames
            for i in range(extra_frames):
                if current_frame + i not in actions.keys():
                    actions[current_frame + i] = {"doing":True, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0,"tts":0}
                action = actions[current_frame + i]
                action["doing"] = True
                action["turning"] = False
                action["shooting"] = False
            #update actions dict
            for i in range(max_turn_frames):
                #actions[current_frame + i] = {"turning":True, "shooting":False, "dropping_mine":False, "turn": sign * 6,}
                if current_frame + extra_frames + i not in actions.keys():
                    actions[current_frame + extra_frames + i] = {"doing":True, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0,"tts":0}
                action= actions[current_frame + extra_frames + i]
                action["doing"] = True
                action["turning"] = True
                action["turn"] = sign *6*30
            if last_thing: #and actions[current_frame + extra_frames + max_turn_frames]["tts"] == 0 or actions[current_frame + extra_frames + max_turn_frames]["tts"] == 1:
                if current_frame + max_turn_frames + extra_frames + 1 not in actions.keys():
                    actions[current_frame + max_turn_frames + extra_frames + 0 ] = {"doing":True, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0,"tts":0}
            
                action= actions[current_frame + max_turn_frames + extra_frames + 0]
           
           
                #still some turning to do after max turn frames, so shoot while turning
                action["turning"] = True
                action["doing"] = True
                action["turn"] = sign * (abs(angle_diff) - 6*max_turn_frames)*30
                action["shooting"] = True
                action["tts"] = 3
                '''
                for i in range(current_frame + extra_frames + max_turn_frames + 1, current_frame + extra_frames + max_turn_frames + 3):
                    if i not in actions.keys():
                        actions[i] = {"doing":False, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0,"tts":0}
                    action = actions[i]
                    prev_action = actions[i-1]
                    action["tts"] = prev_action["tts"] - 1
                '''
                break
            elif (not last_thing): #and actions[current_frame + extra_frames + max_turn_frames]["tts"] == 0 or actions[current_frame + extra_frames + max_turn_frames]["tts"] == 1:
                action["shooting"] = True
                '''
                for i in range(current_frame + extra_frames + max_turn_frames + 1, current_frame + extra_frames + max_turn_frames + 3):
                    if i not in actions.keys():
                        actions[i] = {"doing":False, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0,"tts":0}
                    action = actions[i]
                    prev_action = actions[i-1]
                    action["tts"] = prev_action["tts"] - 1
                '''
                break
            else:
                continue
        else:
            continue





class TEController(KesslerController):
   def __init__(self):
        """
        Any variables or initialization desired for the controller can be set up here
        """
        ...
        #self.dead_asteroids_dict = {} # Dictionary to keep track of dead asteroids
        self.d_list = [] #list of dead asteroids
        self.sequence = {}
        self.current_frame = 0
        self.closest_asteroid ={}
        self.targeted = False
        self.num_mines_to_drop = 0
        self.mine_dropped = False
        self.last_dropped_mine_frame = -100
        self.frame_to_drop = 0 # Track when the last mine was dropped
        self.dropped_mine_cuz_scared = -100  # Flag to indicate if a mine was dropped due to being scared
   def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
       invinc = ship_state["respawn_time_left"]
       doing = True if self.current_frame in actions and actions[self.current_frame]["doing"] else False
       if len(self.d_list) > 2 or len(self.d_list) >= len(game_state["asteroids"]) - 1:
           self.d_list = self.d_list[1:]
       if not doing:
           
           target = None
           score = 10000000000
           for asteroid in game_state["asteroids"]:
                key = canonize(asteroid,self.current_frame)
                if key not in self.d_list:
                     cache = coll_evaluate_asteroid(ship_state, asteroid)
                     if cache is not None and cache < score:
                         score = cache
                         target = asteroid
           if target is None:
                for asteroid in game_state["asteroids"]:
                    key = canonize(asteroid,self.current_frame)
                    if key not in self.d_list:
                        cache = rot_evaluate_asteroid(ship_state, asteroid)
                        if cache is not None and cache < score:
                            score = cache
                            target = asteroid
           if target is not None:
                       canon_pos = canonize(target,self.current_frame)
                       self.d_list.append(canon_pos)
                       aim_bot(ship_state["position"][0], ship_state["position"][1], target["position"][0], target["position"][1], target["velocity"][0], target["velocity"][1], ship_state["heading"])
                       if target["size"]>1:#if big boi wait a frame
                           i = len(actions.keys())
                           i+=1
                           actions[i] = {"doing":True, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0, "tts": 0  }
           #print(actions)
       fire = actions[self.current_frame]["shooting"] if self.current_frame in actions else False
       turn = actions[self.current_frame]["turn"] if self.current_frame in actions else 0
       thrust = 0
       drop_mine = False
       to_drop = do_drop_mine(game_state, ship_state, self.current_frame)
       if to_drop[0]:
           if to_drop[2] >=1 and self.current_frame - self.dropped_mine_cuz_scared > 100:
               self.last_dropped_mine_frame = self.current_frame
               self.dropped_mine_cuz_scared = self.current_frame
               drop_mine = True
           elif to_drop[1]>=5 and self.current_frame - self.last_dropped_mine_frame > 100:
               drop_mine = True
           #if to_drop[2]>=10 and self.current_frame - self.last_dropped_mine_frame > 100:
               #drop_mine = True
       if (game_state["time_limit"] != math.inf and game_state["time_limit"] - self.current_frame < 90 and len(game_state["asteroids"])>30):
           drop_mine = True
       if invinc>0:
           fire = False
           drop_mine = False
           if len(self.d_list)>0:
            self.d_list.pop()
           if self.last_dropped_mine_frame == self.current_frame:
               self.last_dropped_mine_frame = -100
           if self.dropped_mine_cuz_scared == self.current_frame:
               self.dropped_mine_cuz_scared = -100
       #drop_mine = actions[self.current_frame]["dropping_mine"] if self.current_frame in actions else False
       self.current_frame += 1
       return thrust, turn, fire, drop_mine
   @property
   def name(self) -> str:
        """
        Simple property used for naming controllers such that it can be displayed in the graphics engine

        Returns:
            str: name of this controller
        """
        return "T.E. v2.0"

        

        
    