#define imminent asteroids
#better turning logic using only max turns
from kesslergame import KesslerController
from typing import Dict, Tuple
from pprint import pprint
import math
import numpy as np
from scipy.optimize import fsolve
global actions
actions = {0:{"doing":False, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0, "tts": 0  }}



def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

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
        new_a_x = (a_x + a_vx * f * delta_t)%1000
        new_a_y = (a_y + a_vy * f * delta_t)%600
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
                    actions[current_frame + max_turn_frames + extra_frames + 1 ] = {"doing":True, "turning":False, "shooting":False, "dropping_mine":False, "turn": 0,"tts":0}
            
                action= actions[current_frame + max_turn_frames + extra_frames + 1]
           
           
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
        self.dead_asteroids_dict = {} # Dictionary to keep track of dead asteroids
        self.sequence = {}
        self.current_frame = 0
        self.closest_asteroid ={}
        self.targeted = False
        self.num_mines_to_drop = 0
        self.mine_dropped = False
        self.frame_to_drop = 0 # Track when the last mine was dropped
        self.dropped_mine_cuz_scared = False  # Flag to indicate if a mine was dropped due to being scared
   def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
       
       doing = True if self.current_frame in actions and actions[self.current_frame]["doing"] else False
       if not doing:
           aim_bot(ship_state["position"][0], ship_state["position"][1], game_state["asteroids"][0]["position"][0], game_state["asteroids"][0]["position"][1], game_state["asteroids"][0]["velocity"][0], game_state["asteroids"][0]["velocity"][1], ship_state["heading"])
           print(actions)
       fire = actions[self.current_frame]["shooting"] if self.current_frame in actions else False
       turn = actions[self.current_frame]["turn"] if self.current_frame in actions else 0
       thrust = 0
       drop_mine = actions[self.current_frame]["dropping_mine"] if self.current_frame in actions else False
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

        

        
    