# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from kesslergame import KesslerController
from typing import Dict, Tuple
from pprint import pprint
import math
import numpy as np
from scipy.optimize import fsolve

def new_solve_motion_time(X, Y, theta1_deg, V0, Vb=80/3):
    denom = math.sqrt(X**2 + Y**2)
    k = V0 / Vb
    sinTheta = math.sin(math.radians(theta1_deg))
    cosTheta = math.cos(math.radians(theta1_deg))
    # Calculate theta2 using the formula
    theta2_rad = math.atan2(Y,X) + math.asin((k*(X*sinTheta - Y*cosTheta)) / denom)
    theta2_deg = math.degrees(theta2_rad)
    #print(f"theta2_deg: {theta2_deg}")
    cos_theta2 = np.cos(theta2_rad)
    sin_theta2 = np.sin(theta2_rad)
    X_prime = (X / (1 - k * cosTheta / cos_theta2))
    Y_prime = (Y / (1 - k * sinTheta / sin_theta2))

    # Compute distance and time
    distance = np.sqrt((X_prime - X)**2 + (Y_prime - Y)**2)
    time = distance / V0
    return {
        "theta2_deg": theta2_deg,
        "X_prime": X_prime,
        "Y_prime": Y_prime,
        "distance": distance,
        "time": time
    }
def solve_motion_time(X, Y, theta1_deg, V0, Vb=80/3):
    # Convert theta1 to radians
    theta1 = np.radians(theta1_deg)
    
    # Constants
    k = V0 / Vb
    cos_theta1 = np.cos(theta1)
    sin_theta1 = np.sin(theta1)
    #Y_over_X = Y / X
    # Solve for theta2 numerically (initial guess: theta1)
    # Define the equation to solve for theta2
    def equation(theta2_rad):
        cos_theta2 = np.cos(theta2_rad)
        sin_theta2 = np.sin(theta2_rad)
        lhs = Y*(cos_theta2 - k * cos_theta1) 
        rhs = X*(sin_theta2 - k * sin_theta1)
        return lhs - rhs

    theta2_rad = fsolve(equation, theta1)[0]
    theta2_deg = np.degrees(theta2_rad)

    # Compute X' and Y' using solved theta2
    cos_theta2 = np.cos(theta2_rad)
    sin_theta2 = np.sin(theta2_rad)
    X_prime = X / (1 - k * cos_theta1 / cos_theta2)
    Y_prime = Y / (1 - k * sin_theta1 / sin_theta2)

    # Compute distance and time
    distance = np.sqrt((((X_prime - X)))**2 + (((Y_prime - Y)))**2)
    time = distance / V0
    print(f"X: {X}, Y: {Y}, theta1: {theta1_deg}, V0: {V0}, X': {X_prime}, Y': {Y_prime}, distance: {distance}, time: {time}, theta2: {theta2_deg}")



    

    return {
        "theta2_deg": theta2_deg,
        "X_prime": X_prime,
        "Y_prime": Y_prime,
        "distance": distance,
        "time": time
    }

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points.

    Arguments:
        p1 (tuple): First point as (x, y).
        p2 (tuple): Second point as (x, y).

    Returns:
        float: The distance between the two points.
    """
    return math.sqrt((((p1[0] - p2[0]))) ** 2 + (((p1[1] - p2[1]))) ** 2)

def magnitude(v: Tuple[float, float]) -> float:
    """
    Calculate the magnitude of a 2D vector.

    Arguments:
        v (tuple): A 2D vector as (vx, vy).

    Returns:
        float: The magnitude of the vector.
    """
    return math.sqrt(v[0] ** 2 + v[1] ** 2)
def back_to_zero(closest_asteroid : dict, game_state : dict) -> dict:
    """
    Bringing crap back to zero
    """
    #frames or seconds?
    asteroid = dict(closest_asteroid)
    asteroid['position'] = (round(closest_asteroid['position'][0] - (closest_asteroid['velocity'][0] * game_state['time']) % 1000, 5),
                                        round(closest_asteroid['position'][1] - (closest_asteroid['velocity'][1] * game_state['time']) % 800, 5))
    return asteroid['position']
def will_collide(ship_state: Dict, game_state: Dict, frame_number: int) -> list:
    """
    Check if the ship will collide with any asteroids.
    Arguments:
        ship_state (dict): Contains state information for your own ship.
        game_state (dict): Contains state information for all objects in the game.
        frame_number (int): The current frame number in the game.
    Returns:
        List: Returns a list of asteroids that will collide with the ship and the number of frames until collision.
    """
    collisions = []
    for asteroid in game_state['asteroids']:
        f = frame_number + 1 - 1  # To prevent aliasing
        asteroid_copy = dict(asteroid)  # Create a copy of the asteroid to avoid modifying the original
        theta1 = math.degrees(math.atan2(asteroid_copy['velocity'][1], asteroid_copy['velocity'][0]))  # Angle of the asteroid's velocity
        initial_turning = new_solve_motion_time(asteroid_copy['position'][0] - ship_state['position'][0],
                                                asteroid_copy['position'][1] - ship_state['position'][1], theta1,
                                                magnitude(asteroid_copy['velocity']) / 30)  # Divide by 30
        initial_turning = (initial_turning['theta2_deg'] - ship_state["heading"])%360  # Get the angle to turn to hit the asteroid
        if initial_turning >= 180:  # Normalize to [-180, 180]
            initial_turning -= 360
        if abs(initial_turning) <= 6:
            initial_turning = 0
        else:
            initial_turning = 9999999999999
        while distance(ship_state['position'], asteroid_copy['position']) > 22 and f < frame_number + 100:  # Assuming ship radius is 22
            # Change 25
            # Update asteroid position based on its velocity
            asteroid_copy['position'] = ((asteroid['position'][0] + asteroid['velocity'][0]*(f-frame_number) / 30)%1000,  
                                    (asteroid['position'][1] + asteroid['velocity'][1]*(f-frame_number)/ 30)%800)  # Assuming 30 FPS
            if distance(ship_state['position'], asteroid_copy['position']) < 22:  # Assuming ship radius is 22
                collisions.append([asteroid, f, 0, initial_turning])  # 1 frame until collision
                break  # Collision detected, break out of the loop
            else:
                f+=1
        else:
            collisions.append([asteroid, f, 0, initial_turning])  # No collision within the next 25 frames

    return sorted(collisions, key=lambda x: x[1])  # Sort by number of frames until collision
def in_ring(asteroid: dict, ship_state: dict) -> bool:
    """
    Check if the asteroid is within the ring of destruction.
    """
    x = abs(asteroid['position'][0] - ship_state['position'][0]) % 1000  # Assuming the center of the ring is at (500, 400)
    y = abs(asteroid['position'][1] - ship_state['position'][1]) % 800
    return 101 > math.sqrt(x**2 + y**2)   # Ring less than 101 pixels from ship

def normalize_ahead(asteroid: dict, frames: int):
    """
    Normalize the asteroid's position ahead in time based on its velocity.
    """
    normalized_asteroid = dict(asteroid)
    normalized_asteroid['position'] = (
        (asteroid['position'][0] + asteroid['velocity'][0] * frames / 30) % 1000,
        (asteroid['position'][1] + asteroid['velocity'][1] * frames / 30) % 800
    )
    return normalized_asteroid
def Dukem_n_Nukem(nearest_asteroids: list, ship_state: Dict,current_frame:int) -> Tuple[int, int]:
    """
    A function to determine if nuking is required or not.
    Returns 0 if no nuking is required, 1 if minor nuking is required, 2 if you have to up the ante and,
    3 to clear the field.
    
    Arguments:
        game_state (dict): Contains state information for all objects in the game.
        ship_state (dict): Contains state information for your own ship.
        current_frame (int): The current frame number in the game.

    Returns:
        Tuple: A tuple containing number of mines to drop and on what frame.
        If no nuking is required, returns (0, 0).
    """
    initial_count = len(nearest_asteroids)
    final = 0
    for f in range(0+current_frame,3*30+current_frame+30):
        for a in nearest_asteroids:
            asteroid = normalize_ahead(a[0], f)
            if in_ring(asteroid, ship_state):
                final += 1

            if final/initial_count>1/10:
                if f-30*3 > 0:
                    return(3,max(f-3*30+30,0))
                
                break
    return(0,0)      

    
class TestController(KesslerController):
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
        """
        Method processed each time step by this controller to determine what control actions to take

        Arguments:
            ship_state (dict): contains state information for your own ship
            game_state (dict): contains state information for all objects in the game

        Returns:
            float: thrust control value
            float: turn-rate control value
            bool: fire control value. Shoots if true
            bool: mine deployment control value. Lays mine if true
        """
       
       
        foresight = 800/30 # 800 pixels per second, assuming 30 FPS
        f = self.current_frame + 1 - 1 #To prevent aliasing
        time = 1000 #Time until we can shoot, in frames
        self.dead_asteroids_dict = {k: v for k, v in self.dead_asteroids_dict.items() if v > self.current_frame}  # Remove dead asteroids that are no longer relevant
        fire = drop_mine = False  # Initialize fire and drop_mine actions
        thrust = 0  # Initialize thrust action
        turn_rate = 0  # Initialize turn rate
        nearest_asteroids = will_collide(ship_state, game_state, self.current_frame)
        for asteroid in nearest_asteroids[:]:  # Iterate over a copy
            dist = distance(ship_state['position'], asteroid[0]['position'])
            asteroid[2] = dist  # Update the asteroid with its distance
            asteroid_bz = back_to_zero(asteroid[0], game_state)
            if asteroid_bz in self.dead_asteroids_dict.keys():
                nearest_asteroids.remove(asteroid)
        # Sort asteroids by distance
        nearest_asteroids.sort(key=lambda x: (x[3],x[1], x[2]))  # Sort by frames until collision, initial turning, and distance
        #print(f"Nearest asteroids: {nearest_asteroids}")
        if nearest_asteroids != []:
            self.closest_asteroid = nearest_asteroids[0][0]
            if (self.num_mines_to_drop<=0) and ((self.current_frame - self.frame_to_drop > 0) or self.current_frame == 0) and (ship_state['mines_remaining'] > 0):
                nuke_logic = Dukem_n_Nukem(nearest_asteroids, ship_state, self.current_frame)
                if nuke_logic[0] > 0:  # If nuking is required
                    self.num_mines_to_drop = nuke_logic[0]
                    self.frame_to_drop = nuke_logic[1]
                    self.mine_dropped = False
                
                    

      
            
        #nearest_asteroids = will_collide(ship_state, game_state, self.current_frame)
        #print(self.current_frame," : ",len(nearest_asteroids))
        if nearest_asteroids != []:
            closest_asteroid = nearest_asteroids[0][0]  # Get the closest asteroid by heuristic
            #print(self.current_frame)
            #print(f"Closest asteroid at {closest_asteroid['position']} with distance {nearest_asteroids[0][1]} and velocity {closest_asteroid['velocity']}")
            #pprint(game_state)
            #print(ship_state)
        else:
            closest_asteroid = {}
            #print("No asteroids in range")
        if self.current_frame in self.sequence.keys():
            self.targeted = self.sequence[self.current_frame][4]  # Check if we are currently targeting an asteroid
        if self.targeted:  # sanity checking our targeted asteroid
            if (closest_asteroid != {}) and (back_to_zero(closest_asteroid, game_state) != self.sequence[self.current_frame][5]):
                self.targeted = False
                self.sequence[self.current_frame][4] = False  # Reset the targeted flag for the sequence
                self.sequence[self.current_frame][1] = False  # Reset the fire action for the sequence

                for j in list(self.sequence.keys())[self.current_frame:]:
                    if self.sequence[j][4] and back_to_zero(closest_asteroid, game_state) == self.sequence[j][5]:
                        self.sequence[j][4] = False  # Reset the targeted flag for the sequence
                        self.sequence[j][1] = False  # Reset the fire action for the sequence
                        

        while((f<30+self.current_frame) and not (self.targeted) and (closest_asteroid != {})):
            # Calculate the future position of the closest 
            #print(f"Frame {f}: Checking asteroid at {closest_asteroid['position']} with velocity {closest_asteroid['velocity']}")
            X = ((closest_asteroid['position'][0] + closest_asteroid['velocity'][0] * (f-self.current_frame+1) / 30) % 1000)- ship_state['position'][0]
            Y = ((closest_asteroid['position'][1] + closest_asteroid['velocity'][1] * (f-self.current_frame+1) / 30) % 800)- ship_state['position'][1]

            V0 = magnitude(closest_asteroid['velocity'])  # Speed of the asteroid
            theta1 = math.degrees(math.atan2(closest_asteroid['velocity'][1], closest_asteroid['velocity'][0]))  # Angle of the asteroid's velocity 
            #theta1 = math.degrees(math.atan2(Y - ship_state['position'][1], X - ship_state['position'][0]))  # Angle to the asteroid from the ship's position
            #print(f"Frame {f}: X: {X}, Y: {Y}, V0: {V0}, theta1: {theta1}")
            solution = new_solve_motion_time(X,Y,theta1,V0/30)  # Divide by 30 to convert to pixels per frame
            #print(f"Frame {f}: Solution: {solution}")
            angle_to_target = solution['theta2_deg']  
            angle_difference = (angle_to_target - ship_state['heading']) % 360  # Calculate the angle difference
            #print(f"Angle Difference: {angle_difference} degrees for asteroid at {closest_asteroid['position']} with velocity {closest_asteroid['velocity']}")
            #time = solution['distance']/foresight 
            ttk = solution['time']  # Time to collision in frames
            #print(f"ttk: {ttk}, time: {time}")
            #print(f"Frame {f}: Angle to target: {angle_to_target}, Angle difference: {angle_difference}, Time to collision: {ttk}")

            if angle_difference >= 180:  # Normalize to [-180, 180]
                angle_difference -= 360
            #print(f"GameFrame: {self.current_frame}, WhileFrame: {f}, Angle difference: {angle_difference}, Ship heading: {ship_state['heading']}, Angle to target: {angle_to_target}")
            #print(f"TTK: {ttk}, Time: {(time + f - self.current_frame+1)=}")
            #print(f"Diff = {f - self.current_frame}")
            #(ttk >= (time + f - self.current_frame+1)) and
            
            if (abs(angle_difference)<=6*(f-self.current_frame)):
                #print(f"Frame {f}: Targeting asteroid at {closest_asteroid['position']} with angle difference {angle_difference} and time to collision {ttk}")
                #if self.current_frame not in self.sequence.keys():
                    #self.sequence[self.current_frame] = [angle_difference*30,False,False,0,True,back_to_zero(closest_asteroid, game_state)]
                    #format is [turn_rate,fire,drop_mine,thrust,targeted,position wrt 0]
                #else:
                    #self.sequence[self.current_frame][0] = angle_difference*30
                sign = math.copysign(1,angle_difference)
                for i in range(self.current_frame+1,f):
                    
                    if i not in self.sequence.keys():
                        self.sequence[i] = [sign*6*30,False,False,0,True,back_to_zero(closest_asteroid, game_state)]
                        #format is [turn_rate,fire,drop_mine,thrust,targeted,position wrt 0]
                    else:
                        self.sequence[i][0] = sign*6*30
                if f not in self.sequence.keys():
                    self.sequence[f] = [sign*(abs(angle_difference)%6)*30,False,False,0,True,back_to_zero(closest_asteroid, game_state)]
                else:
                    self.sequence[f][0] = sign*(abs(angle_difference)%6)*30
                    self.sequence[f][4] = True
                if f+1 not in self.sequence.keys():
                    self.sequence[f+1] = [0,True,False,0,False,back_to_zero(closest_asteroid, game_state)]
                else:
                    self.sequence[f+1][1] = True
                    self.sequence[f+1][4] = False
                self.dead_asteroids_dict[back_to_zero(closest_asteroid, game_state)] = (ttk+f)+1  # Mark the asteroid as dead at frame f+1
                #print("Broke")
                break
            else:
                f+=1
        if game_state['sim_frame'] == 0:
            print(ship_state['bullets_remaining'])
            #print 
        if self.current_frame in self.sequence.keys():            
            fire = self.sequence[self.current_frame][1]
        if (ship_state['bullets_remaining']>29) or (ship_state['bullets_remaining']==-1):
             #change this to improve accuracy maybe implement bisection
             if (ship_state['can_fire']):
                  if(self.current_frame+1 in list(self.sequence.keys()) and self.current_frame+2 in list(self.sequence.keys()) and not(self.sequence[self.current_frame+1][1] or self.sequence[self.current_frame+2][1])):
                    fire = True
        #else:
            #fire = self.sequence[self.current_frame][1] if self.current_frame in self.sequence.keys() else False
        # Check if we have a sequence of actions for the current frame
        team = ship_state['team']
        if self.current_frame in self.sequence.keys():
            turn_rate = self.sequence[self.current_frame][0]
            thrust = self.sequence[self.current_frame][3]
            if (self.dropped_mine_cuz_scared):
                drop_mine = True
            if (self.num_mines_to_drop>0 and self.current_frame == self.frame_to_drop) or (ship_state['is_respawning']):
                if(ship_state['is_respawning']):
                    self.dropped_mine_cuz_scared = True
                drop_mine = True
                thrust = 10
                self.num_mines_to_drop -= 1
            
            self.targeted = self.sequence[self.current_frame][4]
        
        #s = [s for s in game_state['ships'] if s['team'] != team]
        #if (s) and (ship_state['lives_remaining']<s['lives_remaining']):
            #thrust = -9999999999999999999  # If the enemy has more lives, run away     
        self.current_frame += 1

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        """
        Simple property used for naming controllers such that it can be displayed in the graphics engine

        Returns:
            str: name of this controller
        """
        return "Test Controller"

    # @property
    # def custom_sprite_path(self) -> str:
    #     return "Neo.png"
