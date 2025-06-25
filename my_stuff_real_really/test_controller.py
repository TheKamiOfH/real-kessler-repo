# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from kesslergame import KesslerController
from typing import Dict, Tuple
from pprint import pprint
import math

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points.

    Arguments:
        p1 (tuple): First point as (x, y).
        p2 (tuple): Second point as (x, y).

    Returns:
        float: The distance between the two points.
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
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
    asteroid = dict(closest_asteroid)
    asteroid['position'] = (closest_asteroid['position'][0] - (closest_asteroid['velocity'][0] * game_state['time']) % 1000,
                                        closest_asteroid['position'][1] - (closest_asteroid['velocity'][1] * game_state['time']) % 800)
    return asteroid
class TestController(KesslerController):
    def __init__(self):
        """
        Any variables or initialization desired for the controller can be set up here
        """
        ...
        self.dead_asteroids_list = []
        self.shoot_next_frame = False

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
        nearest_asteroids = []
        for asteroid in game_state['asteroids']:
            # Calculate distance to each asteroid
            dist = distance(ship_state['position'], asteroid['position'])
            asteroid_bz = back_to_zero(asteroid, game_state)
            if asteroid_bz not in self.dead_asteroids_list:
                nearest_asteroids.append((dist, asteroid))
            
        thrust = 0
        turn_rate = -90
        fire = True
        drop_mine = False
        foresight = 800/30 # 800 pixels per second, assuming 30 FPS

        
        # Sort asteroids by distance
        nearest_asteroids.sort(key=lambda x: x[0])
        # Print the nearest asteroid's position and distance
        if nearest_asteroids:
            closest_asteroid = nearest_asteroids[0][1]
            #print(f"Closest asteroid at {closest_asteroid['position']} with distance {nearest_asteroids[0][0]} and velocity {closest_asteroid['velocity']}")
            #pprint(game_state)
            #print(ship_state)
        t = 0
        f = 0
        if nearest_asteroids != []:
            t = nearest_asteroids[0][0] / foresight  # Time to reach the asteroid at current speed 
            
       
        while((t>1 and f<25) ):
            # Calculate the future position of the closest asteroid
        
            future_x = (closest_asteroid['position'][0] + (closest_asteroid['velocity'][0]/30)  * (f))%1000
            future_y = (closest_asteroid['position'][1] + (closest_asteroid['velocity'][1]/30)  * (f))%800

            future_dist = distance(ship_state['position'], (future_x, future_y))
            t = max(((future_dist / foresight)-f),1.1)  # Time to reach the asteroid at current speed
            f+=1
        #print((future_x, future_y))
        #Vb = 800.0 # Bullet speed in pixels per second
        #Va = magnitude(closest_asteroid['velocity']) # Asteroid speed in pixels per second
       #theta = math.atan2(Va, Vb)  # Angle between the asteroid's velocity and the bullet's velocity
        #theta = 180 - math.degrees(theta)  # Convert to degrees and adjust for game coordinate system
        #alpha = math.degrees(math.atan2((Va*(math.sin(math.radians(theta)))), (Vb - Va*(math.cos(math.radians(theta))))))  # Angle to aim at
        #print(f"Angle to aim at: {alpha} degrees")
        future_frames = f #Predicting how many frames it will take to reach the asteroid 
        print(f"Future frames: {future_frames}")   
        if nearest_asteroids != []:    
            asteroid_x = (nearest_asteroids[0][1]['position'][0]+(nearest_asteroids[0][1]['velocity'][0]*future_frames*game_state['delta_time']))%1000
            asteroid_y = (nearest_asteroids[0][1]['position'][1] + (nearest_asteroids[0][1]['velocity'][1]*future_frames*game_state['delta_time']))%800
            ship_x = ship_state['position'][0]
            ship_y = ship_state['position'][1]

            # Calculate the angle to the nearest asteroid
            angle_to_asteroid = math.degrees(math.atan2(asteroid_y - ship_y, asteroid_x - ship_x))
            # Calculate the angle difference
            angle_difference = (angle_to_asteroid - ship_state['heading']) % 360 
            if angle_difference > 180:  # Normalize to [-180, 180]
                angle_difference -= 360
            
            # Max turn rate is 180 degrees per second
            # Set turn rate based on angle difference
            turn_rate = angle_difference*30 # Assuming 30 FPS, adjust as needed
        if self.shoot_next_frame:
            fire = True
            self.shoot_next_frame = False

        if abs(turn_rate) >= 180.0:
            fire = False
        else:
            self.shoot_next_frame = True
            if nearest_asteroids != []:
                self.dead_asteroids_list.append(back_to_zero(closest_asteroid, game_state))
       
        

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
