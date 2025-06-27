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
def will_collide(ship_state: Dict, game_state: Dict, frame_number: int) -> list:
    """
    Check if the ship will collide with any asteroid in the next frame.
    Arguments:
        ship_state (dict): Contains state information for your own ship.
        game_state (dict): Contains state information for all objects in the game.
        frame_number (int): The current frame number in the game.
    Returns:
        List: Returns a list of asteroids that will collide with the ship and the number of frames until collision.
    """
    collisions = []
    for asteroid in game_state['asteroids']:
        f = frame_number + 1 - 1# Start checking from the next frame
        asteroid_copy = dict(asteroid)  # Create a copy of the asteroid to avoid modifying the original
        while distance(ship_state['position'], asteroid_copy['position']) > 22 and f < frame_number + 25:  # Assuming ship radius is 22
            # Update asteroid position based on its velocity
            asteroid_copy['position'] = ((asteroid['position'][0] + asteroid['velocity'][0]*f / 30)%1000,  
                                    (asteroid['position'][1] + asteroid['velocity'][1]*f/ 30)%800)  # Assuming 30 FPS
            if distance(ship_state['position'], asteroid_copy['position']) < 22:  # Assuming ship radius is 22
                collisions.append([asteroid, f, 0])  # 1 frame until collision
            else:
                f+=1
        else:
            collisions.append([asteroid, f, 0])  # No collision within the next 25 frames

    return sorted(collisions, key=lambda x: x[1])  # Sort by number of frames until collision
class TestController(KesslerController):
    def __init__(self):
        """
        Any variables or initialization desired for the controller can be set up here
        """
        ...
        self.dead_asteroids_dict = {} # Dictionary to keep track of dead asteroids
        self.shoot_next_frame = False
        self.shoot_this_frame = False
        self.shot_last_frame = 0
        self.current_frame = 0

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
        if (not(self.current_frame in self.dead_asteroids_dict.keys()) and not(self.current_frame < max(self.dead_asteroids_dict.keys(), default=0))) or (self.current_frame==0):
            nearest_asteroids = will_collide(ship_state, game_state, self.current_frame)
            for asteroid in nearest_asteroids:
                # Calculate distance to each asteroid
                dist = distance(ship_state['position'], asteroid[0]['position'])
                asteroid[2] = dist  # Update the asteroid with its distance
                asteroid_bz = back_to_zero(asteroid[0], game_state)
                #if asteroid_bz not in self.dead_asteroids_list:
                    #nearest_asteroids.remove(asteroid)  # Remove the asteroid if it has already been processed
            # Sort asteroids by distance
            nearest_asteroids.sort(key=lambda x: (x[1],x[2]))
        else:
            if self.current_frame in self.dead_asteroids_dict.keys():
                self.shoot_next_frame = True
            else:
                nearest_asteroids = False
        thrust = 0
        turn_rate = 0
        fire = False
        self.shoot_this_frame = self.shoot_next_frame
        drop_mine = False
        foresight = 800/30 # 800 pixels per second, assuming 30 FPS
        f = self.current_frame + 1 - 1 #To prevent aliasing
        time = 1000 #Time until we can shoot, in frames


        try:
            nearest_asteroids = will_collide(ship_state, game_state, self.current_frame)
            closest_asteroid = nearest_asteroids[0][0]  # Get the closest asteroid by heuristic
            #print(f"Closest asteroid at {closest_asteroid['position']} with distance {nearest_asteroids[0][0]} and velocity {closest_asteroid['velocity']}")
            #pprint(game_state)
            #print(ship_state)
        
        
        


            while((f<25+self.current_frame) and (time > 1)):
                # Calculate the future position of the closest asteroid
            
                future_x = (closest_asteroid['position'][0] + (closest_asteroid['velocity'][0]/30)  * (f-self.current_frame+1))%1000
                future_y = (closest_asteroid['position'][1] + (closest_asteroid['velocity'][1]/30)  * (f-self.current_frame+1))%800
                ship_x = ship_state['position'][0]
                ship_y = ship_state['position'][1]
                dist = distance((future_x,future_y),(ship_x,ship_y))
                angle_to_target = math.degrees(math.atan2(future_y - ship_y, future_x - ship_x))
                angle_difference = (angle_to_target - ship_state['heading']) % 360
                time = (dist/foresight) - (f-self.current_frame)  # this should be less than 1 frame, otherwise we are too far away to shoot
                if angle_difference >= 180:  # Normalize to [-180, 180]
                    angle_difference -= 360

                if (abs(angle_difference) <= 6*(f-self.current_frame)) and (time <= 1):  # If we can turn in one frame and shoot in the next
                    #If we can turn in one frame, we can shoot in the next frame
                    turn_rate = angle_difference*30 # Turn rate in degrees per second, assuming 30 FPS
                    #self.shoot_next_frame = True
                    self.dead_asteroids_dict[f] = back_to_zero(closest_asteroid, game_state) # Store the asteroid in the dead asteroids dictionary
                    break
                else:
                    f+=1
        except:
            pass
            

            
        #print((future_x, future_y))
        #Vb = 800.0 # Bullet speed in pixels per second
        #Va = magnitude(closest_asteroid['velocity']) # Asteroid speed in pixels per second
       #theta = math.atan2(Va, Vb)  # Angle between the asteroid's velocity and the bullet's velocity
        #theta = 180 - math.degrees(theta)  # Convert to degrees and adjust for game coordinate system
        #alpha = math.degrees(math.atan2((Va*(math.sin(math.radians(theta)))), (Vb - Va*(math.cos(math.radians(theta))))))  # Angle to aim at
        #print(f"Angle to aim at: {alpha} degrees")
        
        if self.shoot_this_frame and not(ship_state["is_respawning"]) and ship_state["can_fire"]:
            fire = True
        else:
            fire = False
        
        #if self.shoot_next_frame:
            #fire = True # doesnt matter even if we decide to turn this frame because our shot will be from the original position so we can both turn and shoot
            #self.shoot_next_frame = False

        if abs(turn_rate) >= 180.0:
            fire = False
        else:
            self.shoot_next_frame = True
            #if nearest_asteroids != []:
                #self.dead_asteroids_list.append(back_to_zero(closest_asteroid, game_state))
                
       
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
