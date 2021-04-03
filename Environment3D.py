import pygame as pg
import numpy as np

from Config import *
from typing import Dict, Any
from Entity import Rectangle



class Environment3D:
    def __init__(self) -> None:
        global ROTATION_ANGLE, MOVEMENT_DISTANCE, SCREEN_SIZE
        """
        Keybinds:
            Rotation:
                w - Rotate the world about the x-axis in negative direction
                s - Rotate the world about the x-axis in positive direction
                d - Rotate the world about the y-axis in negative direction
                a - Rotate the world about the y-axis in positive direction
                z - Rotate the world about the z-axis in negative direction                x - Rotate the world about the z-axis in positive direction


            Movement:
                e - Move forward
                q - Move backward
        """
        self.keycodes: Dict[int, Dict[str, Any]] = dict([
                (pg.K_w, {"func": lambda : self.rotate(-np.pi/180 * ROTATION_ANGLE, 0, 0)}),
                (pg.K_s, {"func": lambda : self.rotate(np.pi/180 * ROTATION_ANGLE,0, 0)}),
                (pg.K_d, {"func": lambda : self.rotate(0, -np.pi/180 * ROTATION_ANGLE, 0)}),
                (pg.K_a, {"func": lambda : self.rotate(0, np.pi/180 * ROTATION_ANGLE, 0)}),
                (pg.K_q, {"func": lambda : self.move(0, 0, -MOVEMENT_DISTANCE)}),
                (pg.K_e, {"func": lambda : self.move(0, 0, MOVEMENT_DISTANCE)}), 
                (pg.K_z, {"func": lambda : self.rotate(0, 0, -np.pi/180 * ROTATION_ANGLE)}),
                (pg.K_x, {"func": lambda : self.rotate(0, 0, np.pi/180 * ROTATION_ANGLE)}), 
                ])

        for key in self.keycodes:
            self.keycodes[key]["flag"] = False

        pg.init()
        self.clock: pg.time.Clock = pg.time.Clock()
        self.display: pg.surface.Surface = pg.display.set_mode(SCREEN_SIZE)


        self.entities = [
                Rectangle((75,0,100), (50, 50, 50), (0,0,0)),
                Rectangle((-75,0,100), (50, 50, 50), (0,0,0)),
                Rectangle((0,75,100), (50, 50, 50), (0,0,0)),
                Rectangle((0,-75,100), (50, 50, 50), (0,0,0)),
                ]


    def rotate(self, x_rot: float, y_rot: float, z_rot: float) -> None:
        for entity in self.entities:
            entity.rotate(x_rot, y_rot, z_rot)
    
    def move(self, x_mov: float, y_mov: float, z_mov: float) -> None:
        for entity in self.entities:
            entity.move(x_mov, y_mov, z_mov)

    def _processKeys(self) -> None:
        for event in pg.event.get():
            if (event.type == pg.KEYDOWN or event.type == pg.KEYUP):
                if (event.type == pg.KEYDOWN):
                    if (event.key in self.keycodes):
                        self.keycodes[event.key]["flag"] = True
                elif (event.type == pg.KEYUP):
                    if (event.key in self.keycodes):
                        self.keycodes[event.key]["flag"] = False

        for key in self.keycodes:
            if (key in self.keycodes):
                if (self.keycodes[key]["flag"]):
                    self.keycodes[key]["func"]()

    def run(self) -> None:
        while True:
            self._processKeys()
             
            self.display.fill((255,255,255))
            for entity in self.entities:
                entity.blit(self.display)
            pg.display.update()

            self.clock.tick(60)
            




