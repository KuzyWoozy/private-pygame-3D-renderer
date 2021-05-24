import pygame as pg
import random

from Config import *
from typing import Dict, Any
from Entity import Rectangle
from Util import heapSort


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
                z - Rotate the world about the z-axis in negative direction         
                x - Rotate the world about the z-axis in positive direction


            Movement:
                e - Move forward
                q - Move backward
        """
        self.keycodes: Dict[int, Dict[str, Any]] = dict([
                (pg.K_w, {"func": lambda delta : self.rotate(-ROTATION_ANGLE * delta, 0, 0)}),
                (pg.K_s, {"func": lambda delta : self.rotate(ROTATION_ANGLE * delta,0, 0)}),
                (pg.K_d, {"func": lambda delta : self.rotate(0, -ROTATION_ANGLE * delta, 0)}),
                (pg.K_a, {"func": lambda delta : self.rotate(0, ROTATION_ANGLE * delta, 0)}),
                (pg.K_q, {"func": lambda delta : self.move(0, 0, -MOVEMENT_DISTANCE * delta)}),
                (pg.K_e, {"func": lambda delta : self.move(0, 0, MOVEMENT_DISTANCE * delta)}), 
                (pg.K_z, {"func": lambda delta : self.rotate(0, 0, -ROTATION_ANGLE * delta)}),
                (pg.K_x, {"func": lambda delta : self.rotate(0, 0, ROTATION_ANGLE * delta)}), 
                ])

        for key in self.keycodes:
            self.keycodes[key]["flag"] = False

        pg.init()
        self.clock: pg.time.Clock = pg.time.Clock()
        self.display: pg.surface.Surface = pg.display.set_mode(SCREEN_SIZE)


        self.entities = [Rectangle((0,0,200), (20, 20, 20), (0,0,0), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), (random.uniform(-30, 30), random.uniform(-30, 30), random.uniform(-30, 30))) for _ in range(0, 10)]


    def rotate(self, x_rot: float, y_rot: float, z_rot: float) -> None:
        for entity in self.entities:
            entity.rotate(x_rot, y_rot, z_rot)
    
    def move(self, x_mov: float, y_mov: float, z_mov: float) -> None:
        for entity in self.entities:
            entity.move(x_mov, y_mov, z_mov)

    def _processKeys(self, delta: float) -> None:
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
                    self.keycodes[key]["func"](delta)

    def run(self) -> None:
        delta = 0
        while True:
            self._processKeys(delta)
             
            self.display.fill((255,255,255))
           
            for entity in self.entities:
                vel = entity.velocity * delta
                entity.move(vel[0], vel[1], vel[2])

            sorted_entities = heapSort(self.entities)
            for entity in sorted_entities:
                entity.blit(self.display)
            
            pg.display.update()

            delta = self.clock.tick(60) / 1000
            




