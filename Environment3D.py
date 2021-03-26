import pygame as pg
import numpy as np
from typing import Dict, List, Any
from Entity import EntityRectangle

SCREEN_SIZE = (800, 600)
ROTATION_ANGLE = 2
MOVEMENT_DISTANCE = 10

class Environment3D:
    global ROTATION_ANGLE
    def __init__(self) -> None:
        self.keycodes: Dict[int, List[Any]] = dict([
                (pg.K_w, [lambda : self._rotate(-np.pi/180 * ROTATION_ANGLE, 0, 0)]),
                (pg.K_s, [lambda : self._rotate(np.pi/180 * ROTATION_ANGLE,0, 0)]),
                (pg.K_d, [lambda : self._rotate(0, -np.pi/180 * ROTATION_ANGLE, 0)]),
                (pg.K_a, [lambda : self._rotate(0, np.pi/180 * ROTATION_ANGLE, 0)]),
                (pg.K_UP, [lambda : self._move(0, -MOVEMENT_DISTANCE, 0)]),
                (pg.K_DOWN, [lambda : self._move(0, MOVEMENT_DISTANCE, 0)]), 
                (pg.K_RIGHT, [lambda : self._move(MOVEMENT_DISTANCE, 0, 0)]),
                (pg.K_LEFT, [lambda : self._move(-MOVEMENT_DISTANCE, 0, 0)]),
                (pg.K_q, [lambda : self._move(0, 0, MOVEMENT_DISTANCE)]),
                (pg.K_e, [lambda : self._move(0, 0, -MOVEMENT_DISTANCE)]), 


                ])
        for key in self.keycodes:
            self.keycodes[key].append(False)

        pg.init()
        self.clock: pg.time.Clock = pg.time.Clock()
        self.display: pg.surface.Surface = pg.display.set_mode(SCREEN_SIZE)

        self.entity = EntityRectangle((0,0,100), (100, 100, 100), (0,0,0))

    def _rotate(self, x_rot: float, y_rot: float, z_rot: float) -> None:
        self.entity.rotate(x_rot, y_rot, z_rot)

    def _move(self, x_mov: float, y_mov: float, z_mov: float) -> None:
        self.entity.move(x_mov, y_mov, z_mov)

    def _processKeys(self) -> None:
       
        for event in pg.event.get():
            if (event.type == pg.KEYDOWN or event.type == pg.KEYUP):
                if (event.type == pg.KEYDOWN):
                    if (event.key in self.keycodes):
                        self.keycodes[event.key][1] = True
                elif (event.type == pg.KEYUP):
                    if (event.key in self.keycodes):
                        self.keycodes[event.key][1] = False

        for key in self.keycodes:
            if (key in self.keycodes):
                if (self.keycodes[key][1]):
                    self.keycodes[key][0]()

    def run(self) -> None:
        while True:
            self._processKeys()
             
            self.display.fill((255,255,255))
            self.entity.blit(self.display)
            pg.display.update()

            self.clock.tick(60)
            




