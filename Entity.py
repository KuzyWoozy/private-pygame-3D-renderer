import pygame as pg
import numpy as np
from Environment3D import SCREEN_SIZE
from abc import ABC, abstractmethod
from typing import Tuple
from Config import *



class Entity(ABC):
    def __init__(self, colour):
        global FOV, SCREEN_SIZE

        self.dots = []

        self.colour = colour
        # Calculate the offset of the eye based on the desired vertical field of view
        self.eyeOffset = (SCREEN_SIZE[1]/2)/(np.tan((np.pi/180) * (FOV/2)))

        

    def move(self, x_dist: float, y_dist: float, z_dist: float) -> None:
        self.dots += np.array([x_dist, y_dist, z_dist])
    
    @abstractmethod
    def rotate(self, x_rot: float, y_rot: float, z_rot: float) -> None:
        pass

    @abstractmethod
    def blit(self, surf: pg.surface.Surface) -> None:
        pass


class Rectangle(Entity):
    def __init__(self, center: Tuple[float, float, float], size: Tuple[float, float, float], angle: Tuple[float, float, float]) -> None:
        super().__init__((255, 0 ,0))

        self.size = np.array([size[0], size[1], size[2]])
        
        self.edge_width = 2

        self.dots = np.array([
                            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
                            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                            [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
                            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                            [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
                            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                            ], dtype=np.float64)
        
        self.dots = np.array([0, 0, self.eyeOffset]) + center + ((self.size/2) * self.dots)

        self.dots_t = self.dots.transpose((0, 2, 1))

        self.rotate(angle[0], angle[1], angle[2])

    def _mergeDotVectors(self) -> np.array:
        return np.sum(self.dots, axis=-2)

    # angle == [x_rot, y_rot, z_rot]
    def rotate(self, x_rot: float, y_rot: float, z_rot: float) -> None:
        

        rot_x = np.array([
                        [1, 0, 0],
                        [0, np.cos(x_rot), -np.sin(x_rot)],
                        [0, np.sin(x_rot), np.cos(x_rot)]])
        rot_y = np.array([
                        [np.cos(y_rot), 0, np.sin(y_rot)],
                        [0, 1, 0],
                        [-np.sin(y_rot), 0, np.cos(y_rot)]])
        rot_z = np.array([
                        [np.cos(z_rot), -np.sin(z_rot), 0],
                        [np.sin(z_rot), np.cos(z_rot), 0],
                        [0, 0, 1]])

        # No need to update dots, as dots_t is a view
        np.matmul((rot_z @ rot_y @ rot_x), self.dots_t, out=self.dots_t)


    def blit(self, surf: pg.surface.Surface) -> None:
        global SCREEN_SIZE

        dots = self._mergeDotVectors()

        # tan and arctan cancel out
        dots[:, 1] = self.eyeOffset * (dots[:, 1]/dots[:, 2])
        dots[:, 0] = self.eyeOffset * (dots[:, 0]/dots[:, 2])

        # Shift origin to pygame position
        dots[:, 1] += SCREEN_SIZE[1]/2
        dots[:, 0] += SCREEN_SIZE[0]/2
        
        # Get rid of z-axis as we have finished projecting points onto screen
        dots = np.delete(dots, 2, -1)

        edges = np.array([
                [dots[0], dots[1], dots[2], dots[3]],
                [dots[2], dots[3], dots[4], dots[5]],
                [dots[4], dots[5], dots[6], dots[7]],
                [dots[6], dots[7], dots[0], dots[1]],
                [dots[0], dots[3], dots[4], dots[7]],
                [dots[1], dots[2], dots[5], dots[6]]
                ])

        for edge in edges:
            pg.draw.polygon(surf, self.colour, edge)

        for edge in edges:
            for p1, p2 in zip(edge, edge[1:]):
                pg.draw.line(surf, (0,0,0), p1, p2, width=self.edge_width)
            pg.draw.line(surf, (0,0,0), edge[0], edge[-1], width=self.edge_width)


        
