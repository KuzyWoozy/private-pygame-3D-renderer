import pygame as pg
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from Config import *

# Convert to radians
FOV_VERT *= np.pi/180
FOV_HOR *= np.pi/180

FOV_VERT_HALF = FOV_VERT/2
FOV_HOR_HALF = FOV_HOR/2


class Entity(ABC):
    def __init__(self, colour: Tuple[int, int, int]) -> None:
        global FOV_HALF, SCREEN_SIZE

        self.dots: np.ndarray = np.empty(0)

        self.colour: Tuple[int, int, int] = colour
        # Calculate the offset of the eye based on the desired vertical field of view
        self.eyeOffset: float = (SCREEN_SIZE[0]/2)/(np.tan(FOV_HOR_HALF))

        
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

        self.size: np.ndarray = np.array([size[0], size[1], size[2]])
        
        self.edge_width: int = 2

        self.dots: np.ndarray = np.array([
                            [-1, 1, -1],
                            [1, 1, -1],
                            [1, 1, 1],
                            [-1, 1, 1],
                            [-1, -1, -1],
                            [1, -1, -1],
                            [1, -1, 1],
                            [-1, -1, 1]
                            ], dtype=np.float64)
        
        self.dots = center + ((self.size/2) * self.dots)

        self.dots_t: np.ndarray = self.dots.transpose((1, 0))

        self.rotate(angle[0], angle[1], angle[2])

    # angle == [x_rot, y_rot, z_rot]
    def rotate(self, x_rot: float, y_rot: float, z_rot: float) -> None:
        
        rot_x:  np.ndarray = np.array([
                        [1, 0, 0],
                        [0, np.cos(x_rot), -np.sin(x_rot)],
                        [0, np.sin(x_rot), np.cos(x_rot)]])
        rot_y: np.ndarray = np.array([
                        [np.cos(y_rot), 0, np.sin(y_rot)],
                        [0, 1, 0],
                        [-np.sin(y_rot), 0, np.cos(y_rot)]])
        rot_z: np.ndarray = np.array([
                        [np.cos(z_rot), -np.sin(z_rot), 0],
                        [np.sin(z_rot), np.cos(z_rot), 0],
                        [0, 0, 1]])

        # No need to update dots, as dots_t is a view
        np.matmul((rot_z @ rot_y @ rot_x), self.dots_t, out=self.dots_t)


    def blit(self, surf: pg.surface.Surface) -> None:
        global SCREEN_SIZE

        dots: np.ndarray = self.dots.copy()

        # Horrifc ik, but I cannot think of a better solution yet 
        if (dots[:, 2] < 0).any():
            return


        dots[:, 1] = self.eyeOffset * (dots[:, 1]/dots[:, 2])
        dots[:, 0] = self.eyeOffset * (dots[:, 0]/dots[:, 2])

        # Shift origin to pygame position
        dots[:, 1] += SCREEN_SIZE[1]/2
        dots[:, 0] += SCREEN_SIZE[0]/2
        
        dots = np.delete(dots, 2, axis=-1)

        edges: np.ndarray = np.array([
                [dots[0], dots[1], dots[2], dots[3]],
                [dots[4], dots[5], dots[6], dots[7]],
                [dots[0], dots[4], dots[5], dots[1]],
                [dots[1], dots[5], dots[6], dots[2]],
                [dots[2], dots[6], dots[7], dots[3]],
                [dots[3], dots[7], dots[4], dots[0]],
                ])

        for edge in edges:
            pg.draw.polygon(surf, self.colour, edge)

        for edge in edges:
            for p1, p2 in zip(edge, edge[1:]):
                pg.draw.line(surf, (0,0,0), p1, p2, width=self.edge_width)
            pg.draw.line(surf, (0,0,0), edge[0], edge[-1], width=self.edge_width)


        
