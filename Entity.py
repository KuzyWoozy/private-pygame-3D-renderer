import pygame as pg
import numpy as np
from typing import Tuple

FOV = 20


class EntityRectangle():
    def __init__(self, center: Tuple[float, float, float], size: Tuple[float, float, float], angle: Tuple[float, float, float]) -> None:
        super().__init__()

        self.size = np.array([size[0], size[1], size[2]])
        self.half_size = self.size/2
        self.center = np.array([center[0], center[1], center[2]])
        


        self.mask = np.array([
                            [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
                            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                            [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
                            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                            [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
                            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                            ], dtype=np.float64)
        
        self.mask_t = self.mask.transpose((0, 2, 1))

        self.rotate(angle[0], angle[1], angle[2])

    def _calculateDots(self) -> np.ndarray:
        # Draw dots
        return self.center + (self.half_size * np.sum(self.mask, axis=-2))
    
    def move(self, x_dist: float, y_dist: float, z_dist: float) -> None:
        self.center[0] += x_dist
        self.center[1] += y_dist
        self.center[2] += z_dist

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

        np.matmul((rot_z @ rot_y @ rot_x), self.mask_t, out=self.mask_t)

    def blit(self, surf: pg.surface.Surface) -> None:
        global FOV

        s_x, s_y = surf.get_size()[0]/2, surf.get_size()[1]/2
        # Calculate the offset of the eye based on the desired vertical field of view
        eye_offset = (s_y/2)/np.tan(FOV/2)


        dots = self._calculateDots()
        
        new_dots = []
        for (x, y, z) in dots:
            
            z = z + eye_offset
            
            y_angle = np.arctan(y/z)
            new_y = eye_offset * np.tan(y_angle) 

            x_angle = np.arctan(x/z)
            new_x = eye_offset * np.tan(x_angle)
            
            new_dots.append([new_x+s_x, new_y+s_y])
        
        edges = [[new_dots[0], new_dots[1], new_dots[2], new_dots[3]],
            [new_dots[2], new_dots[3], new_dots[4], new_dots[5]],
            [new_dots[0], new_dots[3], new_dots[4], new_dots[7]],
            [new_dots[1], new_dots[2], new_dots[5], new_dots[6]],
            [new_dots[0], new_dots[1], new_dots[6], new_dots[7]],
            [new_dots[4], new_dots[5], new_dots[6], new_dots[7]]]

        for edge in edges:
            pg.draw.polygon(surf, (255,0,0), edge)
        for edge in edges:
            
            for p1, p2 in zip(edge, edge[1:]):
                pg.draw.line(surf, (0,0,0), p1, p2, width=2)
            pg.draw.line(surf, (0,0,0), edge[0], edge[-1], width=2)





        
