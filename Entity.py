import pygame as pg
import numpy as np
from typing import Tuple

FOV = 90

class EntityRectangle():
    def __init__(self, center: Tuple[float, float, float], size: Tuple[float, float, float], angle: Tuple[float, float, float]) -> None:
        super().__init__()

        self.size = np.array([size[0], size[1], size[2]])
        self.half_size = self.size/2
        self.center = np.array([center[0], center[1], center[2]])
        
        self.edge_width = 2
        self.colour = (255, 0, 0)

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

        # sh == screen half
        sh_x, sh_y = surf.get_size()[0]/2, surf.get_size()[1]/2
        # Calculate the offset of the eye based on the desired vertical field of view
        eye_offset = sh_y/(np.tan((np.pi/180) * (FOV/2)))

        dots = self._calculateDots()
        
        # Add the offset to the z axis
        dots[:, 2] += eye_offset

        # tan and arctan cancel out
        dots[:, 1] = eye_offset * (dots[:, 1]/dots[:, 2])
        dots[:, 0] = eye_offset * (dots[:, 0]/dots[:, 2])

        # Shift origin to pygame position
        dots[:, 1] += sh_y
        dots[:, 0] += sh_x
        
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




        
