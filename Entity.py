import pygame as pg
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Callable
from Config import *

# Convert to radians
FOV_VERT *= np.pi/180
FOV_HOR *= np.pi/180

FOV_VERT_HALF: float = FOV_VERT/2
FOV_HOR_HALF: float = FOV_HOR/2

# For debugging purposes
CULL_OFFSET: float = 0



class Entity(ABC):
    def __init__(self, colour: Tuple[int, int, int]) -> None:
        global FOV_HALF, SCREEN_SIZE

        self.dots: np.ndarray = None

        self.colour: Tuple[int, int, int] = colour
        # Calculate the offset of the eye based on the desired vertical field of view
        self.eyeOffset: float = (SCREEN_SIZE[0]/2)/(np.tan(FOV_HOR_HALF))

        
    def move(self, x_dist: float, y_dist: float, z_dist: float) -> None:
        self.dots += np.array([x_dist, y_dist, z_dist])
    
    @staticmethod
    def cull(planes: List[np.ndarray]) -> List[np.ndarray]:
        global CULL_OFFSET
        

        def planeIntersect(edges: np.ndarray, mask_funcs: List[Callable[[np.ndarray], np.ndarray]], hyperplane: np.ndarray) -> None:
            closing_edge = [[], []]
            for point, mask_func in enumerate(mask_funcs):
                mask: np.ndarray = mask_func(edges)
                if (edges[mask].size > 0):
                    P = edges[mask, 0]
                    Q = edges[mask, 1]
                    PQ_diff = P-Q
                    
                    denominator = PQ_diff.dot(hyperplane)
                    t = np.empty(PQ_diff.shape[0])
                    t[np.abs(denominator) < 1e-2] = 1
                    t[np.abs(denominator) >= 1e-2] = np.array(([0, 0, CULL_OFFSET]-Q[np.abs(denominator) >= 1e-2]).dot(hyperplane)/denominator[np.abs(denominator) >= 1e-2])
                    projected_points = ((PQ_diff * np.expand_dims(t, 0).transpose()) + Q)
                    edges[mask, point] = projected_points
                    

            return np.append(edges, np.array([closing_edge]), axis=0)

        # Gets rid of insignificant edges 
        culled_planes = []
        for edges in planes:
            edges = edges[np.logical_or(edges[:, 0, 2] > CULL_OFFSET, edges[:, 1, 2] > CULL_OFFSET)]
            if edges.size == 0:
                continue

            # z plane
            planeIntersect(edges, [lambda x: x[:, 0, 2] < CULL_OFFSET,
         lambda x: x[:, 1, 2] < CULL_OFFSET], np.array([0, 0, 1]))
            
            # zero divide value guard
            edges[edges[:, :, 2] < 1e-2, 2] = 1e-2

                        
            # borders
            # right
            edges = edges[np.logical_or(np.arctan(edges[:, 0, 0] / edges[:, 0, 2]) <= FOV_HOR_HALF, np.arctan(edges[:, 1, 0] / edges[:, 1, 2]) <= FOV_HOR_HALF)]
            edges = planeIntersect(edges, [lambda x: np.arctan(x[:, 0, 0] / x[:, 0, 2]) > FOV_HOR_HALF, lambda x: np.arctan(x[:, 1, 0] / x[:, 1, 2]) > FOV_HOR_HALF], np.array([np.sin(FOV_HOR_HALF-np.pi/2), 0, np.cos(FOV_HOR_HALF-np.pi/2)]))
            # left
            edges = edges[np.logical_or(np.arctan(edges[:, 0, 0] / edges[:, 0, 2]) >= -FOV_HOR_HALF, np.arctan(edges[:, 1, 0] / edges[:, 1, 2]) >= -FOV_HOR_HALF)]
            edges = planeIntersect(edges, [lambda x: np.arctan(x[:, 0, 0] / x[:, 0, 2]) < -FOV_HOR_HALF, lambda x: np.arctan(x[:, 1, 0] / x[:, 1, 2]) < -FOV_HOR_HALF], np.array([np.sin(-FOV_HOR_HALF+np.pi/2), 0, np.cos(-FOV_HOR_HALF+np.pi/2)]))
            # up
            edges = edges[np.logical_or(np.arctan(edges[:, 0, 1] / edges[:, 0, 2]) >= -FOV_VERT_HALF, np.arctan(edges[:, 1, 1] / edges[:, 1, 2]) >= -FOV_VERT_HALF)]
            edges = planeIntersect(edges, [lambda x: np.arctan(x[:, 0, 1] / x[:, 0, 2]) < -FOV_VERT_HALF, lambda x: np.arctan(x[:, 1, 1] / x[:, 1, 2]) < -FOV_VERT_HALF], np.array([0, np.sin(-FOV_VERT_HALF+np.pi/2), np.cos(-FOV_VERT_HALF+np.pi/2)]))
            # down
            edges = edges[np.logical_or(np.arctan(edges[:, 0, 1] / edges[:, 0, 2]) <= FOV_VERT_HALF, np.arctan(edges[:, 1, 1] / edges[:, 1, 2]) <= FOV_VERT_HALF)]
            edges = planeIntersect(edges, [lambda x: np.arctan(x[:, 0, 1] / x[:, 0, 2]) > FOV_VERT_HALF, lambda x: np.arctan(x[:, 1, 1] / x[:, 1, 2]) > FOV_VERT_HALF], np.array([0, np.sin(FOV_VERT_HALF-np.pi/2), np.cos(FOV_VERT_HALF-np.pi/2)]))
           
            if edges.size > 1:
                culled_planes.append(edges)
            
        return culled_planes

                         
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
                            [-1, -1, -1],
                            [1, -1, -1],
                            [1, -1, 1],
                            [-1, -1, 1],
                            [-1, 1, -1],
                            [1, 1, -1],
                            [1, 1, 1],
                            [-1, 1, 1]
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
        # Stop trying to draw it if it cannot be rendered
        if ((dots[:, 2] < 0).all()):
            return

        # Culling
        dots[dots[:, 2] < 1e-2, 2] = 1e-2
        if not np.logical_and((np.abs(np.arctan(dots[:, 0] / dots[:, 2]) < FOV_HOR_HALF).any()), (np.abs(np.arctan(dots[:, 1] / dots[:, 2])) < FOV_VERT_HALF).any()):
            return

    
        planes: List[np.ndarray] = [
                                    np.array([[dots[0], dots[1]], [dots[1], dots[5]], [dots[5], dots[4]], [dots[4], dots[0]]]),

                                    #np.array([[dots[1], dots[2]], [dots[2], dots[6]], [dots[6], dots[5]], [dots[5], dots[1]]]),
                                    #np.array([[dots[2], dots[3]], [dots[3], dots[7]], [dots[7], dots[6]], [dots[6], dots[2]]]),
                                    #np.array([[dots[3], dots[7]], [dots[7], dots[4]], [dots[4], dots[0]], [dots[0], dots[3]]]),
                                    #np.array([[dots[0], dots[1]], [dots[1], dots[2]], [dots[2], dots[3]], [dots[3], dots[0]]]),
                                    #np.array([[dots[4], dots[5]], [dots[5], dots[6]], [dots[6], dots[7]], [dots[7], dots[4]]])
                                   ]

        planes = Rectangle.cull(planes)

        new_planes = []
        for plane in planes:
            plane[plane[:, :, 2] < 1e-2, 2] = 1e-2
            plane[:, :, 1] = self.eyeOffset * (plane[:, :, 1]/plane[:, :, 2])
            plane[:, :, 0] = self.eyeOffset * (plane[:, :, 0]/plane[:, :, 2])

            # Shift origin to pygame position
            plane[:, :, 1] += SCREEN_SIZE[1]/2
            plane[:, :, 0] += SCREEN_SIZE[0]/2
            
            new_planes.append(np.delete(plane, 2, axis=-1))
        planes = new_planes
         
        for plane in planes:
            if plane.shape[0] >= 2:
                pg.draw.polygon(surf, self.colour, np.concatenate(plane))
        
        for plane in planes:
            for (p1, p2) in plane:
                pg.draw.line(surf, (0,0,0), p1, p2, width=self.edge_width)
        
        
