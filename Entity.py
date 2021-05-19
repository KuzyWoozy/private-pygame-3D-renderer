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
        self.dots_t: np.ndarray = None

        self.colour: Tuple[int, int, int] = colour
        # Calculate the offset of the eye based on the desired vertical field of view
        self.eyeOffset: float = (SCREEN_SIZE[0]/2)/(np.tan(FOV_HOR_HALF))

        
    def move(self, x_dist: float, y_dist: float, z_dist: float) -> None:
        self.dots += np.array([x_dist, y_dist, z_dist])
    """
    @staticmethod
    def polyAssemble(edges: List[np.ndarray]) -> List[np.ndarray]:
        if len(edges) == 0:
            return edges
        new_edges = []
        for poly in edges:
            if poly.shape[0] == 0:
                continue
            new_poly = np.zeros(poly.shape)
            edge1 = poly[0]
            new_poly[0] = edge1
            count = 1
            loop = True
            while(loop):
                for edge2 in poly:
                    if not (np.abs(edge1 - edge2).sum() < 1e-2):

                        if (np.abs(edge1[1] - edge2[0]) < 1e-2).all():
                            # check if we made a cycle
                            if (np.abs(edge2-poly[0]) < 1e-2).all():
                                loop = False
                                break
                            new_poly[count] = edge2
                            count += 1
                            edge1 = edge2
                            break

                        if (np.abs(edge1[1] - edge2[1]) < 1e-2).all():
# check if we made a cycle
                            edge2 = edge2[::-1]
                            if (np.abs(edge2-poly[0]) < 1e-2).all():
                                loop = False
                                break
                            new_poly[count] = edge2
                            count += 1
                            edge1 = edge2
                            break
            new_edges.append(new_poly)
        return new_edges
    """
    @staticmethod
    def cull(edges: List[np.ndarray]) -> List[np.ndarray]:
        global CULL_OFFSET
        

        def planeIntersect(poly: np.ndarray, mask_func: Callable[[np.ndarray, int], np.ndarray], hyperplane: np.ndarray) -> np.ndarray:
            masks: np.ndarray = np.array([mask_func(poly, 0), mask_func(poly, 1)])
            if ((~masks).all()):
                return poly

            seal = np.empty((2,3))
            seal.fill(np.nan)
            for i, mask in enumerate(masks):
                if (~mask).all():
                    continue
                P = poly[mask, 0]
                Q = poly[mask, 1]
                PQ_diff = P-Q
                    
                denominator = PQ_diff.dot(hyperplane)
                t = np.empty(PQ_diff.shape[0])
                t[np.abs(denominator) < 1e-2] = 1
                t[np.abs(denominator) >= 1e-2] = np.array(([0, 0, CULL_OFFSET]-Q[np.abs(denominator) >= 1e-2]).dot(hyperplane)/denominator[np.abs(denominator) >= 1e-2])

                projected_points = (PQ_diff * np.expand_dims(t, 0).transpose()) + Q
                poly[mask, i] = projected_points

                seal[i] = projected_points 

            if not np.isnan(np.sum(seal)):
                p = np.argwhere(masks[0] | masks[1])
                if poly.shape[0] > 2:
                    if (np.abs(p[0] - p[1])) == 1:
                        z = p[-1]
                    else:
                        z = p[0]

                    if (np.abs(seal[1] - poly[z, 0]) < 1e-2).all():
                        poly = np.insert(poly, z, seal, axis=0)
                    else:
                        poly = np.insert(poly, z, seal[::-1], axis=0)
                else:
                    if (np.abs(seal[1] - poly[1, 0]) < 1e-2).all():
                        poly = np.insert(poly, 1, seal, axis=0)
                    elif (np.abs(seal[0] - poly[1, 0]) < 1e-2).all():
                        poly = np.insert(poly, 1, seal[::-1], axis=0)
                    elif (np.abs(seal[1] - poly[0, 0]) < 1e-2).all():
                        poly = np.insert(poly, 0, seal, axis=0)
                    else:
                        poly = np.insert(poly, 0, seal[::-1], axis=0)
   
                                    
                
                
                
            return poly
            
        
        modified_edges = []
        for poly in edges:
            # z plane
            poly = poly[(poly[:, :, 2] > CULL_OFFSET).any(axis=-1)]
            poly = planeIntersect(poly, lambda x, i: (x[:, i, 2] < CULL_OFFSET), np.array([0, 0, 1]))
            
            # zero divide value guard
            poly[poly[:, :, 2] < 1e-2, 2] = 1e-2
                   
            # borders
            # right
            poly = poly[(np.arctan(poly[:, :, 0] / poly[:, :, 2]) <= FOV_HOR_HALF).any(axis=-1)]
            poly = planeIntersect(poly, lambda x, i: (np.arctan(x[:, i, 0] / x[:, i, 2]) > FOV_HOR_HALF), np.array([np.sin(FOV_HOR_HALF-np.pi/2), 0, np.cos(FOV_HOR_HALF-np.pi/2)]))
        
            # zero divide value guard
            poly[poly[:, :, 2] < 1e-2, 2] = 1e-2

        
            # left
            poly = poly[(np.arctan(poly[:, :, 0] / poly[:, :, 2]) >= -FOV_HOR_HALF).any(axis=-1)]
            poly = planeIntersect(poly, lambda x, i: (np.arctan(x[:, i, 0] / x[:, i, 2]) < -FOV_HOR_HALF), np.array([np.sin(-FOV_HOR_HALF+np.pi/2), 0, np.cos(-FOV_HOR_HALF+np.pi/2)]))
        
            # zero divide value guard
            poly[poly[:, :, 2] < 1e-2, 2] = 1e-2

            # up
            poly = poly[(np.arctan(poly[:, :, 1] / poly[:, :, 2]) >= -FOV_VERT_HALF).any(axis=-1)]
            poly = planeIntersect(poly, lambda x, i: (np.arctan(x[:, i, 1] / x[:, i, 2]) < -FOV_VERT_HALF), np.array([0, np.sin(-FOV_VERT_HALF+np.pi/2), np.cos(-FOV_VERT_HALF+np.pi/2)]))

            # zero divide value guard
            poly[poly[:, :, 2] < 1e-2, 2] = 1e-2

            # down
            poly = poly[(np.arctan(poly[:, :, 1] / poly[:, :, 2]) <= FOV_VERT_HALF).any(axis=-1)]
            poly = planeIntersect(poly, lambda x, i: (np.arctan(x[:, i, 1] / x[:, i, 2]) > FOV_VERT_HALF), np.array([0, np.sin(FOV_VERT_HALF-np.pi/2), np.cos(FOV_VERT_HALF-np.pi/2)]))

            # zero divide value guard
            poly[poly[:, :, 2] < 1e-2, 2] = 1e-2
             
            modified_edges.append(poly)
        
        return modified_edges
        
           
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
       
        # Culling
        dots_mask = self.dots[:, 2] > 1e-2 
        
        if (~dots_mask).all():
            return
        
        edges = Entity.cull(self.generateEdges())
        
        # Ugly ik, but there isn't a clean way to handle numpy arrays of arbitary dimensions
        modified_edges = []
        for poly in edges:
            poly[poly[:, :, 2] < 1e-2, 2] = 1e-2
            poly[:, :, 1] = self.eyeOffset * (poly[:, :, 1]/poly[:, :, 2])
            poly[:, :, 0] = self.eyeOffset * (poly[:, :, 0]/poly[:, :, 2])

            # Shift origin to pygame position
            poly[:, :, 1] += SCREEN_SIZE[1]/2
            poly[:, :, 0] += SCREEN_SIZE[0]/2
            
            poly = np.delete(poly, 2, axis=-1)
            modified_edges.append(poly)
        edges = modified_edges

        for poly in edges:
            if poly.shape[0] >= 2:
                pg.draw.polygon(surf, self.colour, np.concatenate(poly))
        
        for poly in edges:
            for (p1, p2) in poly:
                #pg.draw.circle(surf, (255, 0, 0), p1, radius=20)
                #pg.draw.circle(surf, (255, 0, 0), p2, radius=20)
                pg.draw.line(surf, (0,0,0), p1, p2, width=self.edge_width)

    @abstractmethod
    def generateEdges(self) -> np.ndarray:
        pass



class Rectangle(Entity):
    def __init__(self, center: Tuple[float, float, float], size: Tuple[float, float, float], angle: Tuple[float, float, float]) -> None:
        super().__init__((255, 0 ,0))

        self.size: np.ndarray = np.array([size[0], size[1], size[2]])
        
        dots_mask: np.ndarray = np.array([
                            [-1, -1, -1],
                            [1, -1, -1],
                            [1, -1, 1],
                            [-1, -1, 1],
                            [-1, 1, -1],
                            [1, 1, -1],
                            [1, 1, 1],
                            [-1, 1, 1]
                            ], dtype=np.float64)
        
        self.dots = center + ((self.size/2) * dots_mask)
        self.dots_t = self.dots.transpose((1, 0))

        self.edge_width: int = 2

        self.rotate(angle[0], angle[1], angle[2])


    def generateEdges(self) -> List[np.ndarray]:
        return [
                np.array([[self.dots[0], self.dots[1]], [self.dots[1], self.dots[5]], [self.dots[5], self.dots[4]], [self.dots[4], self.dots[0]]]),

                np.array([[self.dots[1], self.dots[2]], [self.dots[2], self.dots[6]], [self.dots[6], self.dots[5]], [self.dots[5], self.dots[1]]]),
                np.array([[self.dots[2], self.dots[3]], [self.dots[3], self.dots[7]], [self.dots[7], self.dots[6]], [self.dots[6], self.dots[2]]]),
                np.array([[self.dots[3], self.dots[0]], [self.dots[0], self.dots[4]], [self.dots[4], self.dots[7]], [self.dots[7], self.dots[3]]]),
                np.array([[self.dots[0], self.dots[3]], [self.dots[3], self.dots[2]], [self.dots[2], self.dots[1]], [self.dots[1], self.dots[0]]]),
                np.array([[self.dots[4], self.dots[7]], [self.dots[7], self.dots[6]], [self.dots[6], self.dots[5]], [self.dots[5], self.dots[4]]])
            ]


    
        
        
