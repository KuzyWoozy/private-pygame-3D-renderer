import pygame as pg
import numpy as np

from abc import abstractmethod
from typing import Tuple, List, Callable
from Config import *

from Util import Sortable


# Convert to radians
FOV_VERT *= np.pi/180
FOV_HOR *= np.pi/180

FOV_VERT_HALF: float = FOV_VERT/2
FOV_HOR_HALF: float = FOV_HOR/2

# For debugging purposes, shifts the Z-plane forward by specified amount
CULL_OFFSET: float = 0



class Entity(Sortable):
    """
    Represents a drawable 3D object.

    Attributes:
        dots -- A numpy array (N, 3) where N is each point of the object in 3D space 
        dots_t -- transposed dots
        colour -- RGB colour of the object
        eyeOffset -- Estimated distance in the z-axis of the eye from the screen, based on the chosen FOV
        edge_width -- default size of the edges
        velocity -- velocity of the object
    """

    def __init__(self, dots: np.ndarray, colour: Tuple[int, int, int], velocity: Tuple[float, float, float]) -> None:
        """
        Parameters:
            dots -- Location of each point of the object in 3D space
            colour -- RGB colouring of the object
        """
        global FOV_HOR_HALF, SCREEN_SIZE, DTYPE

        self.dots: np.ndarray = dots
        self.dots_t: np.ndarray = self.dots.transpose((1, 0))

        self.colour = colour
        # Calculate estimated offset of the eye based on horizontal field of view
        self.eyeOffset: float = (SCREEN_SIZE[0]/2)/(np.tan(FOV_HOR_HALF))
        self.edge_width: int = 2

        self.velocity: np.ndarray = np.array(velocity, dtype=DTYPE)
        
    def translate(self, x_dist: float, y_dist: float, z_dist: float) -> None:
        """Translate the object in x, y, z directions respectively by the specified magnitude."""
        self.dots += np.array([x_dist, y_dist, z_dist], dtype=DTYPE)

    def velocity(self, x: float, y: float, z: float) -> None:
        self.velocity = np.array([x, y, z], dtype=DTYPE)

    def move(self) -> None:
        self.dots += self.velocity

    @staticmethod
    def cull(edges: List[np.ndarray]) -> List[np.ndarray]:
        """Projects bad polygons onto Z hyperplane and then FOV hyperplanes.

        Parameters: edges -- List of polygons of the object generated from edges via dots

        Returns: List of polygons after the projection of out of place polygons
        """
        global CULL_OFFSET
        

        def planeIntersect(poly: np.ndarray, mask_func: Callable[[np.ndarray], np.ndarray], hyperplane: np.ndarray) -> np.ndarray:
            """ Helper method for polygon to hyperplane projection.
            Parameters: 
                poly -- Polygon to project.
                mask_func(poli) -- Lambda generating mask for bad polygons.
                    poli -- Polygons to check.

            Returns: Projected polygon.
            """
            mask: np.ndarray = mask_func(poly)
            # Return if all polygons do not need projection
            if ((~mask).all()):
                return poly

            
            seal: np.ndarray = np.empty((2,3), dtype=DTYPE)
            seal.fill(np.nan)

            for i in range(0, 2):
                # Move on if no projections to be made with the current point within the edge
                if (~mask[:, i]).all():
                    continue

                P: np.ndarray = poly[mask[:, i], 0]
                Q: np.ndarray = poly[mask[:, i], 1]
                PQ_diff: np.ndarray = P-Q
                    
                denominator = PQ_diff.dot(hyperplane).item()
                # Calculate parametric t coefficient for projected vector
                # Handle division by zero
                if (np.abs(denominator) < 1e-2):
                    t: float = 1
                else:
                    # ((P-Q)t + Q).N= 0
                    # solve for t, N is hyperplane P, Q are dots of edge
                    t = ((np.array([0, 0, CULL_OFFSET], dtype=DTYPE)-Q).dot(hyperplane))/denominator

                projected_point = (PQ_diff * t) + Q
                poly[mask[:, i], i] = projected_point

                seal[i] = projected_point
             
            # A check if we have two intersections,
            # if this is the case we need to 'seal' the gap to form
            # a complete polygon again
            if not np.isnan(np.sum(seal)):
                m = mask.any(axis=-1)
                p: np.ndarray = np.argwhere(m)
                if m.size > 2:
                    if (np.abs(p[0] - p[1])) == 1:
                        z = p[-1]
                    else:
                        z = p[0]

                    # A check if our edge is the correct way round
                    if (np.abs(seal[1] - poly[z, 0]) < 1e-2).all():
                        poly = np.insert(poly, z, seal, axis=0)
                    else:
                        poly = np.insert(poly, z, seal[::-1], axis=0)
                
                # handle of special case
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
            # z plane projection
            poly = poly[(poly[:, :, 2] > CULL_OFFSET).any(axis=-1)]
            poly = planeIntersect(poly, lambda x: (x[:, :, 2] < CULL_OFFSET), np.array([0, 0, 1], dtype=DTYPE))
            
            # zero divide value guard
            poly[poly[:, :, 2] < 1e-2, 2] = 1e-2
                   
            # borders

            # right plane projection
            poly = poly[(np.arctan(poly[:, :, 0] / poly[:, :, 2]) <= FOV_HOR_HALF).any(axis=-1)]
            poly = planeIntersect(poly, lambda x: (np.arctan(x[:, :, 0] / x[:, :, 2]) > FOV_HOR_HALF), np.array([np.sin(FOV_HOR_HALF-np.pi/2), 0, np.cos(FOV_HOR_HALF-np.pi/2)], dtype=DTYPE))
        
            # zero divide value guard
            poly[poly[:, :, 2] < 1e-2, 2] = 1e-2

        
            # left plane
            poly = poly[(np.arctan(poly[:, :, 0] / poly[:, :, 2]) >= -FOV_HOR_HALF).any(axis=-1)]
            poly = planeIntersect(poly, lambda x: (np.arctan(x[:, :, 0] / x[:, :, 2]) < -FOV_HOR_HALF), np.array([np.sin(-FOV_HOR_HALF+np.pi/2), 0, np.cos(-FOV_HOR_HALF+np.pi/2)], dtype=DTYPE))
        
            # zero divide value guard
            poly[poly[:, :, 2] < 1e-2, 2] = 1e-2

            # up plane projection
            poly = poly[(np.arctan(poly[:, :, 1] / poly[:, :, 2]) >= -FOV_VERT_HALF).any(axis=-1)]
            poly = planeIntersect(poly, lambda x: (np.arctan(x[:, :, 1] / x[:, :, 2]) < -FOV_VERT_HALF), np.array([0, np.sin(-FOV_VERT_HALF+np.pi/2), np.cos(-FOV_VERT_HALF+np.pi/2)], dtype=DTYPE))

            # zero divide value guard
            poly[poly[:, :, 2] < 1e-2, 2] = 1e-2

            # down plane projection
            poly = poly[(np.arctan(poly[:, :, 1] / poly[:, :, 2]) <= FOV_VERT_HALF).any(axis=-1)]
            poly = planeIntersect(poly, lambda x: (np.arctan(x[:, :, 1] / x[:, :, 2]) > FOV_VERT_HALF), np.array([0, np.sin(FOV_VERT_HALF-np.pi/2), np.cos(FOV_VERT_HALF-np.pi/2)], dtype=DTYPE))

            # zero divide value guard
            poly[poly[:, :, 2] < 1e-2, 2] = 1e-2
             
            modified_edges.append(poly)
        
        return modified_edges
        
           
    # angle == [x_rot, y_rot, z_rot]
    def rotate(self, x_rot: float, y_rot: float, z_rot: float) -> None:
        """Rotate the object in x, y, z directions respectively by the specified magnitude (degrees)."""
        x_rot, y_rot, z_rot = np.radians((x_rot, y_rot, z_rot))
       
        rot_x_mat: np.ndarray = np.array([
                        [1, 0, 0],
                        [0, np.cos(x_rot), -np.sin(x_rot)],
                        [0, np.sin(x_rot), np.cos(x_rot)]], dtype=DTYPE)
        rot_y_mat: np.ndarray = np.array([
                        [np.cos(y_rot), 0, np.sin(y_rot)],
                        [0, 1, 0],
                        [-np.sin(y_rot), 0, np.cos(y_rot)]], dtype=DTYPE)
        rot_z_mat: np.ndarray = np.array([
                        [np.cos(z_rot), -np.sin(z_rot), 0],
                        [np.sin(z_rot), np.cos(z_rot), 0],
                        [0, 0, 1]], dtype=DTYPE)

        # No need to update dots, as dots_t is a view
        np.matmul((rot_z_mat @ rot_y_mat @ rot_x_mat), self.dots_t, out=self.dots_t)
        # Gotta rotate the velocity too
        np.matmul((rot_z_mat @ rot_y_mat @ rot_x_mat), self.velocity, out=self.velocity)
                  

    def blit(self, surf: pg.surface.Surface) -> None:
        global SCREEN_SIZE 
       
        # Culling
        dots_mask = self.dots[:, 2] > 1e-2 
        
        # Stop trying to draw if the object cannot be projected onto the screen
        if (~dots_mask).all():
            return
        
        edges = Entity.cull(Rectangle.generateEdges(self.dots))
        
        # Ugly ik, but there isn't a clean way to handle numpy arrays of arbitary dimensions
        modified_edges = []
        for poly in edges:
            # zero guard
            poly[poly[:, :, 2] < 1e-2, 2] = 1e-2
            # Project the points onto the screen
            poly[:, :, 1] = self.eyeOffset * (poly[:, :, 1]/poly[:, :, 2])
            poly[:, :, 0] = self.eyeOffset * (poly[:, :, 0]/poly[:, :, 2])

            # Shift origin to pygame position
            poly[:, :, 1] += SCREEN_SIZE[1]/2
            poly[:, :, 0] += SCREEN_SIZE[0]/2
            # Remove z axis as we have finished projecting points onto
            # the 2D screen
            poly = np.delete(poly, 2, axis=-1)

            modified_edges.append(poly)
        edges = modified_edges

        # Perform the drawing operations

        for poly in edges:
            if poly.shape[0] >= 2:
                pg.draw.polygon(surf, self.colour, np.concatenate(poly))
        
        for poly in edges:
            for (p1, p2) in poly:
                #pg.draw.circle(surf, (255, 0, 0), p1, radius=20)
                #pg.draw.circle(surf, (255, 0, 0), p2, radius=20)
                pg.draw.line(surf, (0,0,0), p1, p2, width=self.edge_width)
    
    @staticmethod
    @abstractmethod
    def generateEdges(dots: np.ndarray) -> List[np.ndarray]:
        """Generate edges of the object from the given dots"""
        pass



class Rectangle(Entity):
    """Specialized entity which represents a rectangle."""
    def __init__(self, center: Tuple[float, float, float], size: Tuple[float, float, float], angle: Tuple[float, float, float], colour: Tuple[int, int, int], velocity: Tuple[float, float, float]) -> None:
        
        dots_mask: np.ndarray = np.array([
                            [-1, -1, -1],
                            [1, -1, -1],
                            [1, -1, 1],
                            [-1, -1, 1],
                            [-1, 1, -1],
                            [1, 1, -1],
                            [1, 1, 1],
                            [-1, 1, 1]
                            ], dtype=DTYPE)
        
        self.size: np.ndarray = np.array([size[0], size[1], size[2]], dtype=DTYPE)
        dots = center + ((self.size/2) * dots_mask)

        super().__init__(dots, colour, velocity)


        self.rotate(angle[0], angle[1], angle[2])

    def sortBy(self) -> float:
        """Returns the value to sort the entity by."""
        num = np.max(self.dots[:, 2])
        assert(isinstance(num, float))
        return num

    @staticmethod
    def generateEdges(dots: np.ndarray) -> List[np.ndarray]:
        """Returns the edges generated from the given dots for the specialized shape"""
        return [
                np.array([[dots[0], dots[1]], [dots[1], dots[5]], [dots[5], dots[4]], [dots[4], dots[0]]], dtype=DTYPE),

                np.array([[dots[1], dots[2]], [dots[2], dots[6]], [dots[6], dots[5]], [dots[5], dots[1]]], dtype=DTYPE),
                np.array([[dots[2], dots[3]], [dots[3], dots[7]], [dots[7], dots[6]], [dots[6], dots[2]]], dtype=DTYPE),
                np.array([[dots[3], dots[0]], [dots[0], dots[4]], [dots[4], dots[7]], [dots[7], dots[3]]], dtype=DTYPE),
                np.array([[dots[0], dots[3]], [dots[3], dots[2]], [dots[2], dots[1]], [dots[1], dots[0]]], dtype=DTYPE),
                np.array([[dots[4], dots[7]], [dots[7], dots[6]], [dots[6], dots[5]], [dots[5], dots[4]]], dtype=DTYPE)
            ]


    
        
        
