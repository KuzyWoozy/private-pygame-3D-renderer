from numpy import arctan, tan, pi, double
from typing import Tuple

SCREEN_SIZE: Tuple[int, int] = (800, 600)

FOV_HOR: float = 90
FOV_VERT: float = 2 * arctan((SCREEN_SIZE[1]/2)/((SCREEN_SIZE[0]/2)/tan((FOV_HOR/2 * pi/180)))) * 180/pi

ROTATION_ANGLE: int = 2
MOVEMENT_DISTANCE: int = 5

DTYPE=double

