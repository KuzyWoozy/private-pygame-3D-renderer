from numpy import arctan, tan, pi

SCREEN_SIZE = (800, 600)

FOV_HOR = 90
FOV_VERT = 2 * arctan((SCREEN_SIZE[1]/2)/((SCREEN_SIZE[0]/2)/tan((FOV_HOR/2 * pi/180)))) * 180/pi

ROTATION_ANGLE = 2
MOVEMENT_DISTANCE = 5

