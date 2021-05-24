import sys
sys.path.append("src")

from Environment3D import Environment3D

if __name__ == "__main__":
    envi: Environment3D = Environment3D()
    envi.run()
