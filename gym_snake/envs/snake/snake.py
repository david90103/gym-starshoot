from queue import deque
import numpy as np

class Snake():

    STOP = 0
    RIGHT = 1
    LEFT = 2
    HIT_BOX_SIZE = 4
    MOVE_SPEED = 1

    def __init__(self, coord_start, color):

        self.direction = self.STOP
        self.position = np.asarray(coord_start).astype(np.int)
        self.color = color
    
