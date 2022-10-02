from queue import deque
import numpy as np

class Snake():

    STOP = 0
    RIGHT = 1
    LEFT = 2
    HIT_BOX_SIZE = 4
    MOVE_SPEED = 1

    def __init__(self, id, coord_start, color):
        self.id = id
        self.direction = self.STOP if id == 1 else self.LEFT
        self.position = np.asarray(coord_start).astype(np.int)
        self.color = color
        self.mp = 0
    
