from queue import deque
import numpy as np

class Bullet():

    HIT_BOX_SIZE = 2
    BULLET_SPEED = 2
    PATH_RENDER_LIMIT = 50
    UP = 1
    DOWN = -1

    def __init__(self, coord_start, color, direction):

        self.position = np.asarray(coord_start).astype(np.int)
        self.color = color
        self.direction = direction
        self.path_idx = 0
        self.fullpath = self.normal_path()

    
    def normal_path(self):
        path = []
        first_pos = self.position
        for i in range(self.PATH_RENDER_LIMIT):
            path.append([first_pos[0], first_pos[1] + i * self.BULLET_SPEED * self.direction])
        
        return path
        
    
