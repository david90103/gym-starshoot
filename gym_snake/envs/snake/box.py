from queue import deque
import random
import numpy as np

class Box():

    BOX_COLOR = np.array([255,255,0], dtype=np.uint8)
    RIGHT = 1
    LEFT = -1
    HIT_BOX_SIZE = 4
    PATH_RENDER_LIMIT = 120

    def __init__(self, coord_start, direction):
        self.direction = direction
        self.position = np.asarray(coord_start).astype(np.int)
        self.color = self.BOX_COLOR
        self.path_idx = 0
        self.box_speed = random.choice([0.5, 1])
        self.fullpath = self.normal_path()
    
    def normal_path(self):
        path = []
        first_pos = self.position
        for i in range(self.PATH_RENDER_LIMIT):
            path.append([round(first_pos[0] + i * self.box_speed * self.direction), first_pos[1]])
        
        return path


class PBox():

    BOX_COLOR = np.array([255,0,255], dtype=np.uint8)
    UP = 1
    DOWN = -1
    HIT_BOX_SIZE = 2
    BOX_SPEED = 0.5
    PATH_RENDER_LIMIT = 80

    def __init__(self, coord_start, direction):
        self.direction = direction
        self.position = np.asarray(coord_start).astype(np.int)
        self.color = self.BOX_COLOR
        self.path_idx = 0
        self.fullpath = self.normal_path()
    
    def normal_path(self):
        path = []
        first_pos = self.position
        for i in range(self.PATH_RENDER_LIMIT):
            path.append([first_pos[0], round(first_pos[1] + i * self.BOX_SPEED * self.direction)])
        
        return path
    
