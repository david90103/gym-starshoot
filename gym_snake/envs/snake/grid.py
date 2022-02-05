import numpy as np

class Grid():

    """
    This class contains all data related to the grid in which the game is contained.
    The information is stored as a numpy array of pixels.
    The grid is treated as a cartesian [x,y] plane in which [0,0] is located at
    the upper left most pixel and [max_x, max_y] is located at the lower right most pixel.

    Note that it is assumed spaces that can kill a snake have a non-zero value as their 0 channel.
    It is also assumed that HEAD_COLOR has a 255 value as its 0 channel.
    """

    BODY_COLOR = np.array([1,0,0], dtype=np.uint8)
    HEAD_COLOR = np.array([255, 0, 0], dtype=np.uint8)
    FOOD_COLOR = np.array([0,0,255], dtype=np.uint8)
    
    SPACE_COLOR = np.array([0,0,0], dtype=np.uint8)

    def __init__(self, grid_size=[30,30], unit_size=10, unit_gap=1):
        """
        grid_size - tuple, list, or ndarray specifying number of atomic units in
                    both the x and y direction
        unit_size - integer denoting the atomic size of grid units in pixels
        """

        self.unit_size = 1
        self.unit_gap = 0
        self.grid_size = np.asarray(grid_size, dtype=np.int) # size in terms of units
        height = self.grid_size[1]*self.unit_size
        width = self.grid_size[0]*self.unit_size
        channels = 3
        self.grid = np.zeros((height, width, channels), dtype=np.uint8)
        self.grid[:,:,:] = self.SPACE_COLOR
        self.open_space = grid_size[0]*grid_size[1]

    def color_of(self, coord):
        """
        Returns the color of the specified coordinate

        coord - x,y integer coordinates as a tuple, list, or ndarray
        """

        return self.grid[int(coord[1]*self.unit_size), int(coord[0]*self.unit_size), :]

    def cover(self, coord, color):
        """
        Colors a single space on the grid. Use erase if creating an empty space on the grid.
        This function is used like draw but without affecting the open_space count.

        coord - x,y integer coordinates as a tuple, list, or ndarray
        color - [R,G,B] values as a tuple, list, or ndarray
        """

        if self.off_grid(coord):
            return False
        x = int(coord[0]*self.unit_size)
        end_x = x+self.unit_size
        y = int(coord[1]*self.unit_size)
        end_y = y+self.unit_size
        self.grid[y:end_y, x:end_x, :] = np.asarray(color, dtype=np.uint8)
        return True

    def draw(self, coord, color):
        """
        Colors a single space on the grid. Use erase if creating an empty space on the grid.
        Affects the open_space count.

        coord - x,y integer coordinates as a tuple, list, or ndarray
        color - [R,G,B] values as a tuple, list, or ndarray
        """

        if self.cover(coord, color):
            self.open_space -= 1
            return True
        else:
            return False


    def draw_player(self, player):
        cord = player.position
        h = player.HIT_BOX_SIZE // 2
        for i in range(-h, h):
            for j in range(-h, h):
                self.draw([cord[0]+i, cord[1]+j], player.color)
    
    def erase_player(self, player):
        cord = player.position
        h = player.HIT_BOX_SIZE // 2
        for i in range(-h, h):
            for j in range(-h, h):
                self.draw([cord[0]+i, cord[1]+j], self.SPACE_COLOR)
    
    def draw_player_mp(self, player):
        y = 0 if player.id == 1 else self.grid_size[1] - 1
        for i in range(round(self.grid_size[0] * player.mp / 10)):
            self.draw([i, y], player.color)
    
    def erase_player_mp(self, player):
        y = 0 if player.id == 1 else self.grid_size[1] - 1
        for i in range(self.grid_size[1]):
            self.draw([i, y], self.SPACE_COLOR)

    def draw_bullet(self, bullet):
        cord = bullet.position
        h = bullet.HIT_BOX_SIZE // 2
        for i in range(-h, h):
            for j in range(-h, h):
                self.draw([cord[0]+i, cord[1]+j], bullet.color)
    
    def erase_bullet(self, bullet):
        cord = bullet.position
        h = bullet.HIT_BOX_SIZE // 2
        for i in range(-h, h):
            for j in range(-h, h):
                self.draw([cord[0]+i, cord[1]+j], self.SPACE_COLOR)

    def off_grid(self, coord):
        """
        Checks if argued coord is off of the grid

        coord - x,y integer coordinates as a tuple, list, or ndarray
        """

        return coord[0]<0 or coord[0]>=self.grid_size[0] or coord[1]<0 or coord[1]>=self.grid_size[1]
