from gym_snake.envs.snake import Snake
from gym_snake.envs.snake import Grid
from gym_snake.envs.snake.bullet import Bullet
import numpy as np

class Controller():

    WALL_GAP_INIT = 5
    WALL_GAP_MAX = 32
    MP_SPEED = 0.05
    WALL_SHRINK_SPEED = 0.1
    WALL_COUNT_INIT = 10
    PLAYER0_COLOR = np.array([0,0,255], dtype=np.uint8)
    PLAYER1_COLOR = np.array([255,0,0], dtype=np.uint8)

    def __init__(self, grid_size, unit_size, unit_gap):

        self.grid = Grid(grid_size, unit_size, unit_gap)
        self.wall_gap = self.WALL_GAP_INIT
        self.players = [Snake(0, [self.grid.grid_size[0] // 2 - 10, self.grid.grid_size[1] - self.wall_gap], self.PLAYER0_COLOR),
                        Snake(1, [self.grid.grid_size[0] // 2 + 10, self.wall_gap], self.PLAYER1_COLOR),]
        self.bullets = []
        self.done = False
        self.wall_counter = self.WALL_COUNT_INIT
        self.time_punish = 0
        
        for p in self.players:
            self.grid.draw_player(p)

    def move_player(self, player_idx):

        player = self.players[player_idx]
        self.grid.erase_player(player)

        if player.direction == player.RIGHT:
            player.position = np.asarray([self.bounded_x(player.position[0]+player.MOVE_SPEED), player.position[1]]).astype(np.int)
        elif player.direction == player.LEFT:
            player.position = np.asarray([self.bounded_x(player.position[0]-player.MOVE_SPEED), player.position[1]]).astype(np.int)

        self.grid.draw_player(player)
    
    def bounded_x(self, position):
        return max(0, min(position, self.grid.grid_size[0]))
    
    def add_mp(self, player_idx):
        player = self.players[player_idx]
        self.grid.erase_player_mp(player)
        player.mp = min(10, player.mp + self.MP_SPEED)
        self.grid.draw_player_mp(player)
    
    def move_bullets(self):
        should_remove = []
        for i, b in enumerate(self.bullets):
            self.grid.erase_bullet(b)
            b.path_idx += 1
            if b.path_idx >= len(b.fullpath):
                should_remove.append(i)
                continue
            b.position = b.fullpath[b.path_idx]
            self.grid.draw_bullet(b)
        for i in sorted(should_remove, reverse=True):
            self.bullets.pop(i)
        
    def check_hit(self):
        for i in range(self.grid.grid_size[0]):
            if np.array_equal(self.grid.color_of([i, self.players[1].position[1]]), self.players[1].color) and \
               np.array_equal(self.grid.color_of([i, self.players[1].position[1]+1]), self.players[0].color):
                return True, 0
            if np.array_equal(self.grid.color_of([i, self.players[0].position[1]]), self.players[0].color) and \
               np.array_equal(self.grid.color_of([i, self.players[0].position[1]-2]), self.players[1].color):
                return True, 1
            # TODO: Bullet does not travel every pixel!!!
            
        return False, None
    
    def check_move_wall(self):
        if self.wall_counter <= 0:
            self.move_wall()
            self.wall_counter = self.WALL_COUNT_INIT
        else:
            self.wall_counter -= self.WALL_SHRINK_SPEED

    def move_wall(self):
        self.wall_gap = min(self.wall_gap + 1, self.WALL_GAP_MAX)
        p1_pos = self.grid.grid_size[1] - self.wall_gap
        p2_pos = self.wall_gap
        p1 = self.players[0]
        p2 = self.players[1]
        self.grid.erase_player(p1)
        p1.position = np.asarray([p1.position[0], p1_pos]).astype(np.int)
        self.grid.draw_player(p1)
        self.grid.erase_player(p2)
        p2.position = np.asarray([p2.position[0], p2_pos]).astype(np.int)
        self.grid.draw_player(p2)

    def step(self, action):
        self.time_punish += 0.0001
        rewards = -self.time_punish

        if type(action) != type([]):
            action = [action]

        for i, act in enumerate(action):
            if act == 1:
                self.players[0].direction = self.players[0].LEFT
            elif act == 2:
                self.players[0].direction = self.players[0].RIGHT
            elif act == 3:
                direction = -1
                if self.players[0].mp >= 1:
                    self.players[0].mp -= 1
                    self.bullets.append(Bullet(self.players[0].position, self.players[0].color, direction))
            elif act == 4:
                self.players[1].direction = self.players[1].LEFT
            elif act == 5:
                self.players[1].direction = self.players[1].RIGHT
            elif act == 6:
                direction = 1 
                if self.players[1].mp >= 1:
                    self.players[1].mp -= 1
                    self.bullets.append(Bullet(self.players[1].position, self.players[1].color, direction))

        for i in range(len(self.players)):
            self.add_mp(i)
            self.move_player(i)

            # rewards.append(self.move_result(direction, i))

        self.move_bullets()
        self.check_move_wall()
        finish, winner = self.check_hit()
        if finish:
            self.done = True
            rewards = rewards + 10 if winner == 0 else rewards - 10
            print("Player", winner, "is the winner. Reward:", round(rewards, 4))
        
        if self.time_punish > 0.1:
            self.done = True
            print("Times up. Reward:", round(rewards, 4))

        return self.grid.grid.copy(), rewards, self.done, {"snakes_remaining":1}
