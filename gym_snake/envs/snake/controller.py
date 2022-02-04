from gym_snake.envs.snake import Snake
from gym_snake.envs.snake import Grid
from gym_snake.envs.snake.bullet import Bullet
import numpy as np

class Controller():

    def __init__(self, grid_size, unit_size, unit_gap):

        self.grid = Grid(grid_size, unit_size, unit_gap)
        PLAYER1_COLOR = np.array([255,0,0], dtype=np.uint8)
        PLAYER2_COLOR = np.array([0,0,255], dtype=np.uint8)
        self.players = [Snake([self.grid.grid_size[0] // 2, self.grid.grid_size[1] * 7 // 8], PLAYER2_COLOR),
                        Snake([self.grid.grid_size[0] // 2, self.grid.grid_size[1] // 8], PLAYER1_COLOR),]
        self.bullets = []
        self.done = False
        
        for p in self.players:
            self.grid.draw_player(p)

    def move_player(self, player_idx):

        player = self.players[player_idx]
        if player.direction == player.STOP:
            return

        self.grid.erase_player(player)

        if player.direction == player.RIGHT:
            player.position = np.asarray([self.bounded_x(player.position[0]+player.MOVE_SPEED), player.position[1]]).astype(np.int)
        elif player.direction == player.LEFT:
            player.position = np.asarray([self.bounded_x(player.position[0]-player.MOVE_SPEED), player.position[1]]).astype(np.int)

        self.grid.draw_player(player)
    
    def bounded_x(self, position):
        return max(0, min(position, self.grid.grid_size[0]))
    
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
        

    # def move_result(self, direction, snake_idx=0):
    #     """
    #     Checks for food and death collisions after moving snake. Draws head of snake if
    #     no death scenarios.
    #     """

    #     snake = self.snakes[snake_idx]
    #     if type(snake) == type(None):
    #         return 0

    #     # Check for death of snake
    #     if self.grid.check_death(snake.head):
    #         self.dead_snakes[snake_idx] = self.snakes[snake_idx]
    #         self.snakes[snake_idx] = None
    #         self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
    #         self.grid.connect(snake.body.popleft(), snake.body[0], self.grid.SPACE_COLOR)
    #         reward = -1
    #     # Check for reward
    #     elif self.grid.food_space(snake.head):
    #         self.grid.draw(snake.body[0], self.grid.BODY_COLOR) # Redraw tail
    #         self.grid.connect(snake.body[0], snake.body[1], self.grid.BODY_COLOR)
    #         self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
    #         reward = 1
    #         self.grid.new_food()
    #     else:
    #         reward = 0
    #         empty_coord = snake.body.popleft()
    #         self.grid.connect(empty_coord, snake.body[0], self.grid.SPACE_COLOR)
    #         self.grid.draw(snake.head, snake.head_color)

    #     self.grid.connect(snake.body[-1], snake.head, self.grid.BODY_COLOR)

    #     return reward

    def step(self, action):
       
        rewards = []

        if type(action) == type(int()):
            action = [action]

        for i, act in enumerate(action):
            if act == 1:
                self.players[i].direction = self.players[i].LEFT
            elif act == 2:
                self.players[i].direction = self.players[i].RIGHT
            elif act == 3:
                direction = -1
                if i == 1:
                    direction = 1
                self.bullets.append(Bullet(self.players[i].position, self.players[i].color, direction))
                pass # fire bullet

            self.move_player(i)
            # rewards.append(self.move_result(direction, i))

        self.move_bullets()

        if len(rewards) is 1:
            return self.grid.grid.copy(), rewards[0], self.done, {"snakes_remaining":1}
        else:
            return self.grid.grid.copy(), rewards, self.done, {"snakes_remaining":1}
