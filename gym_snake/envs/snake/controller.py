import random
from gym_snake.envs.snake import Snake
from gym_snake.envs.snake import Grid
from gym_snake.envs.snake.bullet import Bullet
from gym_snake.envs.snake.box import Box, PBox
import numpy as np

class Controller():

    # Game
    WALL_GAP_INIT = 5
    WALL_GAP_MAX = 32
    MP_SPEED = 0.05
    WALL_SHRINK_SPEED = 0.1
    WALL_COUNT_INIT = 10
    GEN_BOX_PROB = 0.1
    PLAYER0_COLOR = np.array([0,0,255], dtype=np.uint8)
    PLAYER1_COLOR = np.array([255,0,0], dtype=np.uint8)

    # Agent
    TIME_PUNISHMENT = 0
    NEAR_WALL_PUNISHMENT = 0
    SHOOT_PUNISHMENT = 0
    SHOOT_NOTHING_PUNISHMENT = 0
    HIT_BOX_REWARD = 0
    EAT_PBOX_REWARD = 0
    STAND_MID_REWARD = 0

    def __init__(self, grid_size, unit_size, unit_gap):

        self.grid = Grid(grid_size, unit_size, unit_gap)
        self.wall_gap = self.WALL_GAP_INIT
        self.players = [Snake(0, [self.grid.grid_size[0] // 2 - 10, self.grid.grid_size[1] - self.wall_gap], self.PLAYER0_COLOR),
                        Snake(1, [self.grid.grid_size[0] // 2 + 10, self.wall_gap], self.PLAYER1_COLOR),]
        self.bullets = []
        self.boxes = []
        self.p_boxes = []
        self.done = False
        self.wall_counter = self.WALL_COUNT_INIT
        self.rewards_p1 = 0
        self.rewards_p2 = 0
        # Testing
        self.step_count = 0
        self.hit_count = 0
        self.prev_pos = [0, 0]
        self.box_gened = 0
        
        for p in self.players:
            self.grid.draw_player(p)

    def move_player(self, player_idx):

        player = self.players[player_idx]
        self.grid.erase_player(player)

        if player.direction == player.RIGHT:
            player.position = np.asarray([self.bounded_x(player.position[0]+player.MOVE_SPEED), player.position[1]]).astype(np.int)
        elif player.direction == player.LEFT:
            player.position = np.asarray([self.bounded_x(player.position[0]-player.MOVE_SPEED), player.position[1]]).astype(np.int)
        if player.position[0] <= 2 or player.position[0] >= self.grid.grid_size[0]-2:
            self.rewards_p1 -= self.NEAR_WALL_PUNISHMENT
        if player.position[0] > self.grid.grid_size[0] / 4 and player.position[0] < self.grid.grid_size[0] * 3 / 4:
            self.rewards_p1 += self.STAND_MID_REWARD
        self.grid.draw_player(player)
    
    def bounded_x(self, position):
        return max(0, min(round(position), self.grid.grid_size[0]))
    
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
            if b.path_idx >= len(b.fullpath) or b.position[1] > self.grid.grid_size[1] or b.position[1] < 0:
                should_remove.append(i)
                continue
            b.position = b.fullpath[b.path_idx]
            self.grid.draw_bullet(b)
        for i in sorted(should_remove, reverse=True):
            self.bullets.pop(i)
    
    def move_boxes(self):
        should_remove = []
        for i, b in enumerate(self.boxes):
            self.grid.erase_bullet(b)
            b.path_idx += 1
            if b.path_idx >= len(b.fullpath) or b.position[0] > self.grid.grid_size[0] or b.position[0] < 0:
                should_remove.append(i)
                continue
            b.position = b.fullpath[b.path_idx]
            self.grid.draw_bullet(b)
        for i in sorted(should_remove, reverse=True):
            self.boxes.pop(i)
    
    def move_pboxes(self):
        should_remove = []
        for i, b in enumerate(self.p_boxes):
            self.grid.erase_bullet(b)
            b.path_idx += 1
            if b.path_idx >= len(b.fullpath) or b.position[0] > self.grid.grid_size[0] or b.position[0] < 0:
                should_remove.append(i)
                continue
            b.position = b.fullpath[b.path_idx]
            self.grid.draw_bullet(b)
        for i in sorted(should_remove, reverse=True):
            self.p_boxes.pop(i)
        
    def check_kill(self):
        hbs = self.players[0].HIT_BOX_SIZE
        for i in range(self.grid.grid_size[0]):
            if np.array_equal(self.grid.color_of([i, self.players[1].position[1]]), self.players[1].color) and \
               np.array_equal(self.grid.color_of([i, self.players[1].position[1]+hbs//2]), self.players[0].color):
                return True, 0
            if np.array_equal(self.grid.color_of([i, self.players[0].position[1]]), self.players[0].color) and \
               np.array_equal(self.grid.color_of([i, self.players[0].position[1]-hbs//2]), self.players[1].color):
                return True, 1
            # TODO: Bullet does not travel every pixel!!!
            
        return False, None
    
    def check_shrink_wall(self):
        if self.wall_counter <= 0:
            self.shrink_wall()
            self.wall_counter = self.WALL_COUNT_INIT
        else:
            self.wall_counter -= self.WALL_SHRINK_SPEED

    def shrink_wall(self):
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
    
    def check_gen_box(self):
        if len(self.boxes) < 2 and random.uniform(0, 1) < self.GEN_BOX_PROB:
            self.box_gened += 1
            direction = random.randint(0, 1)
            if direction == 0:
                direction = -1
            # xpos = 0 if direction == 1 else self.grid.grid_size[0] - 1
            xpos = random.uniform(5, self.grid.grid_size[0] - 5)
            ypad = random.uniform(-10, +10)
            self.boxes.append(Box([xpos, self.grid.grid_size[1] // 2 + ypad], direction))

    def check_hit_box(self):
        should_remove = []
        for idx, box in enumerate(self.boxes):
            for i in range(box.HIT_BOX_SIZE):
                hbs = box.HIT_BOX_SIZE
                x = max(0, min(box.position[0]-hbs//2+i, self.grid.grid_size[0] - 1))
                y = max(0, min(box.position[1]+hbs//2+1, self.grid.grid_size[1] - 1))
                # Player 1
                if np.array_equal(self.grid.color_of([x, y]), self.players[0].color) and idx not in should_remove:
                    self.grid.erase_bullet(box)
                    self.p_boxes.append(PBox(box.position, 1))
                    should_remove.append(idx)
                    self.rewards_p1 += self.HIT_BOX_REWARD
                    self.hit_count += 1
                # Player 2
                y = max(0, min(box.position[1]-hbs//2-1, self.grid.grid_size[1] - 1))
                if np.array_equal(self.grid.color_of([x, y]), self.players[1].color) and idx not in should_remove:
                    self.grid.erase_bullet(box)
                    self.p_boxes.append(PBox(box.position, -1))
                    should_remove.append(idx)
                    self.rewards_p2 += self.HIT_BOX_REWARD
        for i in sorted(should_remove, reverse=True):
            self.boxes.pop(i)
    
    def check_hit_pbox(self):
        if len(self.p_boxes) == 0:
            return
        should_remove = []
        p1 = self.players[0]
        p2 = self.players[1]
        for pbox_idx, pb in enumerate(self.p_boxes):
            for i in range(p1.HIT_BOX_SIZE * 2):
                hbs = p1.HIT_BOX_SIZE
                x = max(0, min(p1.position[0]-hbs//2+i, self.grid.grid_size[0] - 1))
                y = max(0, min(p1.position[1]-hbs//2, self.grid.grid_size[1] - 1))
                # Player 1
                if np.array_equal([x, y], pb.position) and pbox_idx not in should_remove:
                    self.grid.erase_bullet(pb)
                    should_remove.append(pbox_idx)
                    self.rewards_p1 += self.EAT_PBOX_REWARD
                    break
                # Player 2
                x = max(0, min(p2.position[0]-hbs//2+i, self.grid.grid_size[0] - 1))
                y = max(0, min(p2.position[1]+hbs//2+1, self.grid.grid_size[1] - 1))
                if np.array_equal([x, y], pb.position) and pbox_idx not in should_remove:
                    self.grid.erase_bullet(pb)
                    should_remove.append(pbox_idx)
                    self.rewards_p2 += self.EAT_PBOX_REWARD
                    break
        for i in sorted(should_remove, reverse=True):
            self.p_boxes.pop(i)

    # def check_stop(self):
    #     if np.array_equal(self.players[0].position, self.prev_pos):
    #         self.rewards_p1 -= self.STOP_PUNISHMENT
    #     self.prev_pos = self.players[0].position

    def get_obs(self):

        # [ player_x, player_y, player_direct, player_mp, opponent_x, opponent_y, opponent_direct, opponent_mp = 8
        # ]

        # TODO
        # [
        #   box_x, box_y, box_direct, box_speed ...*3 = 12 
        #   p_box_x, p_box_y, p_box_direct ...*3 = 9
        #   bullet_x, bullet_y, bullet_color, bullet_direct, bullet_type ...*3 = 15
        # ]
        
        # TODO Normalize all values

        obs = np.zeros(8) # 44
        p = self.players[0]
        o = self.players[1]
        # Player and Opponent
        obs[0:8] = [p.position[0], p.position[1], p.direction, p.mp, o.position[0], o.position[1], o.direction, o.mp]
        # Boxes
        # offset = 8
        # c = 0
        # for b in self.boxes:
        #     if c >= 3:
        #         break
        #     obs[offset: offset+4] = [b.position[0], b.position[1], b.direction, b.box_speed]
        #     offset += 4
        #     c += 1
        # P Boxes
        # offset = 20
        # c = 0
        # for b in self.p_boxes:
        #     if c >= 3:
        #         break
        #     obs[offset: offset+3] = [b.position[0], b.position[1], b.direction]
        #     offset += 3
        #     c += 1
        # # Bullets
        # offset = 23
        # c = 0
        # for b in self.bullets:
        #     if c >= 3:
        #         break
        #     color = 0
        #     if np.equal(b.color, self.PLAYER0_COLOR).all():
        #         color = 1
        #     obs[offset: offset+5] = [b.position[0], b.position[1], b.direction, color, b.bullet_type]
        #     offset += 5
        #     c += 1

        return obs

    def step(self, action):
        self.rewards_p1 -= self.TIME_PUNISHMENT
        self.rewards_p2 -= self.TIME_PUNISHMENT

        if type(action) != type([]):
            action = [action]

        for i, act in enumerate(action):
            if act == 0:
                self.players[0].direction = self.players[0].LEFT
            elif act == 1:
                self.players[0].direction = self.players[0].RIGHT
            elif act == 2:
                direction = -1
                if self.players[0].mp >= 1:
                    self.players[0].mp -= 1
                    self.bullets.append(Bullet(self.players[0].position, self.players[0].color, direction))
                    self.rewards_p1 -= self.SHOOT_PUNISHMENT
                else:
                    self.rewards_p1 -= self.SHOOT_NOTHING_PUNISHMENT
            # elif act == 4:
            #     self.players[1].direction = self.players[1].LEFT
            # elif act == 5:
            #     self.players[1].direction = self.players[1].RIGHT
            # elif act == 6:
            #     direction = 1 
            #     if self.players[1].mp >= 1:
            #         self.players[1].mp -= 1
            #         self.bullets.append(Bullet(self.players[1].position, self.players[1].color, direction))

        for i in range(len(self.players)):
            self.add_mp(i)
            self.move_player(i)

            # rewards.append(self.move_result(direction, i))

        self.move_bullets()
        # self.move_boxes()
        # self.move_pboxes()
        self.check_shrink_wall()
        # self.check_gen_box()
        # self.check_hit_box()
        # self.check_hit_pbox()
        # self.check_stop()
        finish, winner = self.check_kill()
        if finish:
            # self.done = True
            # self.rewards_p1 = self.rewards_p1 + 10 if winner == 0 else self.rewards_p1 - 10
            # self.rewards_p1 = self.rewards_p1 + 1 + (1000 - self.step_count) / 1000
            self.rewards_p1 += 1
            self.hit_count += 1
            # print("Player", winner, "is the winner. Reward:", round(self.rewards_p1, 4))
        
        # if self.rewards_p1 < -0.1:
        #     self.done = True
        #     print("Times up. Reward:", round(self.rewards_p1, 4))

        if self.step_count > 1000:
            self.done = True
            # self.rewards_p1 = 0
            print("Times up. Reward:", round(self.rewards_p1, 4), ", Hit:", self.hit_count)
        
        self.step_count += 1

        return self.get_obs(), self.grid.grid.copy(), self.rewards_p1, self.done, {"snakes_remaining":1}
