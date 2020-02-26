import numpy as np
import pygame as pg
import gym
from gym import spaces

class Ris(gym.Env):
    NOP, LEFT, RIGHT, ROT = range(4)

    def __init__(self, scale, width=8, height=16, piece_set='lettris'):
        self.action_space = spaces.Discrete(4)
        board = np.zeros((height, width))
        self.width = width
        self.height = height
        self.time = 0
        self.cutoff = 4000

        self.board = np.array(board, dtype=int)
        self.size = len(board)
        self.cell_size = scale
        self.piece_types = Ris.piece_sets[piece_set]
        self.falling_piece_pos = (np.random.randint(0, self.width - 3), 0)
        self.falling_piece_shape = self.piece_types[np.random.randint(0, len(self.piece_types))]
        self.subframe = 0
        self.subframes = 5
        self.screen = None
        self.incoming_garbage = 0

    def reset(self):
        board = np.zeros((self.height, self.width))
        self.board = np.array(board, dtype=int)
        self.falling_piece_pos = (np.random.randint(0, self.width - 3), 0)
        self.falling_piece_shape = self.piece_types[np.random.randint(0, len(self.piece_types))]
        self.subframe = 0

        piece = np.array(np.zeros((self.height, self.width)))
        for i in range(4):
            for j in range(4):
                if self.falling_piece_shape[j][i] == 1:
                    pos = (i + self.falling_piece_pos[0], j + self.falling_piece_pos[1])
                    piece[pos[1]][pos[0]] = 1
        self.time = 0

        
        timing_layer = np.zeros((self.height, self.width))
        
        state = np.array([[
            self.board,
            piece,
            timing_layer
        ]])

        return state
        
    def resolve_lines(self):
        removed = 0
        for i in range(len(self.board)):
            line = self.board[i]
            if all(x == 1 for x in line):
                removed = removed + 1
                for j in range(i - 1):
                    self.board[i - j] = self.board[i - j - 1]
        return removed
        
    def apply_garbage(self, n_lines):
        done = False
        
        if n_lines > 0:
            if np.any(self.board[n_lines - 1]):
                done = True
            else:
                self.board = np.roll(self.board, -n_lines, axis=0)
                for i in range(n_lines):
                    garbage_line = np.ones(self.width)
                    garbage_line[np.random.randint(0, self.width)] = 0
                    self.board[self.height - 1 - i] = garbage_line

    def step_old(self, action):
        self.time = self.time + 1
        self.subframe = self.subframe + 1
        done = False
        reward = 0
        lines_cleared = 0
    
        if action == Ris.LEFT:
            coll = False
            for i in range(4):
                for j in range(4):
                    if not coll and self.falling_piece_shape[j][i] == 1:
                        pos_left = (i + self.falling_piece_pos[0] - 1, j + self.falling_piece_pos[1])
                        if pos_left[0] < 0 or self.board[pos_left[1]][pos_left[0]] != 0:
                            coll = True
            if not coll:
                self.falling_piece_pos = (self.falling_piece_pos[0] - 1, self.falling_piece_pos[1])
        
        if action == Ris.RIGHT:
            coll = False
            for i in range(4):
                for j in range(4):
                    if not coll and self.falling_piece_shape[j][i] == 1:
                        pos_left = (i + self.falling_piece_pos[0] + 1, j + self.falling_piece_pos[1])
                        if pos_left[0] >= len(self.board[0]) or self.board[pos_left[1]][pos_left[0]] != 0:
                            coll = True
            if not coll:
                self.falling_piece_pos = (self.falling_piece_pos[0] + 1, self.falling_piece_pos[1])
            
        if action == Ris.ROT:
            rotated = np.rot90(self.falling_piece_shape)
            coll = False
            for i in range(4):
                for j in range(4):
                    if not coll and rotated[j][i] == 1:
                        pos = (i + self.falling_piece_pos[0], j + self.falling_piece_pos[1])
                        if pos[0] not in range(0, len(self.board[0])) or \
                           pos[1] not in range(0, len(self.board)) or \
                           self.board[pos[1]][pos[0]] != 0:
                            coll = True
            if not coll:
                self.falling_piece_shape = rotated

        if self.subframe == self.subframes - 1:
            self.subframe = 0
            
            coll = False
            for i in range(4):
                for j in range(4):
                    if not coll and self.falling_piece_shape[j][i] == 1:
                        pos_below = (i + self.falling_piece_pos[0], j + self.falling_piece_pos[1] + 1)
                        if pos_below[1] >= len(self.board) or self.board[pos_below[1]][pos_below[0]] != 0:
                            coll = True
            if coll:
                bottom = False
                for i in range(4):
                    for j in range(4):
                        if self.falling_piece_shape[j][i] == 1:
                            pos = (i + self.falling_piece_pos[0], j + self.falling_piece_pos[1])
                            self.board[pos[1]][pos[0]] = 1
                            if pos[1] > (len(self.board) // 2):
                                bottom = True
                
                lines_cleared = self.resolve_lines()
                
                if lines_cleared > 0:
                    reward = (2 + lines_cleared) ** 2
                else:
                    self.apply_garbage(self.incoming_garbage)
                    self.incoming_garbage = 0

                if self.falling_piece_pos[1] == 0:
                    done = True
                    reward = -10
                
                self.falling_piece_pos = (np.random.randint(0, self.width - 3), 0)
                self.falling_piece_shape = np.rot90(self.piece_types[np.random.randint(0, len(self.piece_types))], k=np.random.randint(0, 4))

            else:
                self.falling_piece_pos = (self.falling_piece_pos[0], self.falling_piece_pos[1] + 1)

        piece = np.array(np.zeros((self.height, self.width)))
        
        for i in range(4):
            for j in range(4):
                if self.falling_piece_shape[j][i] == 1:
                    pos = (i + self.falling_piece_pos[0], j + self.falling_piece_pos[1])
                    piece[pos[1]][pos[0]] = 1
        
        timing_layer = np.zeros((self.height, self.width))
        
        if self.subframe == self.subframes - 2:
            timing_layer = np.ones((self.height, self.width))

        state = np.array([[
            self.board,
            piece,
            timing_layer
        ]])

        if self.time > self.cutoff:
            done = True

        return state, reward, done, { 'lines_cleared' : lines_cleared }
    
    def step(self, action):
        self.time = self.time + 1
        self.subframe = self.subframe + 1
        done = False
        reward = 0
        lines_cleared = 0
        
        dynamic_layer = np.array(np.zeros((self.height, self.width)), dtype=int)
        lbound, rbound, ubound, dbound = (self.width, 0, self.height, 0)

        for c in range(4):
            for r in range(4):
                if self.falling_piece_shape[r][c] == 1:
                    pos = (c + self.falling_piece_pos[0], r + self.falling_piece_pos[1])
                    dynamic_layer[pos[1]][pos[0]] = 1

                    if pos[0] < lbound: lbound = pos[0]
                    if pos[0] > rbound: rbound = pos[0]
                    if pos[1] < ubound: ubound = pos[1]
                    if pos[1] > dbound: dbound = pos[1]

        if action == Ris.LEFT:
            if lbound > 0:
                preview_layer = np.roll(dynamic_layer, -1)
                if not np.all(np.bitwise_xor(self.board, preview_layer)):
                    self.falling_piece_pos = (self.falling_piece_pos[0] - 1, self.falling_piece_pos[1])
        
        if action == Ris.RIGHT:
            if rbound < self.width - 1:
                preview_layer = np.roll(dynamic_layer, 1)
                if not np.all(np.bitwise_xor(self.board, preview_layer)):
                    self.falling_piece_pos = (self.falling_piece_pos[0] + 1, self.falling_piece_pos[1])
            
        if action == Ris.ROT:
            rotated = np.rot90(self.falling_piece_shape)
            coll = False
            for i in range(4):
                for j in range(4):
                    if not coll and rotated[j][i] == 1:
                        pos = (i + self.falling_piece_pos[0], j + self.falling_piece_pos[1])
                        if pos[0] not in range(0, len(self.board[0])) or \
                           pos[1] not in range(0, len(self.board)) or \
                           self.board[pos[1]][pos[0]] != 0:
                            coll = True
            if not coll:
                self.falling_piece_shape = rotated

        if self.subframe == self.subframes - 1:
            self.subframe = 0
            
            coll = False
            for i in range(4):
                for j in range(4):
                    if not coll and self.falling_piece_shape[j][i] == 1:
                        pos_below = (i + self.falling_piece_pos[0], j + self.falling_piece_pos[1] + 1)
                        if pos_below[1] >= len(self.board) or self.board[pos_below[1]][pos_below[0]] != 0:
                            coll = True
            if coll:
                bottom = False
                for i in range(4):
                    for j in range(4):
                        if self.falling_piece_shape[j][i] == 1:
                            pos = (i + self.falling_piece_pos[0], j + self.falling_piece_pos[1])
                            self.board[pos[1]][pos[0]] = 1
                            if pos[1] > (len(self.board) // 2):
                                bottom = True
                
                lines_cleared = self.resolve_lines()
                
                if lines_cleared > 0:
                    reward = (2 + lines_cleared) ** 2
                else:
                    self.apply_garbage(self.incoming_garbage)
                    self.incoming_garbage = 0

                if self.falling_piece_pos[1] == 0:
                    done = True
                    reward = -10
                
                self.falling_piece_pos = (np.random.randint(0, self.width - 3), 0)
                self.falling_piece_shape = np.rot90(self.piece_types[np.random.randint(0, len(self.piece_types))], k=np.random.randint(0, 4))

            else:
                self.falling_piece_pos = (self.falling_piece_pos[0], self.falling_piece_pos[1] + 1)

        piece = np.array(np.zeros((self.height, self.width)))
        
        for i in range(4):
            for j in range(4):
                if self.falling_piece_shape[j][i] == 1:
                    pos = (i + self.falling_piece_pos[0], j + self.falling_piece_pos[1])
                    piece[pos[1]][pos[0]] = 1
        
        timing_layer = np.zeros((self.height, self.width))
        
        if self.subframe == self.subframes - 2:
            timing_layer = np.ones((self.height, self.width))

        state = np.array([[
            self.board,
            piece,
            timing_layer
        ]])

        if self.time > self.cutoff:
            done = True

        return state, reward, done, { 'lines_cleared' : lines_cleared }

    def draw(self, screen, heatmap=None):
        
        # draw static pieces
        
        for i in range(len(self.board[0])):
            for j in range(len(self.board)):
                cell = pg.Rect(self.cell_size * i, self.cell_size * j, self.cell_size, self.cell_size)
                
                if self.board[j][i] == 1: 
                    pg.draw.rect(screen, (0, 100, 0), cell)
                    pg.draw.rect(screen, (0, 90, 0), cell, 1)
                else:
                    pg.draw.rect(screen, (64, 64, 64), cell)
                    pg.draw.rect(screen, (58, 58, 58), cell, 1)

        # draw falling piece

        for i in range(4):
            for j in range(4):
                cell = pg.Rect(self.cell_size * (i + self.falling_piece_pos[0]), self.cell_size * (j + self.falling_piece_pos[1]), self.cell_size, self.cell_size)
                if self.falling_piece_shape[j][i] == 1: 
                    pg.draw.rect(screen, (0, 120, 0), cell)
                    pg.draw.rect(screen, (0, 110, 0), cell, 1)
    
    def render(self, mode=''):
        self.draw(self.screen)    
        pg.display.flip()

    def play(self, framerate=30):
        clock = pg.time.Clock()
        while True:
            self.render()

            action = Ris.NOP
            events = pg.event.get()
            for event in events:
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_LEFT:
                        action = Ris.LEFT
                    if event.key == pg.K_RIGHT:
                        action = Ris.RIGHT
                    if event.key == pg.K_UP:
                        action = Ris.ROT
            state, reward, done, _ = self.step(action)

            if done: self.reset()
            clock.tick(int(framerate))

    piece_sets = {
        'lettris' : [
            [[0,0,0,0],
            [0,0,0,0],
            [0,1,1,0],
            [0,0,0,0]]
        ],

        'koktris' : [
            [[0,0,0,0],
            [0,0,1,0],
            [1,1,1,0],
            [0,0,0,0]], 
            
            [[0,0,0,0],
            [0,1,0,0],
            [0,1,1,1],
            [0,0,0,0]],
            
            [[0,0,0,0],
            [0,1,1,0],
            [0,1,1,0],
            [0,0,0,0]],

            [[0,0,0,0],
            [0,1,0,0],
            [1,1,1,0],
            [0,0,0,0]],

            [[0,0,0,0],
            [1,1,0,0],
            [0,1,1,0],
            [0,0,0,0]],

            [[0,0,0,0],
            [0,0,1,1],
            [0,1,1,0],
            [0,0,0,0]],

            [[0,0,0,0],
            [0,0,0,0],
            [1,1,1,1],
            [0,0,0,0]],
        ]
    }