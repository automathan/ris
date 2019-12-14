import numpy as np
import pygame as pg

piece_types = [
    [[0,0,0,0],
     [0,0,0,0],
     [0,1,1,0],
     [0,0,0,0]]
]

class Koktris:
    NOP, LEFT, RIGHT, DOWN, ROT = range(5)

    def __init__(self, scale, width=8, height=16):
        class ActionSpace:
            def __init__(self):
                self.n = 5
        self.action_space = ActionSpace()
        board = np.zeros((height, width))
        self.width = width
        self.height = height
        
        self.time = 0
        self.cutoff = 4000

        self.board = np.array(board)
        self.size = len(board)
        self.cell_size = scale
        self.falling_piece_pos = (np.random.randint(0, self.width - 3), 0)
        self.falling_piece_shape = piece_types[np.random.randint(0, len(piece_types))]
        self.subframe = 0
        self.subframes = 5
        self.screen = None
    
    def reset(self):
        board = np.zeros((self.height, self.width))
        self.board = np.array(board)
        self.falling_piece_pos = (np.random.randint(0, self.width - 3), 0)
        self.falling_piece_shape = piece_types[np.random.randint(0, len(piece_types))]
        self.subframe = 0

        piece = np.array(np.zeros((self.height, self.width)))
        for i in range(4):
            for j in range(4):
                if self.falling_piece_shape[j][i] == 1:
                    pos = (i + self.falling_piece_pos[0], j + self.falling_piece_pos[1])
                    piece[pos[1]][pos[0]] = 1
        self.time = 0     
        state = np.array([[
            self.board,
            piece
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
        
    def step(self, action):
        self.time = self.time + 1
        self.subframe = self.subframe + 1
        done = False
        reward = 0

        if action == Koktris.LEFT:
            coll = False
            for i in range(4):
                for j in range(4):
                    if not coll and self.falling_piece_shape[j][i] == 1:
                        pos_left = (i + self.falling_piece_pos[0] - 1, j + self.falling_piece_pos[1])
                        if pos_left[0] < 0 or self.board[pos_left[1]][pos_left[0]] != 0:
                            coll = True
            if not coll:
                self.falling_piece_pos = (self.falling_piece_pos[0] - 1, self.falling_piece_pos[1])
        
        if action == Koktris.RIGHT:
            coll = False
            for i in range(4):
                for j in range(4):
                    if not coll and self.falling_piece_shape[j][i] == 1:
                        pos_left = (i + self.falling_piece_pos[0] + 1, j + self.falling_piece_pos[1])
                        if pos_left[0] >= len(self.board[0]) or self.board[pos_left[1]][pos_left[0]] != 0:
                            coll = True
            if not coll:
                self.falling_piece_pos = (self.falling_piece_pos[0] + 1, self.falling_piece_pos[1])
            
        if action == Koktris.ROT:
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
                
                #reward = 1
                #if bottom:
                #    reward = 1
                #else:
                #    reward = -.01
                
                points = self.resolve_lines()
                if points > 0:
                    reward = (2 + points) ** 2
                    
                if self.falling_piece_pos[1] == 0:
                    done = True
                    reward = -1

                self.falling_piece_pos = (np.random.randint(0, self.width - 3), 0)
                self.falling_piece_shape = piece_types[np.random.randint(0, len(piece_types))]
            
            else:
                self.falling_piece_pos = (self.falling_piece_pos[0], self.falling_piece_pos[1] + 1)

        piece = np.array(np.zeros((self.height, self.width)))
        for i in range(4):
            for j in range(4):
                if self.falling_piece_shape[j][i] == 1:
                    pos = (i + self.falling_piece_pos[0], j + self.falling_piece_pos[1])
                    piece[pos[1]][pos[0]] = 1
                
              
        state = np.array([[
            self.board,
            piece
        ]])

        if self.time > self.cutoff:
            done = True

        return state, reward, done, {}
    

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

