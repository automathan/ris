import pygame as pg
from ris import Ris

cell_size = 24

class Riskrig:
    def __init__(self, cell_size, width, height):
        self.ris_a = Ris(cell_size, height=height, width=width, piece_set='lettris')
        self.ris_b = Ris(cell_size, height=height, width=width, piece_set='lettris')
        self.screen = None
        self.garb_a = 0
        self.garb_b = 0

    def step(self, action_a, action_b):
        _, _, done_a, info_a = self.ris_a.step(action_a)
        _, _, done_b, info_b = self.ris_b.step(action_b)
        
        self.garb_b += info_a['lines_cleared']
        self.garb_a += info_b['lines_cleared']
        
        return (done_a or done_b)

    def reset(self):
        self.ris_a.reset()
        self.ris_b.reset()
        #self.garb_a = 0
        #self.garb_b = 0
        
    def render(self):
        screen_a = pg.Surface((cell_size * len(self.ris_a.board[0]), cell_size * len(self.ris_a.board)))
        self.ris_a.draw(screen_a)
        self.screen.blit(screen_a, (0, 0))

        screen_b = pg.Surface((cell_size * len(self.ris_b.board[0]), cell_size * len(self.ris_b.board)))
        self.ris_b.draw(screen_b)
        self.screen.blit(screen_b, (cell_size * (1 + len(self.ris_b.board[0])), 0))
        
        for i in range(self.garb_a):
            cell = pg.Rect(cell_size * len(self.ris_a.board[0]), cell_size * (len(self.ris_a.board) - i), cell_size, cell_size)
            pg.draw.rect(self.screen, (100, 0, 0), cell)
            pg.draw.rect(self.screen, (90, 0, 0), cell, 1)

        for i in range(self.garb_b):
            cell = pg.Rect(cell_size + 2 * cell_size * len(self.ris_a.board[0]), cell_size * (len(self.ris_a.board) - i), cell_size, cell_size)
            pg.draw.rect(self.screen, (100, 0, 0), cell)
            pg.draw.rect(self.screen, (90, 0, 0), cell, 1)           

rk = Riskrig(24, 7, 14)

env = rk.ris_a

screen = pg.display.set_mode((2 * cell_size + 2 * cell_size * (len(env.board[0])), cell_size * len(env.board)))

rk.screen = screen

while True:
    done = rk.step(rk.ris_a.action_space.sample(), rk.ris_b.action_space.sample())
    if done: rk.reset()
    rk.render()
    pg.display.flip()
