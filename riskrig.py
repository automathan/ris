import pygame as pg
from ris import Ris
import numpy as np

#cell_size = 24

class Riskrig:
    def __init__(self, cell_size, width, height):
        self.ris_a = Ris(cell_size, height=height, width=width)#, piece_set='koktris')
        self.ris_b = Ris(cell_size, height=height, width=width)#, piece_set='koktris')
        
        self.board = self.ris_a.board
        self.action_space = self.ris_a.action_space

        self.screen = None
        self.cell_size = cell_size
        
    def step(self, action):
        state_a, reward_a, done_a, info_a = self.ris_a.step(action)
        state_b, reward_b, done_b, info_b = self.ris_b.step(self.ris_b.action_space.sample())
        
        self.ris_b.incoming_garbage += info_a['lines_cleared']
        self.ris_a.incoming_garbage += info_b['lines_cleared']
        
        multi_state_a = np.array([np.vstack((state_a[0][:3], state_b[0]))])
        multi_state_b = np.array([np.vstack((state_b[0][:3], state_a[0]))])
        
        return multi_state_a, reward_a, done_a or done_b, {}

    def reset(self):
        state_a = self.ris_a.reset()
        state_b = self.ris_b.reset()
        
        multi_state_a = np.array([np.vstack((state_a[0][:3], state_b[0]))])
        multi_state_b = np.array([np.vstack((state_b[0][:3], state_a[0]))])
        
        return multi_state_a

    def render(self):
        cell_size = self.cell_size
        self.screen.fill((0,0,0))

        screen_a = pg.Surface((cell_size * len(self.ris_a.board[0]), cell_size * len(self.ris_a.board)))
        self.ris_a.draw(screen_a)
        self.screen.blit(screen_a, (0, 0))

        screen_b = pg.Surface((cell_size * len(self.ris_b.board[0]), cell_size * len(self.ris_b.board)))
        self.ris_b.draw(screen_b)
        self.screen.blit(screen_b, (cell_size * (1 + len(self.ris_b.board[0])), 0))
        
        for i in range(self.ris_a.incoming_garbage):
            cell = pg.Rect(cell_size * len(self.ris_a.board[0]), cell_size * (len(self.ris_a.board) - 1 - i), cell_size, cell_size)
            pg.draw.rect(self.screen, (100, 0, 0), cell)
            pg.draw.rect(self.screen, (90, 0, 0), cell, 1)

        for i in range(self.ris_b.incoming_garbage):
            cell = pg.Rect(cell_size + 2 * cell_size * len(self.ris_a.board[0]), cell_size * (len(self.ris_a.board) - 1 - i), cell_size, cell_size)
            pg.draw.rect(self.screen, (100, 0, 0), cell)
            pg.draw.rect(self.screen, (90, 0, 0), cell, 1)     

        pg.display.flip()      

if __name__ == "__main__":
    cell_size = 24
    
    rk = Riskrig(cell_size, 7, 14)

    env = rk.ris_a
    
    screen = pg.display.set_mode((2 * cell_size + 2 * cell_size * (len(env.board[0])), cell_size * len(env.board)))

    rk.screen = screen
    clock = pg.time.Clock()

    while True:
        _,_,done,_ = rk.step(rk.ris_a.action_space.sample())
        if done: rk.reset()
        rk.render()
        pg.display.flip()
        #clock.tick(30)