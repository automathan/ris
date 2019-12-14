import numpy as np
import pygame as pg
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from vicero.algorithms.reinforce import Reinforce
from ris import Koktris

cell_size  = 32
framerate  = 32

def plot(history):
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(history)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy(), c='lightgray', linewidth=1)

        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), c='green')
            
        plt.pause(0.001)

env = Koktris(cell_size, height=22, width=9)

pg.init()
clock = pg.time.Clock()
screen = pg.display.set_mode((cell_size * len(env.board[0]), cell_size * len(env.board)))

env.screen = screen

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.conv = nn.Conv2d(2, 12, 3)
        self.conv2 = nn.Conv2d(12, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(540, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 5)

    def forward(self, x):
        x = F.relu(self.conv(x))
        #x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

poligrad = Reinforce(env, polinet=PolicyNet(), learning_rate=0.01, gamma=0.95, batch_size=5, plotter=plot)
poligrad.train(10000)

while True:
    env.draw(screen)
    pg.display.flip()
    action = Koktris.NOP
    events = pg.event.get()
    for event in events:
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_LEFT:
                action = Koktris.LEFT
            if event.key == pg.K_RIGHT:
                action = Koktris.RIGHT
            if event.key == pg.K_UP:
                action = Koktris.ROT
    state, reward, done, _ = env.step(action)

    if done: env.reset()
    clock.tick(int(framerate))