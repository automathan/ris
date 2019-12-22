import numpy as np
import pygame as pg
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from vicero.algorithms.deepqlearning import DQN
import sys
sys.path.insert(0, '..')
from ris import Ris

from warnings import filterwarnings
filterwarnings('ignore')

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 5)
        
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

for i in range(12):
    env = Ris(32, height=10, width=5)
    state = env.reset()
    qnet = torch.load('qnet_{}.dqn'.format((i + 1) * 100))

    pg.init()
    screen = pg.display.set_mode((32 * len(env.board[0]), 32 * len(env.board)))
    env.screen = screen
    clock = pg.time.Clock()

    score = 0

    done = False
    while not done:
        state = torch.from_numpy(state)
        outputs = qnet(state)
        action =  outputs.max(0)[1].numpy()
        state, reward, done, _ = env.step(action)
        score += reward
        env.render()

        clock.tick(50)
        if done:
            print('done, i={}, score={}'.format(i, score))
            score = 0
            state = env.reset()
