import sys
import pygame as pg
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from ris import Ris

if '--novid' in sys.argv:
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"

def main():
    env = Ris()
    screen = pg.display.set_mode((env.cell_size * (len(env.board[0])), env.cell_size * len(env.board)))
    env.screen = screen

    model = PPO.load("ppo_snek")
    obs = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=False)
        obs, _, done, _ = env.step(action)

        env.render() # Comment out this call to train faster

        if done:
            obs = env.reset()

if __name__ == '__main__':
    main()
