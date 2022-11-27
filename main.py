import sys
import pygame as pg
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from ris import Ris

if '--novid' in sys.argv:
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"

def main():
    env = Ris()
    env = make_vec_env(Ris, n_envs=2)
    #agent = Agent(env)
    
    new_logger = configure('./results', ["stdout", "csv", "json", "log"])
    model = PPO("MlpPolicy", env, verbose=1)
    model.set_logger(new_logger)
    model.learn(total_timesteps=1500000, log_interval=4)
    model.save("ppo_koktris")

    total_len = 0
    num_episodes = 0
    
    env = Ris()
    screen = pg.display.set_mode((env.cell_size * (len(env.board[0])), env.cell_size * len(env.board)))
    env.screen = screen
    obs = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=False)
        obs, _, done, _ = env.step(action)
        
        env.render() # Comment out this call to train faster

        if done:
            obs = env.reset()
            
if __name__ == '__main__':
    main()
