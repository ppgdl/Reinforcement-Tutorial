import gym
import time

from utils.Atari.tf_utils import *


class Enviroment(object):

    def __init__(self, name, config):
        self.name = name
        self.config = config

        self.env = gym.make(config.env_name)
        self.env.reset()

        self.screen_width = config.screen_width
        self.screen_height = config.screen_height
        self.action_repeat = config.action_repeat
        self.random_start = config.random_start

        self.display = config.display

        self._screen = None
        self.reward = 0
        self.terminal = True

    def new_game(self):
        if self.lives == 0:
            self._screen = self.env.reset()
        self._step(0)
        self.render()

        return self.screen, 0, 0, self.terminal

    # fresh screen
    def render(self):
        if self.display:
            self.env.render()

    def _step(self, action):
        self._screen, self.reward, self.terminal, _ = self.env.step(action)

    def _random_step(self):
        action = self.env.action_space.sample()
        print('action: {}'.format(action))
        self._step(action)

    @property
    def screen(self):
        return resizeimage(self._screen, self.screen_height, self.screen_width)

    @property
    def lives(self):
        return self.env.ale.lives()

    @property
    def state(self):
        return self.screen, self.reward, self.terminal

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def action_names(self):
        """
        0: None
        1: FIRE: start play game
        2: RIGHT
        3: LEFT
        :return:
        """
        return self.env.action_space.get_action_meanings()


if __name__ == '__main__':
    from models.DQN.config import get_config
    args = get_config(dict())

    # init env and reset env
    env = Enviroment('Breakout-v0', args)

    for i in range(1000):
        env._step(1)
        env.render()
        for j in range(5):
            env._step(3)
            print(3)
            time.sleep(1)
            env.render()
            save_image(env.screen, env.config, j)
        for j in range(5):
            env._step(2)
            print(2)
            time.sleep(1)
            env.render()
            save_image(env.screen, env.config, 5 + j)
        for j in range(5):
            env._step(0)
            print(0)
            time.sleep(1)
            env.render()
            save_image(env.screen, env.config, 10 + j)