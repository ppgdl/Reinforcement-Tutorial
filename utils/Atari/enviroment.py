import gym
import time

from utils.Atari.tf_utils import *


class Enviroment(object):

    def __init__(self, name, config):
        self.name = name
        self.config = config

        self.env = gym.make(config.agent_name)
        self.env.reset()

        self.screen_width = config.screen_width
        self.screen_height = config.screen_height
        self.action_repeat = config.action_repeat
        self.random_start = config.random_start

        self.display = config.display

    def new_game(self):
        state = self.env.reset()
        state = resizeimage(state, self.screen_height, self.screen_width)

        return state

    # fresh screen
    def render(self):
        if self.display:
            self.env.render()

    def step(self, action):
        state_next, reward, terminal, _ = self.env.step(action)
        state_next = resizeimage(state_next, self.screen_height, self.screen_width)
        self.render()

        return state_next, reward, terminal

    @property
    def lives(self):
        return self.env.ale.lives()

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


def test():
    from models.DQN.config import get_config
    args = get_config(dict())

    """
    # init env and reset env
    # env = Enviroment('Breakout-v0', args)
    env = gym.make("Breakout-v0")
    env.reset()
    env.step(3)
    env.step(3)
    env.render()

    for i in range(10):
        screen, rewards, terminal, _ = env.step(0)
        screen = resizeimage(screen, 84 * 4, 84 * 4)

        env.render()
        for j in range(3):
            screen, rewards, terminal, _ = env.step(2)
            env.render()
            print(3)
            screen = resizeimage(screen, 84 * 4, 84 * 4)
            save_image(screen, args, str(i) + '-' + str(j) + '-' + str(2))
            if terminal:
                print("Reset Game")
                _ = env.reset()
        for j in range(3):
            screen, rewards, terminal, _ = env.step(3)
            env.render()
            print(3)
            screen = resizeimage(screen, 84 * 4, 84 * 4)
            save_image(screen, args, str(i) + '-' + str(3 + j) + '-' + str(3))
            if terminal:
                print("Reset Game")
                _ = env.reset()
        for j in range(60):
            screen, rewards, terminal, _ = env.step(1)
            env.render()
            #time.sleep(1)
            print(1)
            image_name = str(i) + '-' + str(6 + j) + '-' + str(1)
            if terminal:
                image_name = str(i) + '-' + str(6 + j) + '-' + str(1) + '-terminal'
            screen = resizeimage(screen, 84 * 4, 84 * 4)
            save_image(screen, args, image_name)
            if terminal:
                print("Reset Game")
                _ = env.reset()
        """
    """
    # init env and reset env
    env = Enviroment('Breakout-v0', args)
    env.step(3)
    env.step(3)
    env.render()

    for i in range(10):
        screen, rewards, terminal = env.step(0)

        env.render()
        for j in range(3):
            screen, rewards, terminal = env.step(2)
            env.render()
            print(3)
            save_image(screen, args, str(i) + '-' + str(j) + '-' + str(2))
            if terminal:
                print("Reset Game")
                _ = env.new_game()
        for j in range(3):
            screen, rewards, terminal = env.step(3)
            env.render()
            print(3)
            save_image(screen, args, str(i) + '-' + str(3 + j) + '-' + str(3))
            if terminal:
                print("Reset Game")
                _ = env.new_game()
        for j in range(60):
            screen, rewards, terminal = env.step(1)
            env.render()
            # time.sleep(1)
            print(1)
            image_name = str(i) + '-' + str(6 + j) + '-' + str(1)
            if terminal:
                image_name = str(i) + '-' + str(6 + j) + '-' + str(1) + '-terminal'
            save_image(screen, args, image_name)
            if terminal:
                print("Reset Game")
                _ = env.new_game()
    """

    action_list = [1, 1, 1, 3, 3, 2, 1, 3, 1, 3, 3, 3, 1, 0, 2, 2, 1, 2, 2, 2, 3, 3, 0, 0, 3]
    env = gym.make("Breakout-v0")
    state = env.reset()

    for i in range(len(action_list)):
        action = action_list[i]
        screen_next, rewards, terminal, _ = env.step(int(action))
        height, width, depth = screen_next.shape
        env.render()
        image = np.zeros((height * 2, width, depth))
        image[0:height, :, :] = state
        image[height:height * 2, :, :] = screen_next
        image = image.astype(np.uint8)
        save_image(image, args, str(i) + '-' + str(action))
        if terminal:
            print("Reset Game")
            state = env.reset()
        else:
            state = screen_next