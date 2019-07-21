import numpy as np
import random


class Agent(object):

    def __init__(self, name, env, learning, config):
        self.name = name
        self.env = env
        self.config = config
        self.learning = learning

    def choice_action(self, state):
        self.learning.check_state_exists(state)

        if np.random.uniform() < self.config.epsilon:
            state_action = self.learning.q_table[state, :]
            action_index = np.random.choice(np.where(state_action == np.max(state_action))[0])
        else:
            action_index = random.randint(0, self.env.n_actions - 1)
        action_name = self.env.action_space[action_index]

        return action_index, action_name
