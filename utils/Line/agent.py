import numpy as np
np.random.seed(2)


class Agent(object):

    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.actions_choice = self.action_space()
        self.action_space()

    def action_space(self):

        return self.config.action_name

    @property
    def actions(self):
        return self.action_space()

    def choice_action(self, state, q_table):
        state_actions = q_table[state, :]
        if sum(state_actions) == 0 or np.random.uniform() > self.config.epsilon:
            action_name = np.random.choice(self.actions_choice)
        else:
            action_index = np.argmax(state_actions)
            action_name = self.actions_choice[action_index]

        return action_name