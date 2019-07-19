import time


class Enviroment(object):

    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.n_states = self.init_env()

    def init_env(self):
        return self.config.n_state

    def get_feedback(self, state, action):
        if action == "right":
            if state == self.n_states - 2:
                state_next = 'terminal'
                reward = 1
            else:
                state_next = state + 1
                reward = 0
        else:
            reward = 0
            if state == 0:
                state_next = state
            else:
                state_next = state - 1

        return state_next, reward

    def update_env(self, state, epoch, step_counter):
        env_list = ['-']*(self.config.n_state-1) + ['T']
        if state == 'terminal':
            line = 'Epoch {:}, total_steps: {}'.format(epoch + 1, step_counter)
            print(line)
            time.sleep(self.config.fresh_update)
        else:
            env_list[state] = 'o'
            line = ''.join(env_list)
            print('\r{}'.format(line), end='')
            time.sleep(0.3)

    @property
    def env_states(self):
        return self.n_states
