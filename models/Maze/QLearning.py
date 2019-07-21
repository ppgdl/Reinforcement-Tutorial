import numpy as np


class QLeaning(object):
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.q_table = self.create_q_table()

    def create_q_table(self):
        rows = self.config.n_rows
        columns = self.config.n_columns

        return np.zeros((rows, columns))

    def check_state_exists(self, state):
        """
        check whether state exists in q_table, if not exists, update q_table
        :param state:
        :return:
        """

        q_table = self.q_table
        q_table_y, q_table_x = q_table.shape
        if state < q_table_y:
            pass
        else:
            new_q_table = np.zeros((state, q_table_x))
            for i in range(q_table_y):
                for j in range(q_table_x):
                    new_q_table[i, j] = q_table[i, j]

            self.q_table = new_q_table

    def update_q_table(self, state, action, reward, state_next):
        print("state: {} state_next: {}".format(state, state_next))
        if state_next != 'terminal':
            self.check_state_exists(state_next)

        q_predict = self.q_table[state, action]
        if state_next != 'terminal':
            q_target = reward + self.config.lambbds * np.max(self.q_table[state_next, :])
        else:
            q_target = reward

        self.q_table[state, action] += self.config.lr * (q_target - q_predict)
