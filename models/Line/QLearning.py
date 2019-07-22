import numpy as np

from utils.Line.agent import Agent
from utils.Line.enviroment import Enviroment


class QLearning(object):
    def __init__(self, agent_name, env_name, config):
        self.agent = Agent(agent_name, config)
        self.env = Enviroment(env_name, config)
        self.config = config
        self.q_table = self.build_q_table()

    def build_q_table(self):
        rows = len(self.agent.actions)
        columns = self.env.env_states

        return np.zeros((columns, rows))



