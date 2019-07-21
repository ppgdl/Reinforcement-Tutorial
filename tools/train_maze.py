import argparse
from models.Maze.config import *
from models.Maze.QLearning import QLeaning
from utils.Maze.enviroment import *
from utils.Maze.agent import *


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Reinforcement baseline.')
    parser.add_argument('--agent_name', dest='agent_name', help='Agent Name',
                        default='buheng', type=str)
    parser.add_argument('--env_name', dest='env_name', help='Env Name',
                        default='PPGDL', type=str)

    args = parser.parse_args()

    return args


def train():
    for episode in range(100):
        state = env.reset()

        while True:
            # fresh env
            env.render()

            # choose action
            state_index = env.state_transform(state)
            action, action_name = agent.choice_action(state_index)

            # step action and get reward
            state_next, reward, terminate = env.step(action)
            state_next_index = env.state_transform(state_next)

            # update Q-table
            brain.update_q_table(state_index, action, reward, state_next_index)

            state = state_next

            if terminate:
                break

    print('Game Over')
    env.destroy()


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args)
    env = Maze()
    setattr(config, 'n_columns', env.n_actions)

    brain = QLeaning('PPGDL', config)

    agent = Agent('buheng', env, brain, config)

    env.after(100, train)
    env.mainloop()