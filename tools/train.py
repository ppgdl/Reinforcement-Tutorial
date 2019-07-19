import argparse

from models.QLearning.config import *
from models.QLearning.QLearning import *


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Reinforcement baseline.')
    parser.add_argument('--agent_name', dest='agent_name', help='Agent Name',
                        default='buheng', type=str)
    parser.add_argument('--env_name', dest='env_name', help='Env Name',
                        default='PPGDL', type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args)

    Q = QLearning(config.agent_name, config.env_name, config)

    for epoch_i in range(config.max_epoch):
        step_counter = 0
        state = 0
        is_terminated = False

        Q.env.update_env(state, epoch_i, step_counter)

        while not is_terminated:
            action = Q.agent.choice_action(state, Q.q_table)
            state_next, reward = Q.env.get_feedback(state, action)
            action_index = Q.agent.actions_choice.index(action)
            q_predict = Q.q_table[state, action_index]

            if state_next != 'terminal':
                q_target = reward + config.lambbds * np.max(Q.q_table[state_next, :])
            else:
                q_target = reward
                is_terminated = True

            Q.q_table[state, action_index] += config.alpha * (q_target - q_predict)

            state = state_next
            Q.env.update_env(state, epoch_i, step_counter)
            step_counter += 1

        print('\n')
        print(Q.q_table)
