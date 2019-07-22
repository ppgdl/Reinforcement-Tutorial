import tensorflow as tf
import argparse
import os
import time

from models.DQN.config import get_config
from utils.Atari.enviroment import Enviroment
from models.DQN.DQN import DQN
from utils.Atari.agent import Agent

ROOT = os.path.join(os.getcwd(), '..')

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Reinforcement baseline.')
    parser.add_argument('--gpu_index', dest='gpu_index', help='GPU Index',
                        default=0, type=int)
    parser.add_argument('--gpu_fraction', dest='gpu_fraction', help='GPU fraction',
                        default=0.7, type=float)
    parser.add_argument('--batch_size', dest='batch_size', help='batch_size',
                        default=32, type=int)
    parser.add_argument('--num_epoch', dest='num_epoch', help='num_epoch',
                        default=10000, type=int)
    parser.add_argument('--agent_name', dest='agent_name', help='Agent Name',
                        default='Breakout-v0', type=str)
    parser.add_argument('--env_name', dest='env_name', help='Env Name',
                        default='PPGDL', type=str)
    parser.add_argument('--train', dest='train', help='whether train',
                        default=True, type=bool)
    parser.add_argument('--log', dest='log', help='log_path',
                        default='log', type=str)
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint_path',
                        default='checkpoint', type=str)

    args = parser.parse_args()

    return args


def update_log_checkpotin_path(config):
    prefix = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_path = os.path.join(ROOT, config.log, prefix)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    setattr(config, 'log', log_path)
    checkpoint_path = os.path.join(ROOT, config.checkpoint, prefix)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    setattr(config, 'checkout', checkpoint_path)


if __name__ == '__main__':
    args = parse_args()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
        config = get_config(args)

        # setup log path and checkpoint path
        update_log_checkpotin_path(config)

        env = Enviroment("PPGDL", config)
        brain = DQN("buheng", config, sess)
        agent = Agent(config.agent_name, config, env)

        if config.train:
            DQN.train()
        else:
            DQN.eval()

    print('Training Done!')

