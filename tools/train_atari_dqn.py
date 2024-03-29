import tensorflow as tf
import argparse
import os
import time
import sys
ROOT = os.path.join(os.getcwd(), '..')
sys.path.append(ROOT)

from models.DQN.config import get_config
from utils.Atari.enviroment import Enviroment
from models.DQN.DQN import DQN
from utils.Atari.agent import Agent



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Reinforcement baseline.')
    parser.add_argument('--gpu_index', dest='gpu_index', help='GPU Index',
                        default=0, type=int)
    parser.add_argument('--gpu_fraction', dest='gpu_fraction', help='GPU fraction',
                        default=0.4, type=float)
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
    parser.add_argument('--max_epsilon_step', dest='max_epsilon_step', help='epsilon_rate',
                        default=500000, type=int)

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
    img_path = os.path.join(ROOT, 'log_image', prefix)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    setattr(config, 'imgpath', img_path)


if __name__ == '__main__':
    import os

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
        config = get_config(args)

        # setup log path and checkpoint path
        update_log_checkpotin_path(config)

        # init env
        env = Enviroment("PPGDL", config)
        # init DQN including network
        brain = DQN("buheng", config, sess)
        # init agent
        agent = Agent(config.agent_name, config, env, brain)

        if config.train:
            agent.train()
        else:
            agent.eval()

    print('Training Done!')

