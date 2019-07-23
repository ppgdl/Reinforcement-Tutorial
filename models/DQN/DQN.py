import tensorflow as tf

from utils.Atari.tf_utils import *


class DQN(object):
    def __init__(self, name, config, sess):
        self.name = name
        self.config = config
        self.sess = sess

        self.replay_buffer = None

        # predict network
        self.predict_input_pl = None
        self.predicts = None
        self.predict_action = None
        self.predict_action_pl = None
        # predict_target_pl = reward + Q_max(start_next)
        self.predict_target_pl = None
        self.learning_rate_step = None
        self.predict_extra = []
        self.learning_rate_op = None
        self.optim = None

        # target network
        self.target_input_pl = None
        self.targets = None
        self.loss = None
        self.target_extra = []

        self.build_dqn(sess)

    def build_dqn(self, sess):
        #with tf.Graph().as_default():
        self.create_predict_network()
        self.create_target_network()

    def create_predict_network(self):
        # build network
        batch_size = self.config.batch_size
        height = self.config.screen_height
        width = self.config.screen_width
        channels = self.config.history_length
        self.predict_input_pl = tf.placeholder(tf.float32, [batch_size, height, width, channels], name='predict_input')
        self.predict_action_pl = tf.placeholder(tf.int32, [batch_size], name='predict_actions')
        # real reward
        self.predict_target_pl = tf.placeholder(tf.float32, [batch_size], name='predict_target')
        self.predicts, self.predict_action = self._build_network('predict', self.predict_input_pl, trainable=True)
        self.learning_rate_step = tf.placeholder(tf.int64, None, name='learning_rate_step')

        # cal loss
        gather_indices = tf.range(self.config.batch_size) * tf.shape(self.predicts)[1] + self.predict_action_pl
        predict_rewards = tf.gather(tf.reshape(self.predicts, [-1]), gather_indices)
        delta = self.predict_target_pl - predict_rewards
        losses = squared_loss(delta)
        self.loss = tf.reduce_mean(losses)

        # optimizer
        self.learning_rate_op = tf.maximum(self.config.learning_rate_minimum,
                                           tf.train.exponential_decay(
                                               self.config.learning_rate,
                                               self.learning_rate_step,
                                               self.config.learning_rate_decay_step,
                                               self.config.learning_rate_decay,
                                               staircase=True))

        self.opti = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

    def create_target_network(self):
        # build network
        batch_size = self.config.batch_size
        height = self.config.screen_height
        width = self.config.screen_width
        channels = self.config.history_length
        self.target_input_pl = tf.placeholder(tf.float32, [batch_size, height, width, channels], name='target_input')
        # output reward
        self.targets, _ = self._build_network('target', self.target_input_pl, trainable=False)

    def _build_network(self, name, input_pl, trainable=True):
        with tf.variable_scope(name):
            # conv
            conv1 = tf.contrib.layers.conv2d(input_pl, 32, 8, 4, activation_fn=tf.nn.relu, trainable=trainable)
            conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu, trainable=trainable)
            conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu, trainable=trainable)

            # full connection
            flattened = tf.contrib.layers.flatten(conv3)
            fc1 = tf.contrib.layers.fully_connected(flattened, 512, trainable=trainable)
            predictions = tf.contrib.layers.fully_connected(fc1, self.config.action_length, trainable=trainable)
            predict_action = tf.argmax(predictions, dimension=1)

            return predictions, predict_action

    """
    1. copy rewards from predict network to target network
    """
    def update_target_network(self):
        pass

    def train(self):
        # replay buffer store


        # create predict_network

        # create target network

        pass
