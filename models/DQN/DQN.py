import tensorflow as tf

from utils.Atari.tf_utils import *


class DQN(object):
    def __init__(self, name, config, sess):
        self.name = name
        self.config = config
        self.sess = sess

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
        self.optimizer = None
        self.train_op = None
        self.loss = None
        self.delta = None

        # target network
        self.target_input_pl = None
        self.targets = None
        self.target_action = None
        self.target_extra = []

        self.build_dqn(sess)

    def build_dqn(self, sess):
        #with tf.Graph().as_default():
        self.create_predict_network()
        self.create_target_network()
        sess.run(tf.global_variables_initializer())

    def create_predict_network(self):
        # build network
        height = self.config.screen_height
        width = self.config.screen_width
        channels = self.config.history_length
        self.predict_input_pl = tf.placeholder(tf.float32, [None, height, width, channels], name='predict_input')
        self.predict_target_pl = tf.placeholder(tf.float32, [None], name='predict_target')
        # real reward [batch_size, action_number]
        self.predict_action_pl = tf.placeholder(tf.float32, [None, 4], name='predict_action')
        self.predicts, self.predict_action = self._build_network('predict', self.predict_input_pl, trainable=True)
        self.learning_rate_step = tf.placeholder(tf.int64, None, name='learning_rate_step')

        # cal loss
        predict_rewards = self.predicts * self.predict_action_pl
        predict_rewards = tf.reduce_sum(predict_rewards, axis=1)
        self.delta = self.predict_target_pl - predict_rewards
        losses = squared_loss(self.delta)
        self.loss = tf.reduce_mean(losses)

        # optimizer
        self.learning_rate_op = tf.maximum(self.config.learning_rate_minimum,
                                           tf.train.exponential_decay(
                                               self.config.learning_rate,
                                               self.learning_rate_step,
                                               self.config.learning_rate_decay_step,
                                               self.config.learning_rate_decay,
                                               staircase=True))

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01)
        self.train_op = self.optimizer.minimize(self.loss)

    def create_target_network(self):
        # build network
        height = self.config.screen_height
        width = self.config.screen_width
        channels = self.config.history_length
        self.target_input_pl = tf.placeholder(tf.float32, [None, height, width, channels], name='target_input')
        # output reward
        self.targets, self.target_action = self._build_network('target', self.target_input_pl, trainable=False)

    def _build_network(self, name, input_pl, trainable=True):
        with tf.variable_scope(name):
            # conv
            conv1 = tf.contrib.layers.conv2d(input_pl, 32, 8, 4, activation_fn=tf.nn.relu,
                                             weights_initializer=tf.truncated_normal_initializer(0, 0.02),
                                             biases_initializer=tf.constant_initializer(0.0),
                                             trainable=trainable)
            conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu,
                                             weights_initializer=tf.truncated_normal_initializer(0, 0.02),
                                             biases_initializer=tf.constant_initializer(0.0),
                                             trainable=trainable)
            conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu,
                                             weights_initializer=tf.truncated_normal_initializer(0, 0.02),
                                             biases_initializer=tf.constant_initializer(0.0),
                                             trainable=trainable)

            # full connection
            flattened = tf.contrib.layers.flatten(conv3)
            fc1 = tf.contrib.layers.fully_connected(flattened, 512,
                                                    weights_initializer=tf.truncated_normal_initializer(0, 0.02),
                                                    biases_initializer=tf.constant_initializer(0.0),
                                                    trainable=trainable)
            predictions = tf.contrib.layers.fully_connected(fc1,
                                                            self.config.action_length,
                                                            weights_initializer=tf.truncated_normal_initializer(0, 0.02),
                                                            biases_initializer=tf.constant_initializer(0.0),
                                                            trainable=trainable)
            predict_action = tf.argmax(predictions, dimension=1)

            return predictions, predict_action

    def forward_predict_network(self, input_pl):
        feed_dict = {self.predict_input_pl: input_pl}
        v_predicts, v_predict_action = self.sess.run([self.predicts, self.predict_action], feed_dict=feed_dict)

        return v_predicts, v_predict_action

    def forward_target_network(self, input_pl):
        feed_dict = {self.target_input_pl: input_pl}
        v_targets, v_target_action  = self.sess.run([self.targets, self.target_action], feed_dict = feed_dict)

        return v_targets, v_target_action

    """
    1. copy rewards from predict network to target network
    """
    def update_target_network(self):
        predict_params = [t for t in tf.trainable_variables() if t.name.startswith("predict")]
        target_params = [t for t in tf.global_variables() if t.name.startswith("target")]

        update_ops = []
        for target_tensor in target_params:
            for predict_tensor in predict_params:
                if target_tensor.name.replace('target', 'predict') == predict_tensor.name:
                    op = target_tensor.assign(predict_tensor)
                    update_ops.append(op)

        self.sess.run(update_ops)