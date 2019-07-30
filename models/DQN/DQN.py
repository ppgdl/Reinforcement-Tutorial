import tensorflow as tf

from utils.Atari.tf_utils import *


class DQN(object):
    def __init__(self, name, config, sess):
        self.name = name
        self.config = config
        self.sess = sess

        # predict network
        self.predict_input_pl = None
        self.predict_input = None
        self.predicts = None
        self.predict_action = None
        self.predict_action_pl = None
        # predict_target_pl = reward + Q_max(start_next)
        self.predict_target_pl = None
        self.learning_rate_step = None
        self.predict_extra = []
        self.optimizer = None
        self.train_op = None
        self.update_network_ops = None
        self.loss = None
        self.losses = None
        self.gradients = None
        self.delta = None
        self.merged = None


        # target network
        self.target_input_pl = None
        self.target_input = None
        self.targets = None
        self.target_action = None
        self.target_extra = []

        self.build_dqn(sess)

    def build_dqn(self, sess):
        self.create_predict_network()
        self.create_target_network()
        self.update_network_operation()
        sess.run(tf.global_variables_initializer())

    def create_predict_network(self):
        # build network
        height = self.config.screen_height
        width = self.config.screen_width
        channels = self.config.history_length
        self.predict_input_pl = tf.placeholder(tf.uint8, [None, height, width, channels], name='predict_input')
        self.predict_input = tf.to_float(self.predict_input_pl) / 255.0
        self.predict_target_pl = tf.placeholder(tf.float32, [None], name='predict_target')
        # real reward [batch_size, action_number]
        self.predict_action_pl = tf.placeholder(tf.float32, [None, 4], name='predict_action')
        self.predicts, self.predict_action, self.predict_tensor_list = self._build_network('predict', self.predict_input, trainable=True)
        self.learning_rate_step = tf.placeholder(tf.int64, None, name='learning_rate_step')

        # cal loss
        predict_rewards = self.predicts * self.predict_action_pl
        predict_rewards = tf.reduce_sum(predict_rewards, axis=1)
        self.predict_tensor_list.append(predict_rewards)
        self.delta = self.predict_target_pl - predict_rewards
        self.losses = tf.squared_difference(self.predict_target_pl, predict_rewards)
        self.loss = tf.reduce_mean(self.losses)
        self.gradients = tf.gradients(self.loss, tf.trainable_variables()[0:10])

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("delta", tf.reduce_mean(self.delta))
        tf.summary.scalar("p_max_rewards", tf.reduce_max(self.predicts))
        tf.summary.scalar("p_r_max_rewards", tf.reduce_max(predict_rewards))
        tf.summary.scalar("p_r_min_rewards", tf.reduce_min(predict_rewards))
        tf.summary.scalar("t_max_rewards", tf.reduce_max(self.predict_target_pl))
        tf.summary.scalar("t_min_rewards", tf.reduce_min(self.predict_target_pl))
        tf.summary.histogram("loss_hist", self.losses)
        tf.summary.histogram("q_value_hist", self.predicts)

        self.merged = tf.summary.merge_all()

        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.00, 1e-06)
        self.train_op = self.optimizer.minimize(self.loss)

    def create_target_network(self):
        # build network
        height = self.config.screen_height
        width = self.config.screen_width
        channels = self.config.history_length
        self.target_input_pl = tf.placeholder(tf.float32, [None, height, width, channels], name='target_input')
        self.target_input = self.target_input_pl / 255.0
        # output reward
        self.targets, self.target_action, _ = self._build_network('target', self.target_input, trainable=False)

    def _build_network(self, name, input_pl, trainable=True):
        with tf.variable_scope(name):
            conv1 = tf.contrib.layers.conv2d(input_pl, 32, 8, 4, activation_fn=tf.nn.relu,
                                             trainable=trainable)
            conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu,
                                             trainable=trainable)
            conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu,
                                             trainable=trainable)

            # full connection
            flattened = tf.contrib.layers.flatten(conv3)
            fc1 = tf.contrib.layers.fully_connected(flattened, 512,
                                                    trainable=trainable)
            predictions = tf.contrib.layers.fully_connected(fc1,
                                                            self.config.action_length,
                                                            trainable=trainable)
            predict_action = tf.argmax(predictions, dimension=1)

            tensor_list = [conv1, conv2, conv3, fc1, predictions]

            return predictions, predict_action, tensor_list

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
    def update_network_operation(self):
        predict_params = [t for t in tf.trainable_variables() if t.name.startswith("predict")]
        target_params = [t for t in tf.global_variables() if t.name.startswith("target")]

        self.update_network_ops = []
        for target_tensor in target_params:
            for predict_tensor in predict_params:
                if target_tensor.name.replace('target', 'predict') == predict_tensor.name:
                    op = target_tensor.assign(predict_tensor)
                    self.update_network_ops.append(op)

    def update_target_network(self):

        self.sess.run(self.update_network_ops)