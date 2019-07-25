import numpy as np
from tqdm import tqdm
import itertools
import random

from utils.Atari.tf_utils import *


class Agent(object):

    def __init__(self, name, config, env, brain):
        self.name = name
        self.config = config
        self.env = env
        self.brain = brain
        self.replay_buffer = []

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []

        self.state = None

    def action_choose(self, screen_state, epsilon):
        action_length = self.config.action_length
        norm_distribute = np.ones(action_length, dtype=np.float32) * epsilon / action_length
        screen_state = screen_state[np.newaxis, ...]
        v_predicts, v_predict_actions = self.brain.forward_predict_network(screen_state)
        norm_distribute[v_predict_actions] += (1.0 - epsilon)

        action = np.random.choice(np.arange(action_length), p=norm_distribute)

        return action

    def restore_buffer(self):
        replay_buffer_size = self.config.replay_buffer_size
        # screen have been resize to 84 * 84
        screen, _, _, terminal = self.env.new_game()
        screen_state = np.stack([screen] * 4, axis=2)

        for i in range(replay_buffer_size):
            action = self.action_choose(screen_state, 1.0)
            self.env.step(action)
            screen_next, reward, terminal = self.env.state
            screen_state_next = np.append(screen_state[:, :, 1:], np.expand_dims(screen_next, 2), axis=2)
            self.replay_buffer.append([screen_state, action, reward, screen_state_next, terminal])

            if terminal:
                screen, _, _, terminal = self.env.new_game()
                screen_state = np.stack([screen] * 4, axis=2)
            else:
                screen_state = screen_state_next

        self.state = screen_state

    def get_epsilons(self, epsilon_start, epsilon_end, t_step, max_step):
        epsilon = epsilon_start + ((max_step - t_step) / max_step) * (epsilon_end - epsilon_start)

        return epsilon

    def train(self):
        # restore buffer
        self.restore_buffer()

        # save model
        saver = tf.train.Saver(max_to_keep=20)

        total_step = 0
        num_epoch = self.config.num_epoch
        max_epsilon_step = self.config.max_epsilon_step
        epsilon_start = self.config.epsilon_start
        epsilon_end = self.config.epsilon_end
        update_target_network_interval = self.config.update_target_network_interval
        replay_buffer_size = self.config.replay_buffer_size
        batch_size = self.config.batch_size
        discont_factor = self.config.discount_factor

        for i_epoch in range(num_epoch):
            self.episode_rewards.append([])
            self.episode_lengths.append([])
            self.episode_losses.append([])
            self.episode_rewards[-1] = 0
            self.episode_rewards[-1] = 0
            self.episode_losses[-1] = 0

            # reset environment
            screen, _, _, _ = self.env.new_game()
            screen_state = np.stack([screen] * 4, axis=2)
            loss = None

            for t in itertools.count():
                epsilon = self.get_epsilons(epsilon_start, epsilon_end, total_step, max_epsilon_step)

                # update target network
                if total_step % update_target_network_interval == 0:
                    self.brain.update_target_network()

                # update replay_buffer
                action = self.action_choose(screen_state, 1.0)
                self.env.step(action)
                screen_next, reward, terminal = self.env.state
                screen_state_next = np.append(screen_state[:, :, 1:], np.expand_dims(screen_next, 2), axis=2)
                if len(self.replay_buffer) == replay_buffer_size:
                    self.replay_buffer.pop(0)

                self.replay_buffer.append([screen_state, action, reward, screen_state_next, terminal])

                # record reward and step in each game
                self.episode_rewards[i_epoch] += reward
                self.episode_rewards[i_epoch] = t

                # sample train sample
                samples = random.sample(self.replay_buffer, batch_size)
                states_batch, action_batch, reward_batch, next_state_batch, terminal_batch = map(np.array, zip(*samples))
                v_predict, _ = self.brain.forward_predict_network(states_batch)

                v_targets, v_target_action = self.brain.forward_predict_network(next_state_batch)
                best_action = np.argmax(v_targets, axis=1)
                v_targets = v_targets[np.arange(batch_size), best_action]
                v_target_batch = reward_batch + np.invert(terminal_batch) * discont_factor * v_targets

                states_batch_pl = states_batch
                action_batch_pl = np.zeros((batch_size, 4))
                action_batch_pl[np.arange(batch_size), action_batch] = 1
                predict_target_pl = v_target_batch

                feed_dict = {self.brain.predict_input_pl: states_batch_pl,
                             self.brain.predict_action_pl: action_batch_pl,
                             self.brain.predict_target_pl: predict_target_pl}

                loss_v, delta_v =  self.brain.sess.run([self.brain.loss, self.brain.delta], feed_dict=feed_dict)
                self.episode_losses[i_epoch] += loss_v

                if total_step % 100 == 0:
                    print("loss: {:4f}, {:}/{:}".format(loss_v, total_step, i_epoch))

                if terminal:
                    break

            avg_loss = self.episode_losses[i_epoch] / self.episode_rewards[i_epoch]
            print("avg_loss: {:4f}, reward: {:}, step: {:}, {:}/{:}".format(avg_loss,
                                                                            self.episode_rewards[i_epoch],
                                                                            self.episode_lengths[i_epoch],
                                                                            i_epoch,
                                                                            num_epoch))