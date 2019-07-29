import numpy as np
from tqdm import tqdm
import time
import itertools
import random
import sys

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
        self.train_batch_time = []

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
        replay_buffer_init_size = self.config.replay_buffer_init_size
        # screen have been resize to 84 * 84
        screen = self.env.new_game()
        screen_state = np.stack([screen] * 4, axis=2)
        print("restore_buffer_starting")
        for i in range(replay_buffer_init_size):
            action = self.action_choose(screen_state, 1.0)
            screen_next, reward, terminal = self.env.step(action)
            screen_state_next = np.append(screen_state[:, :, 1:], np.expand_dims(screen_next, 2), axis=2)
            self.replay_buffer.append([screen_state, action, reward, screen_state_next, terminal])

            if terminal:
                screen = self.env.new_game()
                screen_state = np.stack([screen] * 4, axis=2)
            else:
                screen_state = screen_state_next
        print("restore_buffer_done!")
        self.state = screen_state

        """
        for i in range(replay_buffer_size):
            samples = self.replay_buffer[i]
            states, action, reward, next_state, terminal = samples
            image = visualization_tool(states)
            image_next = visualization_tool(next_state)
            image = np.concatenate([image, image_next], axis=1)
            name = str(i) + '-' + str(action)
            image_path = os.path.join(self.config.imgpath, name + '.png')
            cv2.imwrite(image_path, image)
        """

    def get_epsilons(self, epsilon_start, epsilon_end, t_step, max_step):
        epsilon = epsilon_end + ((max_step - t_step) / max_step) * (epsilon_start - epsilon_end)

        return epsilon

    def train(self):
        # restore buffer
        self.restore_buffer()

        # save model
        saver = tf.train.Saver(max_to_keep=20)

        # tf summary file
        train_writer = tf.summary.FileWriter(self.config.log, self.brain.sess.graph)
        log_writer = open(os.path.join(self.config.log, "log.txt"), "w")

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
            self.train_batch_time.append([])
            self.episode_rewards[-1] = 0
            self.episode_rewards[-1] = 0
            self.episode_losses[-1] = 0
            self.train_batch_time[-1] = 0

            # reset environment
            screen = self.env.new_game()
            screen_state = np.stack([screen] * 4, axis=2)
            loss = None

            for t in itertools.count():
                t1 = time.time()
                epsilon = self.get_epsilons(epsilon_start, epsilon_end, total_step, max_epsilon_step)

                # update target network
                if total_step % update_target_network_interval == 0:
                    self.brain.update_target_network()
                    line = "copy parameters"
                    print(line)
                    log_writer.writelines(line + "\n")

                # update replay_buffer
                action = self.action_choose(screen_state, epsilon)
                screen_next, reward, terminal = self.env.step(action)
                screen_state_next = np.append(screen_state[:, :, 1:], np.expand_dims(screen_next, 2), axis=2)
                if len(self.replay_buffer) == replay_buffer_size:
                    self.replay_buffer.pop(0)

                self.replay_buffer.append([screen_state, action, reward, screen_state_next, terminal])

                # record reward and step in each game
                self.episode_rewards[i_epoch] += reward
                self.episode_lengths[i_epoch] = t

                # sample train sample
                samples = random.sample(self.replay_buffer, batch_size)
                states_batch, action_batch, reward_batch, next_state_batch, terminal_batch = map(np.array, zip(*samples))
                v_predict, _ = self.brain.forward_predict_network(states_batch)

                v_predict_next, _ = self.brain.forward_predict_network(next_state_batch)
                v_targets, v_target_action = self.brain.forward_target_network(next_state_batch)
                best_action = np.argmax(v_predict_next, axis=1)
                v_targets = v_targets[np.arange(batch_size), best_action]
                v_target_batch = reward_batch + np.invert(terminal_batch) * discont_factor * v_targets

                states_batch_pl = states_batch
                action_batch_pl = np.zeros((batch_size, 4))
                action_batch_pl[np.arange(batch_size), action_batch] = 1
                predict_target_pl = v_target_batch

                feed_dict = {self.brain.predict_input_pl: states_batch_pl,
                             self.brain.predict_action_pl: action_batch_pl,
                             self.brain.predict_target_pl: predict_target_pl,
                             self.brain.learning_rate_step: total_step}

                loss_v, delta_v, summary, _ = self.brain.sess.run([self.brain.loss,
                                                                 self.brain.delta,
                                                                 self.brain.merged,
                                                                 self.brain.train_op],
                                                                 feed_dict=feed_dict)
                episode_summary = tf.Summary()
                episode_summary.value.add(simple_value=epsilon, tag="epsilon")
                train_writer.add_summary(episode_summary, total_step)
                train_writer.add_summary(summary, total_step)
                t2 = time.time()
                delta_time = t2 - t1

                self.episode_losses[i_epoch] += loss_v
                self.train_batch_time[i_epoch] += delta_time
                avg_predict_targets = np.mean(predict_target_pl)

                line = "\rloss: {:8f}, {:}/{:}".format(loss_v, total_step, i_epoch)

                if total_step % 10 == 0:
                    print(line, end="")
                    sys.stdout.flush()
                    log_writer.writelines(line + '\n')

                if total_step % 5000 == 0:
                    save_path = saver.save(self.brain.sess, os.path.join(self.config.checkout,
                                                                         "DQN_" + str(total_step) + ".ckpt"))
                    line = "Model save in file: {}".format(save_path)
                    print(line)
                    log_writer.writelines(line + '\n')

                if terminal:

                    break

                total_step += 1

            avg_loss = self.episode_losses[i_epoch] / self.episode_lengths[i_epoch]
            avg_time = self.train_batch_time[i_epoch] / self.episode_lengths[i_epoch]
            avg_line = "avg_loss: {:4f}, reward: {:}, batch_time: {:4f} step: {:}, {:}/{:}".format(avg_loss,
                                                                                                   self.episode_rewards[i_epoch],
                                                                                                   avg_time,
                                                                                                   self.episode_lengths[i_epoch],
                                                                                                   i_epoch,
                                                                                                   num_epoch)
            print(avg_line)
            log_writer.writelines(avg_line + "\n")