import numpy as np
from tqdm import tqdm
import time
import itertools
import random
import sys
import pickle

from utils.Atari.tf_utils import *


class Agent(object):

    def __init__(self, name, config, env, brain):
        self.name = name
        self.config = config
        self.env = env
        self.brain = brain
        self.replay_buffer = []

        self.episode_e_rewards = []
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

        return action, norm_distribute

    def restore_buffer(self):
        name = "replay_memory.pkl"
        if os.path.exists(name):
            f = open(name, 'rb')
            tmp = pickle.load(f)
            for i in range(len(tmp)):
                self.replay_buffer.append(tmp[i])
        else:
            replay_buffer_init_size = self.config.replay_buffer_init_size
            # screen have been resize to 84 * 84
            screen = self.env.new_game()
            screen_state = np.stack([screen] * 4, axis=2)
            print("restore_buffer_starting")
            for i in range(replay_buffer_init_size):
                action, _ = self.action_choose(screen_state, 1.0)
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

            if not os.path.exists(name):
                f1 = open(name, 'wb')
                pickle.dump(self.replay_buffer, f1)

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
        if t_step < max_step:
            epsilon = epsilon_end + ((max_step - t_step) / max_step) * (epsilon_start - epsilon_end)
        else:
            epsilon = epsilon_end

        return epsilon

    def train(self):
        # restore buffer
        self.restore_buffer()
       
        #name = r"D:\PDD\code\Reinforcement-Demo\utils\Atari\replay_memory.pkl"
        #if os.path.exists(name):
        #    f = open(name, 'rb')
        #    tmp = pickle.load(f)
        #    for i in range(len(tmp)):
        #        ts = []
        #        ts.append(tmp[i]['state'])
        #        ts.append(tmp[i]['action'])
        #        ts.append(tmp[i]['reward'])
        #        ts.append(tmp[i]['next_state'])
        #        ts.append(tmp[i]['done'])
        #        self.replay_buffer.append(ts)

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
            self.episode_e_rewards.append([])
            self.episode_rewards.append([])
            self.episode_lengths.append([])
            self.episode_losses.append([])
            self.train_batch_time.append([])
            self.episode_rewards[-1] = 0
            self.episode_rewards[-1] = 0
            self.episode_losses[-1] = 0
            self.train_batch_time[-1] = 0
            self.episode_e_rewards[-1] = 0

            # reset environment
            screen = self.env.new_game()
            screen_state = np.stack([screen] * 4, axis=2)
            distribution = None

            for t in itertools.count():
                t1 = time.time()
                epsilon = self.get_epsilons(epsilon_start, epsilon_end, total_step, max_epsilon_step)

                load_model_from_npy = True
                if load_model_from_npy and t == 0:
                    model_weight = r"D:\PDD\code\Reinforcement-Demo\utils\Atari\model_weight.npy"
                    if os.path.exists(model_weight) and t == 0:
                        tmp = np.load(model_weight).item()
                        e1_params = [t for t in tf.trainable_variables() if t.name.startswith('predict')]
                        for param in e1_params:
                            for key in tmp.keys():
                                new_key = key.replace("q", "predict")
                                if param.name == new_key:
                                    self.brain.sess.run(param.assign(tmp[key]))
                                    print("Assign param: {:} from {:}".format(param.name, key))

                # update target network
                if total_step % update_target_network_interval == 0:
                    self.brain.update_target_network()
                    line = " copy parameters"
                    print(line)
                    log_writer.writelines(line + "\n")

                # update replay_buffer
                action, distribution = self.action_choose(screen_state, epsilon)
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
                raw_v_targets, v_target_action = self.brain.forward_target_network(next_state_batch)
                best_action = np.argmax(v_predict_next, axis=1)
                v_targets = raw_v_targets[np.arange(batch_size), best_action]
                v_target_batch = reward_batch + np.invert(terminal_batch) * discont_factor * v_targets

                states_batch_pl = states_batch
                action_batch_pl = np.zeros((batch_size, 4))
                action_batch_pl[np.arange(batch_size), action_batch] = 1
                predict_target_pl = v_target_batch

                feed_dict = {self.brain.predict_input_pl: states_batch_pl,
                             self.brain.predict_action_pl: action_batch_pl,
                             self.brain.predict_target_pl: predict_target_pl,
                             self.brain.learning_rate_step: total_step}

                loss_v, delta_v, summary, _, gradient, predict_tensor_list \
                    = self.brain.sess.run([self.brain.loss, self.brain.delta,
                                           self.brain.merged, self.brain.train_op,
                                           self.brain.gradients, self.brain.predict_tensor_list],
                                                                 feed_dict=feed_dict)
                episode_summary = tf.Summary()
                episode_summary.value.add(simple_value=epsilon, tag="epsilon")

                if total_step % 50 == 0:
                    train_writer.add_summary(episode_summary, total_step)
                    train_writer.add_summary(summary, total_step)

                t2 = time.time()
                delta_time = t2 - t1

                self.episode_losses[i_epoch] += loss_v
                self.episode_e_rewards[i_epoch] += np.mean(np.max(v_predict, axis=1))
                self.train_batch_time[i_epoch] += delta_time

                line = "\rloss: {:8}, max_predict: {:4}, reward: {:4}, v_targets: {:4}, {:}/{:}".\
                    format(loss_v, str(np.max(v_predict)),
                           str(np.max(reward)),
                           str(np.max(v_targets)),
                           total_step, i_epoch)

                if total_step % 1 == 0:
                    print(line, end="")
                    sys.stdout.flush()
                    log_writer.writelines(line + '\n')

                if total_step % 5000 == 0:
                    save_path = saver.save(self.brain.sess, os.path.join(self.config.checkout,
                                                                         "DQN_" + str(total_step) + ".ckpt"))
                    line = "\nModel save in file: {}".format(save_path)
                    print(line)
                    log_writer.writelines(line + '\n')

                if terminal:

                    break

                total_step += 1

            avg_loss = self.episode_losses[i_epoch] / self.episode_lengths[i_epoch]
            avg_time = self.train_batch_time[i_epoch] / self.episode_lengths[i_epoch]
            avg_line = "\navg_loss: {:4f}, reward: {:}, batch_time: {:4f} step: {:}, distribution: {:} {:} {:}/{:}".format(avg_loss,
                                                                                                   self.episode_rewards[i_epoch],
                                                                                                   avg_time,
                                                                                                   self.episode_lengths[i_epoch],
                                                                                                   str(distribution),
                                                                                                   total_step,
                                                                                                   i_epoch,
                                                                                                   num_epoch)
            print(avg_line)
            log_writer.writelines(avg_line + "\n")
