class AgentConfig(object):
    # DQN config
    scale = 100
    history_length = 4

    # train config
    batch_size = 32
    num_epoch = 10000
    replay_buffer_size = 5000 * scale
    replay_buffer_init_size = 500 * scale
    max_epsilon_step = 5000 * scale
    epsilon_start = 1.0
    epsilon_end = 0.1
    update_target_network_interval = 100 * scale

    history = 4
    gpu_index = 0
    gpu_fraction = 0.4
    env_name = "PPGDL"
    learning_rate = 0.00025
    learning_rate_minimum = 0.00025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 5 * scale
    discount_factor = 0.99

    # env config
    env_name = 'Breakout-v0'
    agent_name = 'Breakout-v0'
    screen_width = 84
    screen_height = 84
    max_reward = 1
    min_reward = -1
    action_repeat = 1
    random_start = 30
    display = False

    # agent config
    action_length = 4

    # log
    imgpath = "E:\\code\\RL\\Reinforcement-Tutorial\\log_image\\save"
    log = "log"
    checkpoint = "checkpoint"


def get_config(FLAGS):
    config = AgentConfig()

    if isinstance(FLAGS, dict):
        return config

    FLAGS = FLAGS.__dict__

    for k in FLAGS.keys():
        v = FLAGS[k]
        if hasattr(config, k):
            setattr(config, k, v)
        else:
            setattr(config, k, v)

    return config
