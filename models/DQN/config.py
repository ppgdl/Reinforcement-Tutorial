class AgentConfig(object):
    # traininig config
    batch_size = 32
    num_epoch = 10000
    replay_buffer_size = 10000
    history = 4
    gpu_index = 0
    gpu_fraction = 0.7
    env_name = "PPGDL"
    lr = 0.001

    # env config
    env_name = 'Breakout-v0'
    screen_width = 84
    screen_height = 84
    max_reward = 1
    min_rewars = -1
    actoin_repeat = 1
    random_start = 30
    display = True



def get_config(FLAGS):
    config = AgentConfig()
    FLAGS = FLAGS.__dict__

    for k in FLAGS.keys():
        v = FLAGS[k]
        if hasattr(config, k):
            setattr(config, k, v)

    return config