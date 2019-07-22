class AgentConfig(object):
    agent_name = ""
    env_name = ""
    n_state = 6
    action_name = ['right', 'left']
    epsilon = 0.9
    alpha = 0.1
    lambbds = 0.9
    max_epoch = 13
    fresh_update = 2


def get_config(FLAGS):
    config = AgentConfig()
    FLAGS = FLAGS.__dict__

    for k in FLAGS.keys():
        v = FLAGS[k]
        if hasattr(config, k):
            setattr(config, k, v)

    return config
