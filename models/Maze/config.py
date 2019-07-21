class AgentConfig(object):
    agent_name = ""
    env_name = ""
    n_rows = 100
    lambbds = 0.9
    epsilon = 0.9
    lr = 0.01


def get_config(FLAGS):
    config = AgentConfig()
    FLAGS = FLAGS.__dict__

    for k in FLAGS.keys():
        v = FLAGS[k]
        if hasattr(config, k):
            setattr(config, k, v)

    return config
