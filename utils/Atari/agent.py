class Agent(object):

    def __init__(self, name, config, env, brain):
        self.name = name
        self.config = config
        self.env = env
        self.brain = brain

    def train(self):
        sess = self.brain.sess