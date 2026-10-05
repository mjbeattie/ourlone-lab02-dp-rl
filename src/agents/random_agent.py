class RandomAgent:
    def __init__(self, env):
        self.action_space = env.action_space

    def select_action(self, state):
        return self.action_space.sample()
