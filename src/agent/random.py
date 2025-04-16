import numpy as np


class RandomAgent(TaskAgent):
    def __init__(self, observation_space, action_space, config):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config.agent

    def start_task(self, id, hint):
        pass

    def end_task(self, id):
        pass

    def update(self, id, batch):
        pass

    def sample_action(self, id, obs):
        return self.action_space.sample()
